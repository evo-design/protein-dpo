import esm
import torch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from esm.inverse_folding.util import score_sequence, get_sequence_loss, load_structure, extract_coords_from_structure

def add_masked_coords(coord, seq):
    """Handle masked coordinates for residues marked with 'X'"""
    mask_idx = np.array([res == 'X' for res in seq])
    new_coords = np.full((len(seq), 3, 3), np.inf, dtype=coord.dtype)
    new_coords[~mask_idx] = coord
    return new_coords

# Parse command line arguments
parser = argparse.ArgumentParser(description='Scoring Script for evaluating likelihoods of protein variants.')
parser.add_argument('--weights_path', type=str, required=False, help='path to model weights')
parser.add_argument('--dataset_path', type=str, required=True, help='path to csv containing sequences/mutations')
parser.add_argument('--feature', type=str, required=False, help='name of fitness value')
parser.add_argument('--normalize', action='store_true', help='whether or not to normalize by subtracting wild-type')
parser.add_argument('--whole_seq', action='store_true', help='whether or not to use whole seq instead of mut pos')
parser.add_argument('--sum', action='store_true', help='if scoring whole seq whether or not to sum likelihoods instead of average')
parser.add_argument('--out_path', type=str, default='results.csv', required=False)
args = parser.parse_args()

# Initialize model
print('Loading Model')
model, alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm_if1_gvp4_t16_142M_UR50.pt')

# Load custom weights if provided
if args.weights_path is not None:
    state_dict = torch.load(args.weights_path)
    model.load_state_dict(state_dict, strict=True)

# Set device (GPU or CPU)
if torch.cuda.is_available():
    device = f'cuda:0'
    model = model.to(device)
else:
    device = 'cpu'

# Load dataset
print('Loading Data')
dataset = pd.read_csv(args.dataset_path)

# Initialize output dictionary
outputs_dict = {'vals': [], 'scores': []}

# Score sequences
print('Scoring Sequences')
pdb_grouped = dataset.groupby('WT_name')
model.eval()

for pdb_path, group_df in tqdm(pdb_grouped, total=len(pdb_grouped)):
    # Load structure and extract coordinates
    structure = load_structure(pdb_path)
    coord, _ = extract_coords_from_structure(structure)
    
    for seq, feature, muts in zip(group_df['aa_seq'], group_df[args.feature], group_df['mut_type']):
        # Parse mutation information
        mutation_info = [(mut[0], int(mut[1:-1]), mut[-1]) for mut in muts.split(':') if muts != 'wt']

        # Verify mutation indices are correct
        for wt_res, mut_idx, mut_res in mutation_info:
            assert mut_res == seq[mut_idx], 'mutation idx is incorrect'

        # Handle missing coordinates in structure
        if 'X' in seq:
            coord = add_masked_coords(coord, seq)
        
        # Generate wild-type sequence if normalizing
        if args.normalize:
            wt_lls = []
            wt_seq = seq
            for wt_res, mut_idx, mut_res in mutation_info:
                assert mut_res == wt_seq[mut_idx], 'mutation idx is incorrect'
                wt_seq = wt_seq[:mut_idx] + wt_res + wt_seq[mut_idx + 1:]

        lls = []
        # Score using whole sequence likelihood
        if args.whole_seq:
            if args.normalize:
                if args.sum:
                    wt_loss, _ = get_sequence_loss(model, alphabet, coord, wt_seq)
                    wt_lls.append(np.nansum(wt_loss))
                else:
                    wt_ll = score_sequence(model, alphabet, coord, wt_seq)
                    wt_lls.append(wt_ll)
            
            if args.sum:
                loss, _ = get_sequence_loss(model, alphabet, coord, seq)
                lls.append(np.nansum(loss))
            else:
                ll, _ = score_sequence(model, alphabet, coord, seq)
                lls.append(ll)

        # Score using mutation sites only
        else:           
            if args.normalize:
                wt_loss, _ = get_sequence_loss(model, alphabet, coord, wt_seq)
                for wt_res, mut_idx, mut_res in mutation_info:
                    assert len(wt_loss) == len(seq) == len(wt_seq), 'mismatch in length'
                    wt_lls.append(-wt_loss[mut_idx])

            loss, _ = get_sequence_loss(model, alphabet, coord, seq)
            for _, mut_idx, _ in mutation_info:
                assert len(loss) == len(seq), 'mismatch in length'
                lls.append(-loss[mut_idx])
        
        # Calculate average log-likelihood
        avg_ll = np.mean(lls)
        
        # Normalize if requested
        if args.normalize:
            avg_ll = avg_ll - np.mean(wt_lls)

        # Store results
        outputs_dict['vals'].append(feature)
        outputs_dict['scores'].append(avg_ll)
        
    # Save results
    out_df = pd.DataFrame(outputs_dict).T
    out_df.to_csv(args.out_path)