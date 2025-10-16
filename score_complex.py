import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import biotite
import esm
from esm.inverse_folding.util import get_sequence_loss
from esm.inverse_folding.multichain_util import (
    score_sequence_in_complex,
    extract_coords_from_complex,
    _concatenate_coords
)

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Protein sequence scoring with ESM-IF.')
    
    # String arguments
    parser.add_argument('--weights_path', type=str, required=False, 
                        help='Path to weights for the model')
    parser.add_argument('--device', type=str, default='0', required=False, 
                        help='Which CUDA device to use')
    parser.add_argument('--dataset', type=str, default='skempi', required=False, 
                        help='Dataset type: ab-bind or skempi')
    parser.add_argument('--feature', type=str, default='affinity', required=False, 
                        help='Name of fitness value to score against')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    parser.add_argument('--out_path', type=str, required=False,
                        default='scores_complex.csv',
                        help='Path for the output CSV file')
    
    # Binary/flag arguments
    parser.add_argument('--vanilla', action='store_true', 
                        help='Use the vanilla model weights (no fine-tuning)')
    parser.add_argument('--whole_seq', action='store_true', 
                        help='Score using the whole sequence rather than just mutation sites')
    parser.add_argument('--no_mutations', action='store_true', 
                        help='Process sequences without mutation data')
    parser.add_argument('--normalize', action='store_true', 
                        help='Normalize scores by subtracting wild type score')
    parser.add_argument('--delta', action='store_true', 
                        help='Take difference in mutant and wild-type affinity (only for skempi)')
    parser.add_argument('--sf', action='store_true', 
                        help='Subtract wild-type log-likelihood from mutant log-likelihood before averaging')
    parser.add_argument('--sum', action='store_true', 
                        help='Sum log-likelihoods instead of averaging (only for no_mutations mode)')
    
    return parser.parse_args()


def main():
    """Main function to run the sequence scoring pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Load model
    print('Loading Model')
    model, alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm_if1_gvp4_t16_142M_UR50.pt')

    if args.weights_path is not None:
        state_dict = torch.load(args.weights_path)
        model.load_state_dict(state_dict, strict = True)

    # Set CUDA device and determine if GPU is available
    if torch.cuda.is_available():
        device = f'cuda:0'
        model = model.to(device)
        print('Running on GPU')
    else:
        device = 'cpu'
        print('Running on CPU')
    
    # Load dataset
    print('Loading Data')
    dataset = pd.read_csv(args.dataset_path)
    
    # Initialize output dictionary
    outputs_dict = {'vals': [], 'scores': []}
    
    # Score sequences
    print('Scoring Sequences')
    pdb_grouped = dataset.groupby('WT_name')
    model.eval()

    for pdb_path, pdb_group_df in tqdm(pdb_grouped):

        # Extract structure information
        chains = pdb_group_df.iloc[0]['chains']
        structure = esm.inverse_folding.util.load_structure(pdb_path)
        # Filter out hetero atoms (non-protein atoms like ligands, water)
        structure = biotite.structure.array([atom for atom in structure if not atom.hetero])
        all_coords, _ = extract_coords_from_complex(structure)
        
        # Process each row in the group
        for _, row in pdb_group_df.iterrows():
            # Handle cases with mutation information
            if not args.no_mutations:
                # Parse mutation information
                mutation_info = [(mut[0], mut[1], mut[-1]) for mut in row['muts'].split(',') if mut != '']
                mut_idxs = [int(idx) for idx in row['mut_chain_idx'].split(',') if idx != '']
                mutation_info_and_idx = list(zip(mutation_info, mut_idxs))
                
                lls = []  # Log-likelihoods for mutations
                wt_lls = []  # Wild-type log-likelihoods if normalizing
                
                # Score each mutation
                for mutation_info, mut_idx in mutation_info_and_idx:
                    wt_res, mutated_chain, mut_res = mutation_info
                    mutant_chain_seq = row[mutated_chain]
                    # Construct wild-type sequence
                    wt_chain = mutant_chain_seq[:mut_idx] + wt_res + mutant_chain_seq[mut_idx+1:]
                    
                    # Score based on whole sequence or specific site
                    if args.whole_seq:
                        # Score entire sequence
                        ll, _ = score_sequence_in_complex(model, alphabet, all_coords, 
                                                         mutated_chain, mutant_chain_seq)
                        if args.normalize:
                            # Calculate wild-type score if normalizing
                            wt_ll, _ = score_sequence_in_complex(model, alphabet, all_coords, 
                                                                mutated_chain, wt_chain)
                            wt_lls.append(wt_ll)
                        
                        # Subtract wild-type score if sf flag is set
                        if args.sf:
                            ll -= wt_ll
                    else:
                        # Score specific mutation site
                        coords = _concatenate_coords(all_coords, mutated_chain)
                        loss, _ = get_sequence_loss(model, alphabet, coords,
                                                                    mutant_chain_seq)
                        ll = -loss[mut_idx]  # Log-likelihood is negative of loss
                        
                        if args.normalize:
                            # Calculate wild-type score if normalizing
                            wt_loss, _ = get_sequence_loss(model, alphabet, 
                                                                           coords, wt_chain)
                            wt_ll = -wt_loss[mut_idx]
                            wt_lls.append(wt_ll)
                        
                        # Subtract wild-type score if sf flag is set
                        if args.sf:
                            ll -= wt_ll
                    
                    lls.append(ll)
                
                # Calculate average log-likelihood
                avg_ll = np.average(lls)
                
                # Normalize if needed (and not already done with sf)
                if args.normalize and not args.sf:
                    avg_ll -= np.average(wt_lls)
            
            # Handle cases without mutation information
            else:
                lls = []
                # Score each chain
                for chain in chains:
                    if args.sum:
                        # Sum log-likelihoods across the sequence
                        coords = _concatenate_coords(all_coords, chain)
                        ll, _ = get_sequence_loss(model, alphabet, coords, row[chain])
                        lls.append(np.nansum(ll))
                    else:
                        # Calculate average log-likelihood
                        ll, _ = score_sequence_in_complex(model, alphabet, all_coords, 
                                                         chain, row[chain])
                        lls.append(ll)
                
                # Calculate average across chains
                avg_ll = np.average(lls)
            
            # Store results
            outputs_dict['vals'].append(row[args.feature])
            outputs_dict['scores'].append(avg_ll)
    
    # Save results to CSV
    out_df = pd.DataFrame(outputs_dict)
    out_df.to_csv(args.out_path)
    print(f"Results saved to {args.out_path}")


if __name__ == "__main__":
    main()