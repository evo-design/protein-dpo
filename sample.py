
import argparse
import numpy as np
from pathlib import Path
import torch
import esm
import esm.inverse_folding
import os

def sample_seq_singlechain(model, alphabet, args):
    coords, seq = esm.inverse_folding.util.load_coords(args.pdbfile, args.chain)
    partial_seq = ['<mask>'] * len(seq)

    if args.fixed_pos is not None:
        for fixed_idx in fixed:
            partial_seq[fixed_idx - 1] = seq[fixed_idx - 1]
            
    print(f'Saving sampled sequences to {args.outpath}.')

    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outpath, 'w') as f:
        for i in range(args.num_samples):
            print(f'\nSampling.. ({i+1} of {args.num_samples})')
            sampled_seq = model.sample(coords, partial_seq = partial_seq, temperature=args.temperature, device=torch.device('cuda'))
            print('Sampled sequence:')
            print(sampled_seq)
            f.write(f'>sampled_seq_{i+1}\n')
            f.write(sampled_seq + '\n')

            recovery = np.mean([(a==b) for a, b in zip(seq, sampled_seq)])
            print('Sequence recovery:', recovery)

def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
            '--pdbfile', type=str,
            help='input filepath, either .pdb or .cif',
    )
    parser.add_argument(
            '--chain', type=str,
            help='chain id for the chain of interest', default=None,
    )
    parser.add_argument(
            '--temperature', type=float,
            help='temperature for sampling, higher for more diversity',
            default=1.,
    )
    parser.add_argument(
            '--outpath', type=str,
            help='output filepath for saving sampled sequences',
            default='output/sampled_seqs.fasta',
    )
    parser.add_argument(
            '--num-samples', type=int,
            help='number of sequences to sample',
            default=1000,
    )
    parser.add_argument(
        '--weights_path', type=str, default=None', 
        required =False, help='path'
    )
    parser.add_argument(
        '--fixed_pos', type=str, default=None', 
        required =False, help='path'
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    model, alphabet = esm.pretrained.load_model_and_alphabet_local('weights/esm_if1_gvp4_t16_142M_UR50.pt')
    model.eval()
    model.to('cuda')

    if args.weights_path is not None:
        state_dict = torch.load(args.weights_path)
        model.load_state_dict(state_dict, strict = True)   
    
    sample_seq_singlechain(model, alphabet, args)
