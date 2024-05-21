# protein-dpo

<img width="875" alt="Screenshot 2024-05-20 at 10 34 10 AM" src="https://github.com/evo-design/protein-dpo/assets/63631602/66f67770-0d84-4a28-aa12-36bc2fbc9a52">

## Introduction

This repository holds inference and training code for ProteinDPO (**P**rotein **D**irect **P**reference **O**ptimization), a preference optimized structure-conditioned protein language model based on [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/esm/inverse_folding). We describe ProteinDPO in the paper [“Aligning Protein Generative Models with Experimental Fitness via Direct Preference Optimization”](https://www.biorxiv.org/content/early/2024/05/21/2024.05.20.595026).


## Getting Started

1. Clone this repository:

```
git clone https://github.com/evo-design/protein-dpo.git
```
2. Navigate to the repository directory:

```
cd protein-dpo
```

3. Use conda to install required dependencies:

Use the `environment.yml` file provided in this repository to create and activate a Conda environment with all the necessary dependencies.

```
conda env create -f environment.yml
conda activate <environment_name>
```

4. Download Model Weights

Download Protein DPO model weights from the [Zenodo Repository](https://doi.org/10.5281/zenodo.11218181) and instert them in the `weights` folder.

Download vanilla ESM-IF1 model weights within the `weights` directory with the following commands:

```
cd weights/
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt
```

## Sampling

Sampling is simply a slightly modified script from the [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/esm/inverse_folding) github. Note, stabilization of any protein backbone with ProteinDPO is not guaranteed to preserve its function, thus we strongly recommend functional or heavily conserved residues be preserved with the `--fixed_pos` argument.

1. Run The Sampling Script

```
python sample.py --pdbfile <path_to_input_pdb> --weights_path <path_to_model_weights> [additional_arguments]
```

If no `weights_path` is provided the scripts defaults to the vanilla model weights.

Additional arguments:

```
--temperature: sampling temperature, lower temperature sampling will have lower diversity
--outpath: path for sampled sequence output
--num-samples: desired number of samples
--fixed_pos: positions to fix for sampling, first residue is 1 not 0
```

## Scoring

1. Prepare your dataset:

```
aa_seq : Amino acid sequence of mutant variant
WT_Name : Path to the native PDB file 
<feature> : Scalar label of the feature for optimization
wt_seq: Amino acid sequence of the native sequence
mut_type: comma seperated string of <native_aa>:<pos>:<mutant_aa>,<native_aa>:<pos>:<mutant_aa>,... etc.
```

2. Run Scoring Script

```
python score.py --dataset_path <path_to_sequences_csv> --weights_path <path_to_model_weights> [additional_arguments]
```

Replace `<path_to_model_weights>` with the path to the trained protein-dpo model or any ESM-IF1 compatible weights of your choice. If no `weights_path` is provided the script defaults to the vanilla model weights.

Additional arguments:

```
--normalize: pass if you want to normalize likelihood with wild-type sequence
--whole_seq: pass if you want to utilize liklihood of entire sequence, not just mutated residue(s)
--sum: pass if you want to sum likelihoods instead of averaging
--out_path: path for output csv
```

3. Analyze Results

Located at the path given by the `--out_path` argument will be a csv containing the specified model likelihood for each sequence.

## Citation

Please cite the following preprint when referencing ProteinDPO.

@article {widatalla2024aligning,
	author = {Widatalla, Talal and Rafailov, Rafael and Hie, Brian},
	title = {Aligning protein generative models with experimental fitness via Direct Preference Optimization},
	year = {2024},
	doi = {10.1101/2024.05.20.595026},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/05/21/2024.05.20.595026},
	journal = {bioRxiv}
}


## License
This project is licensed under the MIT License - see the LICENSE file for details.

