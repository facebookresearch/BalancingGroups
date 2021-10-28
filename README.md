# BalancingGroups

Code to replicate the experimental results from [Simple data balancing baselines achieve competitive worst-group-accuracy](https://arxiv.org/abs/2110.14503).

## Replicating the main results

### Installing dependencies

Easiest way to have a working environment for this repo is to create a conda environement with the following commands

```bash
conda env create -f environment.yaml
conda activate balancinggroups
```	

If conda is not available, please install the dependencies listed in the requirements.txt file.

### Download, extract and Generate metadata for datasets

This script downloads, extracts and formats the datasets metadata so that it works with the rest of the code out of the box.

```bash
python setup_datasets.py --download --data_path data
```

### Launch jobs

```bash
# Launching 1400 combo seeds = 50 hparams for 4 datasets for 7 algorithms
# Each combo seed is ran 5 times to compute error bars, totalling 7000 jobs
python train.py --data_path data --output_dir main_sweep --num_hparams_seeds 1400 --num_init_seeds 5
```

### Parse results

The parse.py script can generate all of the plots and tables from the paper. 
By default, it generates the best test worst-group-accuracy table for each dataset/method.
This script can be called while the experiments are still running. 

```bash
python parse.py main_sweep
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).
