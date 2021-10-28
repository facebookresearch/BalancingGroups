# BalancingGroups

Code to replicate the experimental results from [Simple data balancing baselines achieve competitive worst-group-accuracy](https://arxiv.org/abs/2110.14503).

## Replicating the main results

### Set environment variables

```bash
export DATASETS_PATH=/path/to/datasets
export SLURM_PATH=/path/to/slurm/logs
```

### Download and extract datasets

* [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) to `$DATASETS_PATH/waterbirds`
* [CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset) to `$DATASETS_PATH/celeba`
* [CivilComments](https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/) to `$DATASETS_PATH/civilcomments`
* [MultiNLI](https://github.com/kohpangwei/group_DRO#multinli-with-annotated-negations) to `$DATASETS_PATH/multinli`

### Generate dataset metadata

```bash
cd metadata/
python generate_metadata_waterbirds.py
python generate_metadata_celeba.py
python generate_metadata_civilcomments.py
python generate_metadata_multinli.py
cd ..
```

### Launch jobs

```bash
# Launching 1400 combo seeds = 50 hparams for 4 datasets for 7 algorithms
# Each combo seed is ran 5 times to compute error bars, totalling 7000 jobs
./train.py --output_dir main_sweep --num_hparams_seeds 1400 
```

### Parse results

```bash
./parse.py main_sweep
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).
