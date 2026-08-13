# Your 5-Core Train Split May Not Be 5-Core

This repository contains code for the paper:

**Your 5-Core Train Split May Not Be 5-Core: Core Filtering and Temporal
Splitting Do Not Commute**

Temporal recommender papers often state that they apply p-core filtering and use
a temporal split, but that does not define a unique benchmark. The paper compares
two protocol orders: V1 filters the full log before splitting, while V2 splits,
removes validation events, filters the fitted train split, and then projects
held-out data to the fitted-train user/item universe.


## Environment

Python 3.10 or newer is recommended. For the CSV-level checks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Classical baselines and split generation run on CPU with these dependencies.
For neural reruns, install a CUDA-capable PyTorch build matching the local
driver, or start from:

```bash
pip install -r requirements-neural.txt
```


## Regenerating Splits

Place raw interaction CSVs under `exps/data/raw/` using the filenames expected
by `exps/runs/configs/dataset/*.yaml`. Then set the data root:

```bash
export SEQ_SPLITS_DATA_PATH="$PWD/exps/data"
```

Example V1/V2 p-core split generation for Beauty p=5:

```bash
python exps/runs/variant_pipeline.py \
  --dataset Beauty \
  --filter-type pcore \
  --p 5 \
  --order filter_then_split \
  --quantile 0.9 \
  --validation-type last_train_item \
  --drop-conseq-repeats

python exps/runs/variant_pipeline.py \
  --dataset Beauty \
  --filter-type pcore \
  --p 5 \
  --order split_then_filter_train \
  --quantile 0.9 \
  --validation-type last_train_item \
  --drop-conseq-repeats
```

The outputs are written to:

```text
exps/data/splitted/global_timesplit/val_last_train_item/<variant>/q09/
```

## Running Models

Classical baselines for generated variants:

```bash
python exps/runs/train_baselines.py \
  --data-path "$PWD/exps/data" \
  --datasets Beauty \
  --variant-prefix Beauty__pcore_p5 \
  --models MostPopular,Markov,SKNN,ItemKNN,EASER \
  --skip-existing
```

Neural model example:

```bash
SEQ_SPLITS_DATA_PATH="$PWD/exps/data" python exps/runs/train.py \
  dataset=Beauty \
  split_type=global_timesplit \
  split_subtype=val_last_train_item \
  model=SASRec \
  seed=17
```

The model-training script uses Hydra-style overrides inherited from the included
configuration files.

## Recomputing Paper Diagnostics

After split and metric files are regenerated, run:

```bash
python scripts/filter_split_analysis.py
python scripts/additional_diagnostics.py
python scripts/common_test_probe.py --scope all
python scripts/seed_check_summary.py
python scripts/write_final_summary.py
```

Some robustness scripts require saved neural per-user outputs or checkpoints.


## Citation

If you find our work helpful, please consider citing the paper:

```bibtex
@inproceedings{gusak2026core,
  title={Your {5-Core} Train Split May Not Be {5-Core}: Core Filtering and Temporal Splitting Do Not Commute},
  author={Gusak, Danil and Frolov, Evgeny},
  booktitle={Proceedings of the 20th ACM Conference on Recommender Systems},
  doi={10.1145/3773078.3841280},
  year={2026}
}
```