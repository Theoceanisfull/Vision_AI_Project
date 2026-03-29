# SCNN Base-Encoding Comparison

This folder contains the Conv-SNN training stack for ASL-DVS and a workflow to compare three input encodings:

- `rate`
- `latency`
- `delta`

## Prerequisites

From repo root:

```bash
python3 -m pip install snntorch torch scipy matplotlib pandas
```

Dataset path expected by default:

- `data/ASLDVS`

## Files

- `config.py`: typed config objects
- `model.py`: Conv-SNN architectures (`base` and `deep`)
- `spike_data.py`: ASL-DVS loader + encoding transforms
- `t_loop.py`: train/eval loop (`manual` and `snntorch.backprop` modes)
- `train_three_base.py`: trains `base` model on all 3 encodings
- `configs/base_rate.json`, `configs/base_latency.json`, `configs/base_delta.json`: starter configs

## Run 3-Model Base Comparison

Full run (rate + latency + delta):

```bash
python3 -m models.scnn.train_three_base
```

Quick smoke run:

```bash
python3 -m models.scnn.train_three_base --quick
```

Outputs:

- `runs/scnn/base_rate/` (or `base_rate_quick/`)
- `runs/scnn/base_latency/` (or `base_latency_quick/`)
- `runs/scnn/base_delta/` (or `base_delta_quick/`)
- `runs/scnn/base_encoding_comparison/summary.json`

## Run a Single Training Config

Write default config:

```bash
python3 -m models.scnn.t_loop --write-default-config models/scnn/default_config.json
```

Train from config:

```bash
python3 -m models.scnn.t_loop --config models/scnn/default_config.json
```

Quick smoke:

```bash
python3 -m models.scnn.t_loop --quick-smoke
```

## Compare in Notebook

Open:

- `notebooks/compare_base_models_encodings.ipynb`

The notebook reads:

- `runs/scnn/base_encoding_comparison/summary.json` (if present)
- otherwise fallback run dirs:
  - `runs/scnn/base_rate`
  - `runs/scnn/base_latency`
  - `runs/scnn/base_delta`

It plots:

- loss curves
- accuracy curves
- spike count curves
- bar charts for test metrics
- confusion matrices

## Notes

- `runs/` is ignored by Git in `.gitignore`, so checkpoints/plots are not pushed.
- Full training can take significant time on CPU; use `train.mode`, `epochs`, and batch limits in config as needed.
