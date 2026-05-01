# SCNN Base-Encoding Comparison

This folder contains the Conv-SNN training stack for ASL-DVS and workflows to compare three input encodings across both base and deep Conv-SNN variants:

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
- `train_three_base.py`: trains the `base` model on all 3 encodings
- `train_three_deep.py`: trains the `deep` model on all 3 encodings
- `configs/base_rate.json`, `configs/base_latency.json`, `configs/base_delta.json`: starter configs for base runs
- `configs/deep_rate.json`, `configs/deep_latency.json`, `configs/deep_delta.json`: starter configs for deep runs

## Run 3-Model Base Comparison

Full run (rate + latency + delta):

```bash
python3 -m models.scnn.train_three_base
```

By default this command now spawns a detached background worker, writes a log under
`runs/scnn/launch/`, prints the PID, and returns immediately so the job keeps running
after you close the terminal.

Foreground run for debugging:

```bash
python3 -m models.scnn.train_three_base --foreground
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
- each run directory now includes both `checkpoint_last.pt` and `checkpoint_best.pt`

## Run 3-Model Deep Comparison

Full run (rate + latency + delta, deep architecture, +5 epochs over the base config by default):

```bash
python3 -m models.scnn.train_three_deep
```

Foreground run for debugging:

```bash
python3 -m models.scnn.train_three_deep --foreground
```

Quick smoke run:

```bash
python3 -m models.scnn.train_three_deep --quick
```

Outputs:

- `runs/scnn/deep_rate/` (or `deep_rate_quick/`)
- `runs/scnn/deep_latency/` (or `deep_latency_quick/`)
- `runs/scnn/deep_delta/` (or `deep_delta_quick/`)
- `runs/scnn/deep_encoding_comparison/summary.json`
- each run directory now includes both `checkpoint_last.pt` and `checkpoint_best.pt`

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
- `notebooks/compare_deep_models_encodings.ipynb`

The notebook reads:

- `runs/scnn/base_encoding_comparison/summary.json` (if present)
- otherwise fallback run dirs:
  - `runs/scnn/base_rate`
  - `runs/scnn/base_latency`
  - `runs/scnn/base_delta`

The deep notebook reads:

- `runs/scnn/deep_encoding_comparison/summary.json` (if present)
- otherwise fallback run dirs:
  - `runs/scnn/deep_rate`
  - `runs/scnn/deep_latency`
  - `runs/scnn/deep_delta`

It plots:

- loss curves
- accuracy curves
- spike count curves
- bar charts for test metrics
- confusion matrices

## Notes

- `runs/` is ignored by Git in `.gitignore`, so checkpoints/plots are not pushed.
- Full training can take significant time on CPU; use `train.mode`, `epochs`, and batch limits in config as needed.
- Sweep runs now apply encoding-aware presets:
  - `rate`: time-varying Poisson rate coding with rate-based loss/accuracy
  - `latency`: static activity map with temporal loss/accuracy
  - `delta`: signed temporal delta coding with lower LR, lower input threshold, and a lower neuron threshold
