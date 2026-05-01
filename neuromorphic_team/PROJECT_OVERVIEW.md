# Vision AI Project Overview

## What This Project Does

This project studies neuromorphic sign-language recognition on the ASL-DVS dataset and compares two event-based model families:

- `scnn/`: a Conv-SNN baseline using `snntorch`
- `event2vec/`: an Event2Vec-style token model for event classification

It also includes:

- `scripts/`: dataset setup, result analysis, architecture visualization, EventGAN-to-Event2Vec evaluation, and demo video export
- `Analytics/`: generated charts, tables, videos, and written summaries
- `notebooks/`: comparison and inspection notebooks
- `EventGAN/`: vendored EventGAN code used to generate synthetic event streams from ASL images
- `Kohl/`: older standalone image/video baseline experiments that are separate from the current ASL-DVS event-comparison pipeline

## End-to-End Process

1. Set up ASL-DVS under `data/ASLDVS`.
   The dataset setup flow is documented in [DATASET_SETUP.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/DATASET_SETUP.md:1) and automated in [setup_asldvs_for_tonic.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scripts/setup_asldvs_for_tonic.py:1).

2. Load raw event files.
   Both stacks read `.mat` event files with `x`, `y`, `ts`, and `pol`, apply a vertical flip to `y`, and create train/val/test splits per class.

3. Build one of two input representations.
   `scnn/` converts events into spike tensors `[T, 2, H, W]`.
   `event2vec/` converts events into sparse tokens `[x, y, t, p, rho]` after binning, pooling, encoding, and token pruning.

4. Train comparison runs across `rate`, `latency`, and `delta`.
   `scnn/scnn/train_three_base.py` and `scnn/scnn/train_three_deep.py` sweep the Conv-SNN family.
   `event2vec/train_three.py` sweeps the Event2Vec family.

5. Save artifacts under `runs/`.
   Each run writes metrics, history, plots, and checkpoints. The best checkpoint is selected by validation accuracy, then validation loss.

6. Aggregate and visualize results.
   [analyze_latest_runs.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scripts/analyze_latest_runs.py:1) builds CSVs, plots, and markdown summaries. The main human-readable outputs are in `Analytics/`.

7. Optional synthetic-data extension.
   [eventgan_asl_event2vec.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scripts/eventgan_asl_event2vec.py:1) uses EventGAN to generate events from ASL images, then classifies those generated events with Event2Vec and exports charts/videos.

## Model Architectures

### 1. Conv-SNN Baseline

Defined in [scnn/scnn/model.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scnn/scnn/model.py:40).

- Input: encoded spike volume `[B, T, 2, H, W]`
- Core block: `Conv2d -> BatchNorm -> LIF -> MaxPool`
- `base` model: 2 conv stages with channels `(32, 64)`
- `deep` model: 3 conv stages with channels `(32, 64, 128)`
- Head: flatten -> dropout -> linear -> LIF -> linear -> output LIF
- Training uses `snntorch` losses/accuracies and encoding-aware presets in [presets.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scnn/scnn/presets.py:1)

Interpretation:
- `rate` keeps time-varying spike activity
- `latency` converts a collapsed static activity map into time-to-first-spike
- `delta` emphasizes temporal changes between bins

### 2. Event2Vec Stack

Defined in [event2vec/e2v.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/e2v.py:31) and [event2vec/data.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/data.py:25).

Local pipeline:

1. Load raw events
2. Bin them into `num_steps=20` ON/OFF frames
3. Average-pool the `180x240` sensor by `(6, 8)` to `30x30`
4. Apply `rate`, `latency`, or `delta` encoding
5. Convert nonzero encoded activations into tokens `[x, y, t, p, rho]`
6. Cap token count to `max_tokens`
7. Feed tokens into the Event2Vec model

Model internals:

- Spatial embedding: MLP `3 -> D/4 -> D/2 -> D`
- Temporal embedding: Conv1d over `delta-t`, `1 -> D/4 -> D/2 -> D`
- Fusion: `V = (log(rho) + 1) * (Vs + Vt)`
- Backbone: repeated shared bidirectional attention blocks
- Readout: masked mean over tokens, then linear classifier

Default ASL-DVS config in [default_config.json](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/default_config.json:1):

- `D=64`
- `depth=2`
- `heads=2`
- `FFN=128`
- `batch_size=32`
- `max_tokens=1024`

Important note:
- The Event2Vec core is paper-like, but this project adds a custom preprocessing front-end before tokenization.

### 3. EventGAN2Vec Extension

Defined in [eventgan_asl_event2vec.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scripts/eventgan_asl_event2vec.py:1).

- Reads ASL still images
- Applies synthetic motion to create image pairs
- Uses EventGAN to generate event volumes
- Converts generated volumes into `.mat` event files
- Evaluates them with a trained Event2Vec classifier
- Exports CSV summaries, accuracy plots, and videos

## Experiments Run In This Project

### Main comparison

The main experiment is: compare `rate`, `latency`, and `delta` encodings across two model families.

Families:

- SCNN `base`
- SCNN `deep`
- SCNN extended `x2` epoch runs
- Event2Vec `5e`
- Event2Vec `10e`
- Event2Vec `50e`

Artifacts:

- Run outputs: `runs/`
- Comparison notebooks: `notebooks/compare_*.ipynb`
- Analysis summaries: [Analytics/report.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/Analytics/report.md:1), [Analytics/data_analysis.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/Analytics/data_analysis.md:1)

### Main findings

From the latest completed summaries in `Analytics/`:

- Best completed model: Event2Vec `latency` at `95.57%` test accuracy
- Best Event2Vec efficiency tradeoff: `rate` at `92.10%`
- Best completed SCNN run: `SCNN x2 deep rate` at `79.18%`
- Event2Vec clearly outperformed SCNN on the completed ASL-DVS comparisons
- `delta` was the most token-efficient Event2Vec encoding, but it paid a clear accuracy penalty
- SCNN `latency` runs were reported as unhealthy and near chance in the live analysis

### SCNN-specific conclusions

Documented in [scnn_encoding_performance_analysis.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scnn_encoding_performance_analysis.md:1).

Key takeaway:
- The original SCNN comparison mixed different encoding semantics under one shared training recipe
- The project later added encoding-aware presets, better checkpoint selection, and milder delta settings to make the comparison fairer

## Where To Look First

- Dataset/setup: [DATASET_SETUP.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/DATASET_SETUP.md:1)
- Event2Vec model: [event2vec/e2v.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/e2v.py:31)
- Event2Vec preprocessing: [event2vec/data.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/data.py:65)
- Event2Vec training: [event2vec/train.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/event2vec/train.py:254)
- SCNN model: [scnn/scnn/model.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scnn/scnn/model.py:40)
- SCNN training: [scnn/scnn/t_loop.py](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/scnn/scnn/t_loop.py:123)
- Latest results summary: [Analytics/report.md](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/Analytics/report.md:1)
- Architecture comparison: [Analytics/architecture/event2vec_architecture_comparison.png](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/Analytics/architecture/event2vec_architecture_comparison.png)
- Simplified paper-vs-local comparison: [Analytics/architecture/event2vec_alignment_matrix.png](/Users/zac/Desktop/Vision_AI/Vision_AI_Project/Analytics/architecture/event2vec_alignment_matrix.png)

## One-Sentence Summary

This project evolved into an ASL-DVS event-recognition comparison effort: it sets up the dataset, builds spike/tokens from raw events, trains SCNN and Event2Vec variants across multiple encodings, analyzes the results, and extends the best Event2Vec pipeline to EventGAN-generated event data.
