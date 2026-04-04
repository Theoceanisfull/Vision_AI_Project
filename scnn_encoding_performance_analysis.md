# SCNN Encoding Performance Analysis

## Executive Summary

The current SCNN stack is partially working, but the original comparison was unfair in a few important ways:

1. `rate` and `latency` were built from a collapsed static image, while `delta` was built from the full temporal event sequence.
2. All three encodings were trained with the same rate-oriented objective, even though `latency` and `delta` express information differently.
3. Training always reported the last epoch, not the best validation epoch, which penalized runs like `deep_rate` that peaked early and then drifted.
4. The sweep scripts started from `default_config.json`, so the per-encoding configs in `models/scnn/configs/` were not actually controlling the main experiment.

The net result was that the project was mixing three different encoding assumptions under one training recipe. That does not mean the codebase is fundamentally broken. It does mean the comparison was not yet a clean measurement of "which encoding is best."

## What The Existing Results Show

### Base architecture

| Encoding | Test Acc | Test Loss | Avg Test Output Spikes |
| --- | ---: | ---: | ---: |
| `rate` | `39.02%` | `2.7674` | `44.68` |
| `delta` | `34.60%` | `2.8680` | `36.90` |
| `latency` | `22.37%` | `2.9542` | `53.51` |

### Deep architecture

| Encoding | Test Acc | Test Loss | Avg Test Output Spikes |
| --- | ---: | ---: | ---: |
| `rate` | `53.65%` | `2.7329` | `75.93` |
| `latency` | `32.95%` | `2.7855` | `49.96` |
| `delta` | `4.16%` | `3.1781` | `0.00` |

## Per-Encoding Analysis

### Rate encoding

Rate coding is the most forgiving representation in this project. It distributes evidence over many time steps, which makes the optimization target dense and easier for a Conv-SNN to learn. In the original implementation, `rate` was given a particularly favorable setup because the temporal event bins were first collapsed into one static image and then re-expanded into Poisson spikes. That removed much of the irregularity of the original event sequence.

The deep rate run is the strongest evidence that the overall model family can work. Its validation accuracy peaked near `74.23%` around epoch `7`, while the reported final test accuracy came from the last epoch rather than the best validation epoch. That means the headline metric understated the best-performing deep model.

Why rate tends to work better here:

- It gives the network repeated evidence across time.
- It is compatible with `ce_rate_loss`, which rewards the correct class for firing consistently.
- The representation is robust to noise because evidence is averaged over many Bernoulli draws.

Primary weakness:

- If rate coding is built from a collapsed static image, it stops being a faithful event-stream encoding and becomes closer to a conventional frame-based baseline.

### Latency encoding

Latency coding compresses activity into spike timing rather than repeated firing. In this codebase, latency is generated from a single static image derived from the event sequence. That means latency is not seeing the original temporal progression of the gesture. Instead, it is seeing the aggregate activity map and then converting magnitude into first-spike time.

This representation is much sparser than rate coding. The model must learn to decode timing cleanly, and that is harder when the training objective assumes that the correct class should fire repeatedly across time. The original setup therefore asked latency-coded inputs to behave like rate-coded outputs.

Observed behavior:

- Training accuracy rose strongly.
- Validation accuracy lagged far behind.
- The gap suggests a mismatch between representation and objective, plus weaker generalization.

Why latency underperformed:

- The encoding discards within-sample temporal ordering before latency conversion.
- The original rate-based loss and metric were a poor fit.
- One-spike-style codes are less redundant than rate codes, so mistakes are less recoverable.

### Delta encoding

Delta coding detects changes between adjacent event frames. Unlike latency, it preserves the temporal sequence. Unlike rate, it is explicitly sparse and signed. Positive changes produce ON spikes and negative changes can produce OFF spikes.

This is the most brittle encoding in the current stack. It can work, as shown by the base model reaching `34.60%` test accuracy, but it is much more sensitive to optimization settings.

The deep delta run collapsed completely:

- training accuracy dropped to chance for `24` classes
- validation accuracy flattened at chance
- output spikes fell to nearly zero by the middle of training

That pattern is consistent with output silence rather than ordinary underfitting.

Why delta is fragile here:

- The inputs are sparse and signed, so the network receives less redundant evidence.
- `ce_rate_loss` was encouraging the correct class to fire on every step, which conflicts with sparse delta behavior.
- The shared neuron threshold and learning rate appear too aggressive for deep delta.
- Once the network drifts into a low-spike regime, it becomes difficult to recover.

## Measured Representation Differences

I sampled the training set representation statistics after loading the dataset:

| Encoding | Avg abs sum per sample | Avg nonzero fraction | Notes |
| --- | ---: | ---: | --- |
| `rate` | `43553.20` | `0.0252` | dense relative to the others |
| `latency` | `7948.92` | `0.0046` | very sparse, one-spike-like |
| `delta` | `24750.38` | `0.0143` | sparse and signed |

For `delta`, the positive and negative fractions were almost balanced:

- avg positive fraction: `0.00719`
- avg negative fraction: `0.00714`

This is important because it shows the input loader itself is not producing empty delta samples. The collapse happens later, inside the model/training dynamics.

## Root Causes Of The Subpar Performance

### 1. Unfair preprocessing across encodings

Original behavior:

- `rate`: temporal bins -> collapsed static image -> Poisson spikes
- `latency`: temporal bins -> collapsed static image -> time-to-first-spike
- `delta`: temporal bins -> delta operator

This gave `delta` access to true temporal differences while `rate` and `latency` were effectively operating on an accumulated frame representation.

Impact:

- the comparison mixed temporal and non-temporal inputs
- `rate` was advantaged by a denoised aggregate input
- `latency` lost event-order information before encoding

### 2. Objective mismatch

Original behavior:

- all three encodings used `ce_rate_loss`
- all three runs reported `accuracy_rate`

Impact:

- `latency` was judged by repeated firing instead of early correct spikes
- `delta` was pushed toward dense firing even though its input is sparse by design

### 3. No best-checkpoint selection

Original behavior:

- the test set was evaluated only on the last epoch
- no `checkpoint_best.pt` was saved

Impact:

- overfit runs looked worse than their best validation state
- the deep rate run is the clearest example of this problem

### 4. Per-encoding configs were not driving the sweep

Original behavior:

- `train_three_base.py` and `train_three_deep.py` loaded `default_config.json`
- they then changed only architecture and encoding

Impact:

- the JSONs under `models/scnn/configs/` did not govern the main sweep
- intended encoding-specific settings could drift out of sync with actual training

### 5. Delta-specific tuning was missing

Deep delta likely needed gentler defaults from the start:

- lower learning rate
- lower delta threshold
- lower neuron firing threshold
- a loss that tolerates sparse evidence

Without those adjustments, the deep stack appears to have fallen into a silent solution.

## Changes Implemented

### Training behavior

- Added best-validation checkpoint selection with `checkpoint_best.pt`
- Preserved `checkpoint_last.pt`
- Test evaluation now runs on the best validation checkpoint
- Metrics now include both selected-checkpoint test metrics and last-checkpoint test metrics

### Encoding-aware presets

- `rate` now uses time-varying Poisson rate coding on the temporal event bins
- `latency` now uses temporal loss/accuracy (`ce_temporal_loss`, `accuracy_temporal`)
- `delta` now uses count-based loss (`ce_count_loss`) plus milder defaults:
  - lower learning rate
  - lower input delta threshold
  - lower neuron threshold
  - lower dropout
  - tighter gradient clipping

### Sweep reliability

- Sweep scripts now apply explicit encoding presets, so the launched experiment matches the intended setup more closely
- Starter config JSONs were updated to match the new presets

## Residual Limitations

Even after the fixes, the comparison is still not perfectly symmetric because latency coding in `snntorch` is fundamentally a static-feature-to-spike-timing transform. That means:

- `rate` and `delta` can naturally operate on a temporal frame sequence
- `latency` still needs a collapsed activity map before encoding

So the revised setup is better, but not mathematically identical across all three representations.

If we want a stricter comparison, we should decide on one of these experiment goals:

1. Compare "best practical encoder for this dataset and model family"
2. Compare "encoders under identical upstream information content"

Those are related, but not identical, experiments.

## Recommended Next Experiments

1. Rerun the base and deep sweeps with the new best-checkpoint logic and encoding presets.
2. Track both last and best-selected test metrics in the comparison notebook.
3. For deep delta, run a small grid over:
   - `lr`: `1e-4`, `3e-4`, `5e-4`
   - `delta_threshold`: `0.02`, `0.05`, `0.08`
   - neuron `threshold`: `0.5`, `0.75`, `1.0`
4. Try more time steps, such as `30` or `40`, to reduce temporal aliasing in the event binning.
5. Add early stopping or at least report the best validation epoch in the notebook summary tables.
6. Consider event-domain augmentations, because the current pipeline appears to rely heavily on the raw split without additional regularization.

## Practical Bottom Line

The disappointing results are not just "the model is bad." The deeper issue is that the original experiment mixed different input semantics under one shared training recipe.

The strongest conclusions from the current evidence are:

- the Conv-SNN pipeline is viable because `deep_rate` learned well
- latency needs a timing-aware objective
- delta needs its own optimization regime
- last-epoch reporting was hiding the best version of the better runs

That means the right next step is not to abandon the project. It is to rerun the comparison under the corrected training logic and then judge the encodings again from that cleaner baseline.
