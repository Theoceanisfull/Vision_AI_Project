# Vision AI Project

Comparative neuromorphic sign-language recognition on ASL-DVS using Conv-SNN and Event2Vec model families.

## Overview

This project studies event-camera classification for American Sign Language recognition on the ASL-DVS dataset. The main goal is to compare how different event encodings interact with two different model families:

- `scnn/`: a convolutional spiking neural network baseline built with `snntorch`
- `event2vec/`: an Event2Vec-style token model for event classification

The project also includes an exploratory `EventGAN -> Event2Vec` pipeline that generates synthetic event streams from still ASL images and evaluates them with the same downstream classifier.

## Main Result

The strongest completed model in this project is `Event2Vec + latency`, which reaches `95.57%` test accuracy on the `24-class` ASL-DVS task. The strongest completed Conv-SNN result is `79.18%`, so the current experimental record favors Event2Vec.

## Why The Event2Vec Result Matters

Compared to the Event2Vec paper's ASL-DVS setting of `512 random events`, this project achieves competitive performance with materially fewer average tokens:

- `delta`: `142.75` tokens on average, about `3.6x fewer tokens`
- `rate`: `179.31` tokens on average, about `2.9x fewer tokens`
- `latency`: `341.13` tokens on average, about `1.5x fewer tokens`, while reaching `95.57%` test accuracy

This matters because the local Event2Vec pipeline is operating under a reduced token budget relative to the paper's reported ASL-DVS event count.

## Project Question

The central question is:

How do `rate`, `latency`, and `delta` event representations behave when paired with:

- dense spike-volume processing in a Conv-SNN
- sparse token processing in an Event2Vec-style model

## Data And Preprocessing

The project uses ASL-DVS `.mat` event files containing raw event arrays:

- `x`
- `y`
- `ts`
- `pol`

Both major pipelines sanitize coordinates, vertically flip `y`, and split the data into stratified train, validation, and test sets.

### Conv-SNN Input Path

The Conv-SNN stack converts each sample into a dense spike tensor:

`[T, 2, H, W]`

Default behavior:

- `T = 20` time bins
- `2` channels for ON/OFF events
- sensor size `180 x 240`

Encodings:

- `rate`: time-varying spike activity
- `latency`: time-to-first-spike after collapsing the temporal bins
- `delta`: temporal differences between adjacent bins

### Event2Vec Input Path

The Event2Vec stack uses a stronger preprocessing front-end before the learned model:

1. raw events are binned into `20` ON/OFF frames
2. the `180 x 240` sensor is average-pooled by `(6, 8)` to `30 x 30`
3. `rate`, `latency`, or `delta` encoding is applied
4. nonzero activations are converted into sparse tokens `[x, y, t, p, rho]`
5. token count is pruned and capped before entering the model

This is an important detail for class discussion: the local Event2Vec implementation is not a strict direct raw-event pipeline. It keeps the Event2Vec core, but adds a project-specific front-end that performs time binning, spatial pooling, encoding, sparsification, and token reduction before the model sees the sequence.

## Model Architecture

### Conv-SNN

Implemented in [`scnn/scnn/model.py`](scnn/scnn/model.py).

Core block:

`Conv2d -> BatchNorm -> LIF -> MaxPool`

Variants:

- `base`: 2 convolutional stages with channels `(32, 64)`
- `deep`: 3 convolutional stages with channels `(32, 64, 128)`

Head:

- flatten
- dropout
- linear
- LIF
- linear
- output LIF

### Event2Vec

Implemented in [`event2vec/e2v.py`](event2vec/e2v.py) with preprocessing in [`event2vec/data.py`](event2vec/data.py).

Default ASL-DVS configuration:

- embedding dimension `D = 64`
- depth `= 2`
- attention heads `= 2`
- feed-forward dimension `= 128`
- dropout `= 0.1`

Main components:

- spatial embedding: MLP over `(x, y, p)`
- temporal embedding: `Conv1d` over `delta-t`
- fusion: `V = (log(rho) + 1) * (Vs + Vt)`
- backbone: shared bidirectional attention blocks
- readout: masked mean pooling and linear classifier

The learned Event2Vec classifier is compact relative to the Conv-SNN baselines, but its behavior must be interpreted together with the front-end preprocessing that reduces and transforms the event stream before tokenization.

## Experimental Design

The main experiment compares three input encodings:

- `rate`
- `latency`
- `delta`

Across these model groups:

- Conv-SNN `base`
- Conv-SNN `deep`
- Conv-SNN extended `x2` runs
- Event2Vec `5e`
- Event2Vec `10e`
- Event2Vec `50e`

Best checkpoints are selected by validation accuracy, with validation loss used as a tie-breaker.

## Best Completed Results

### Event2Vec 50-Epoch Comparison

| Encoding | Best Epoch | Test Accuracy | Test Loss | Average Tokens |
| --- | ---: | ---: | ---: | ---: |
| latency | 49 | 95.57% | 0.1589 | 341.13 |
| rate | 47 | 92.10% | 0.2530 | 179.31 |
| delta | 49 | 72.37% | 0.8902 | 142.75 |

Interpretation:

- `latency` gives the best absolute accuracy
- `rate` gives the best accuracy-efficiency balance
- `delta` is the most token-efficient, but with a clear accuracy penalty

### Best Cross-Family Comparison

| Encoding | Best Conv-SNN | Accuracy | Best Event2Vec | Accuracy |
| --- | --- | ---: | --- | ---: |
| rate | SCNN x2 deep | 79.18% | Event2Vec 50e | 92.10% |
| latency | SCNN deep | 32.95% | Event2Vec 50e | 95.57% |
| delta | SCNN base | 34.60% | Event2Vec 50e | 72.37% |

## Interpretation

Three conclusions define the current project:

- Event2Vec is the strongest completed model family on ASL-DVS in this project.
- Representation choice matters. For Event2Vec, `latency` is best for accuracy, `rate` is best for balance, and `delta` is best for token compression.
- Preprocessing is a major part of the result. The difference between Conv-SNN and Event2Vec is not only the backbone, but also the representation each model receives.

## Project Structure

- [`event2vec/`](event2vec): Event2Vec preprocessing, model, and training code
- [`scnn/`](scnn): Conv-SNN models and training code
- [`scripts/`](scripts): dataset setup, analysis, architecture visualization, report export, and EventGAN evaluation
- [`Analytics/`](Analytics): generated figures, summaries, and report outputs
- [`notebooks/`](notebooks): comparison and inspection notebooks
- [`EventGAN/`](EventGAN): synthetic event generation code
- [`Kohl/`](Kohl): earlier standalone image and video baselines

## Key Files

- [`DATASET_SETUP.md`](DATASET_SETUP.md)
- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)
- [`Analytics/Vision_AI_Project_Academic_Report.pdf`](Analytics/Vision_AI_Project_Academic_Report.pdf)
- [`Analytics/Vision_AI_Project_Academic_Report.md`](Analytics/Vision_AI_Project_Academic_Report.md)
- [`event2vec/e2v.py`](event2vec/e2v.py)
- [`event2vec/data.py`](event2vec/data.py)
- [`scnn/scnn/model.py`](scnn/scnn/model.py)
- [`Analytics/architecture/event2vec_architecture_comparison.png`](Analytics/architecture/event2vec_architecture_comparison.png)
- [`Analytics/architecture/event2vec_alignment_matrix.png`](Analytics/architecture/event2vec_alignment_matrix.png)

## Reproducing Main Outputs

Generate the architecture comparison:

```bash
python scripts/visualize_event2vec_architecture.py
```

Generate the academic report PDF:

```bash
python scripts/export_academic_repo_report.py
```

## One-Sentence Summary

This project builds event representations from raw ASL-DVS data, compares Conv-SNN and Event2Vec architectures across multiple encodings, and shows that the strongest local Event2Vec pipeline achieves high accuracy with a reduced token budget relative to the Event2Vec paper's ASL-DVS setting.
