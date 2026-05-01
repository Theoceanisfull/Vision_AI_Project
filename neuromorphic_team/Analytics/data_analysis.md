# Data Analysis

Generated on April 4, 2026.

## Overview

This document summarizes the latest run assessment for the Neuromorphic Vision experiments, including the freshest completed Event2Vec results, the most relevant SCNN comparisons, and the current status of the still-running SCNN latency jobs.

## Key Findings

- Event2Vec `latency` is the best completed run so far, reaching `95.57%` test accuracy with `0.1589` test loss.
- Event2Vec `rate` offers the best efficiency-performance tradeoff, with `92.10%` test accuracy at roughly half the token/spike budget of `latency`.
- Event2Vec `delta` is the most token-efficient encoding, but it falls well behind in accuracy at `72.37%`.
- Event2Vec rate training keeps improving with more epochs: `74.97%` at 5 epochs, `83.36%` at 10 epochs, and `92.10%` at 50 epochs.
- SCNN rate runs do improve with additional training, but even the stronger `SCNN x2 deep` run at `79.18%` still trails `Event2Vec 50e rate` by `12.93` percentage points.
- The active SCNN latency jobs look unhealthy so far, with validation accuracy near chance-level and very high loss, which suggests spike collapse or an encoding/optimization mismatch rather than simple undertraining.

## Event2Vec 50-Epoch Snapshot

| Encoding | Best Epoch | Test Acc | Test Loss | Test Tokens | Best Val Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| latency | 49 | 95.57% | 0.1589 | 341.13 | 95.71% |
| rate | 47 | 92.10% | 0.2530 | 179.31 | 92.31% |
| delta | 49 | 72.37% | 0.8902 | 142.75 | 72.93% |

## Rate-Encoding Progression

| Run | Test Acc | Test Loss | Spike Count |
| --- | ---: | ---: | ---: |
| SCNN base | 39.02% | 2.7674 | 44.68 |
| SCNN deep | 53.65% | 2.7329 | 75.93 |
| SCNN x2 base | 68.03% | 2.5084 | 46.45 |
| SCNN x2 deep | 79.18% | 2.4462 | 42.80 |
| Event2Vec 5e | 74.97% | 0.7979 | 179.09 |
| Event2Vec 10e | 83.36% | 0.5218 | 179.13 |
| Event2Vec 50e | 92.10% | 0.2530 | 179.31 |

## Active SCNN Latency Jobs

| Job | Last Completed Epoch | Val Acc | Val Loss | Val Spikes |
| --- | ---: | ---: | ---: | ---: |
| SCNN base latency | 5 | 5.98% | 3.1294 | 0.10 |
| SCNN deep latency | 4 | 4.16% | 3.1781 | 0.00 |

## Data Science Interpretation

- `Latency` is the accuracy-maximizing encoding in the latest completed Event2Vec sweep. In practical terms, it preserves more discriminative timing structure, which likely helps the classifier separate similar gesture classes more reliably.
- `Rate` looks like the best operating point if compute or token budget matters. It gives up only `3.47` percentage points versus latency while using about `1.90x` fewer tokens/spikes.
- `Delta` appears to compress too aggressively for this task. Its lower token budget is attractive, but the accuracy penalty is large enough that it is probably discarding useful event dynamics.
- The Event2Vec learning behavior is healthy. Accuracy rises steadily and loss falls steadily through 50 epochs, which is what we want to see when a model is still learning signal instead of memorizing noise.
- The SCNN rate trend suggests that training budget matters, because longer runs helped a lot. But the gap to Event2Vec is still wide, so architecture and representation differences are likely contributing too.
- The live SCNN latency runs are concerning. Near-chance validation accuracy, almost no spikes, and flat high loss often mean the network is not firing usefully, which can happen when thresholds, encoding scale, loss choice, or timing dynamics are mismatched.

## Charts

- [Event2Vec 50 test metrics](event2vec_50_test_metrics.png)
- [Event2Vec 50 efficiency plot](event2vec_50_efficiency.png)
- [Event2Vec 50 learning curves](event2vec_50_learning_curves.png)
- [Rate-encoding progression](rate_encoding_progression.png)
- [Latency-encoding progression](latency_encoding_progression.png)
- [Delta-modulation progression](delta_encoding_progression.png)
- [Best-model training time comparison](best_models_training_time.png)
- [Live SCNN latency progress](scnn_live_latency_progress.png)

## Source Artifacts

- Detailed generated report: [report.md](report.md)
- Event2Vec metrics table: [event2vec_50_metrics.csv](event2vec_50_metrics.csv)
- Rate progression table: [rate_progression.csv](rate_progression.csv)
- Latency progression table: [latency_progression.csv](latency_progression.csv)
- Delta progression table: [delta_progression.csv](delta_progression.csv)
- Best-model training time table: [best_models_training_time.csv](best_models_training_time.csv)
- Live SCNN latency table: [scnn_live_latency.csv](scnn_live_latency.csv)
