# Latest Run Analysis

Generated: 2026-04-04T09:24:48

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

## Latency-Encoding Progression

| Run | Test Acc | Test Loss | Spike Count |
| --- | ---: | ---: | ---: |
| SCNN base | 22.37% | 2.9542 | 53.51 |
| SCNN deep | 32.95% | 2.7855 | 49.96 |
| Event2Vec 5e | 71.67% | 0.9090 | 341.13 |
| Event2Vec 50e | 95.57% | 0.1589 | 341.13 |

## Delta-Modulation Progression

| Run | Test Acc | Test Loss | Spike Count |
| --- | ---: | ---: | ---: |
| SCNN base | 34.60% | 2.8680 | 36.90 |
| SCNN deep | 4.16% | 3.1781 | 0.00 |
| Event2Vec 5e | 53.04% | 1.4868 | 142.75 |
| Event2Vec 50e | 72.37% | 0.8902 | 142.75 |

## Best Model Per Training Bundle: Time to Train

| Best Model | Test Acc | Test Loss | Train Time |
| --- | ---: | ---: | ---: |
| Event2Vec 5e rate | 74.97% | 0.7979 | 6m 31s |
| Event2Vec 10e rate | 83.36% | 0.5218 | 12m 50s |
| Event2Vec 50e latency | 95.57% | 0.1589 | 54m 4s |
| SCNN base rate | 39.02% | 2.7674 | 4h 1m 32s |
| SCNN deep rate | 53.65% | 2.7329 | 7h 47m 13s |
| SCNN x2 base rate | 68.03% | 2.5084 | 9h 40m 34s |
| SCNN x2 deep rate | 79.18% | 2.4462 | 22h 25m 23s |

## Active SCNN Latency Jobs

| Job | Last Completed Epoch | Val Acc | Val Loss | Val Spikes |
| --- | ---: | ---: | ---: | ---: |
| SCNN base latency | 5 | 5.98% | 3.1294 | 0.10 |
| SCNN deep latency | 4 | 4.16% | 3.1781 | 0.00 |

## Interpretation

- Event2Vec latency is the accuracy leader at 95.57%, beating Event2Vec rate by 3.47 percentage points, but it spends 1.90x more tokens/spikes.
- Event2Vec delta is the most token-efficient encoding, but it trails Event2Vec rate by 19.74 percentage points, which is a large accuracy tax for the smaller token budget.
- Event2Vec rate improves strongly with longer training: +8.38 points from 5 to 10 epochs, then another +8.75 points from 10 to 50 epochs.
- SCNN deep rate improves by 25.53 points between the first deep run and the x2-epoch deep run, so additional training helped substantially, but it still trails Event2Vec 50e rate by 12.93 points.
- The live SCNN latency jobs look unhealthy so far: validation accuracy is near chance-level and the validation loss remains high, which usually points to an encoding or optimization mismatch rather than simple undertraining.

## Generated Charts

- `event2vec_50_test_metrics.png`
- `event2vec_50_efficiency.png`
- `event2vec_50_learning_curves.png`
- `rate_encoding_progression.png`
- `latency_encoding_progression.png`
- `delta_encoding_progression.png`
- `best_models_training_time.png`
- `scnn_live_latency_progress.png`
