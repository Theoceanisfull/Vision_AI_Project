# Event2Vec Architecture Comparison

Generated from:
- Paper: [Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space](https://arxiv.org/abs/2504.15371)
- Local config: `event2vec/default_config.json`

## Paper Reference

Figure 6: The network architecture for event classification using the event2vec representation.

Selected paper preset: `asl_dvs` (ASL-DVS)

- Embedding dimension `D=64`
- FFN hidden dimension `Df=128`
- Attention heads `nhead=2`
- Backbone depth `l=2`
- Repeats `1`
- GPUs `7`
- `lrmin=1e-6`
- Note: No pooling after FFN in the appendix configuration.

## Local Project Pipeline

- Data root: `data/ASLDVS`
- Encoding: `rate`
- Sensor size: `(180, 240)`
- Pool kernel: `(6, 8)`
- Pooled sensor size: `(30, 30)`
- Time steps: `20`
- Max tokens: `1024`
- Model classes: `24`
- Model dimensions: `D=64`, `Df=128`, `heads=2`, `depth=2`
- Backbone pooling enabled: `False`
- Parameter count: `96,024`

## Key Differences

- The paper Figure 6 starts from raw events and optional clustering-derived intensity rho. This project inserts a preprocessing pipeline in `event2vec/data.py` that bins events to frames, average-pools the sensor, applies a spike encoding (`rate`, `latency`, or `delta`), and only then converts nonzero activations into `[x, y, t, p, rho]` tokens.
- The paper appendix discusses bidirectional variants of FoX and GLA attention. The local model in `event2vec/e2v.py` uses `nn.MultiheadAttention` inside `SharedBidirectionalAttentionBlock`, not FoX or GLA.
- The local default config matches the paper's ASL-DVS scale fairly closely on backbone size: D=64, Df=128, heads=2, depth=2.
- The local default config disables pooling in the backbone (`pool_after_each_block=[False, False]`), which is aligned with the paper's ASL-DVS appendix setting but differs from the DVS Gesture preset.

## torchinfo Summary

```text
===========================================================================================================================================================
Layer (type (var_name):depth-idx)                       Input Shape               Output Shape              Param #                   Trainable
===========================================================================================================================================================
Event2VecClassifier (Event2VecClassifier)               [1, 1024, 5]              [1, 24]                   --                        True
├─Event2Vec (event2vec): 1-1                            [1, 1024, 5]              [1, 1024, 64]             --                        True
│    └─SpatialEmbedding (spatial): 2-1                  [1, 1024]                 [1, 1024, 64]             --                        True
│    │    └─Linear (fc1): 3-1                           [1, 1024, 3]              [1, 1024, 16]             64                        True
│    │    └─LayerNorm (ln1): 3-2                        [1, 1024, 16]             [1, 1024, 16]             32                        True
│    │    └─Linear (fc2): 3-3                           [1, 1024, 16]             [1, 1024, 32]             544                       True
│    │    └─LayerNorm (ln2): 3-4                        [1, 1024, 32]             [1, 1024, 32]             64                        True
│    │    └─Linear (fc3): 3-5                           [1, 1024, 32]             [1, 1024, 64]             2,112                     True
│    │    └─LayerNorm (ln3): 3-6                        [1, 1024, 64]             [1, 1024, 64]             128                       True
│    └─TemporalEmbedding (temporal): 2-2                [1, 1024]                 [1, 1024, 64]             --                        True
│    │    └─Conv1d (conv1): 3-7                         [1, 1, 1024]              [1, 16, 1024]             64                        True
│    │    └─LayerNorm (ln1): 3-8                        [1, 1024, 16]             [1, 1024, 16]             32                        True
│    │    └─Conv1d (conv2): 3-9                         [1, 16, 1024]             [1, 32, 1024]             1,568                     True
│    │    └─LayerNorm (ln2): 3-10                       [1, 1024, 32]             [1, 1024, 32]             64                        True
│    │    └─Conv1d (conv3): 3-11                        [1, 32, 1024]             [1, 64, 1024]             6,208                     True
│    │    └─LayerNorm (ln3): 3-12                       [1, 1024, 64]             [1, 1024, 64]             128                       True
├─ModuleList (blocks): 1-2                              --                        --                        --                        True
│    └─SharedBidirectionalAttentionBlock (0): 2-3       [1, 1024, 64]             [1, 1024, 64]             --                        True
│    │    └─LayerNorm (norm1): 3-13                     [1, 1024, 64]             [1, 1024, 64]             128                       True
│    │    └─MultiheadAttention (attn): 3-14             [1, 1024, 64]             [1, 1024, 64]             16,640                    True
│    │    └─MultiheadAttention (attn): 3-15             [1, 1024, 64]             [1, 1024, 64]             (recursive)               True
│    │    └─Linear (fuse): 3-16                         [1, 1024, 128]            [1, 1024, 64]             8,256                     True
│    │    └─Dropout (drop1): 3-17                       [1, 1024, 64]             [1, 1024, 64]             --                        --
│    │    └─LayerNorm (norm2): 3-18                     [1, 1024, 64]             [1, 1024, 64]             128                       True
│    │    └─Sequential (ffn): 3-19                      [1, 1024, 64]             [1, 1024, 64]             --                        True
│    │    │    └─Linear (0): 4-1                        [1, 1024, 64]             [1, 1024, 128]            8,320                     True
│    │    │    └─GELU (1): 4-2                          [1, 1024, 128]            [1, 1024, 128]            --                        --
│    │    │    └─Dropout (2): 4-3                       [1, 1024, 128]            [1, 1024, 128]            --                        --
│    │    │    └─Linear (3): 4-4                        [1, 1024, 128]            [1, 1024, 64]             8,256                     True
│    │    └─Dropout (drop2): 3-20                       [1, 1024, 64]             [1, 1024, 64]             --                        --
│    └─SharedBidirectionalAttentionBlock (1): 2-4       [1, 1024, 64]             [1, 1024, 64]             --                        True
│    │    └─LayerNorm (norm1): 3-21                     [1, 1024, 64]             [1, 1024, 64]             128                       True
│    │    └─MultiheadAttention (attn): 3-22             [1, 1024, 64]             [1, 1024, 64]             16,640                    True
│    │    └─MultiheadAttention (attn): 3-23             [1, 1024, 64]             [1, 1024, 64]             (recursive)               True
│    │    └─Linear (fuse): 3-24                         [1, 1024, 128]            [1, 1024, 64]             8,256                     True
│    │    └─Dropout (drop1): 3-25                       [1, 1024, 64]             [1, 1024, 64]             --                        --
│    │    └─LayerNorm (norm2): 3-26                     [1, 1024, 64]             [1, 1024, 64]             128                       True
│    │    └─Sequential (ffn): 3-27                      [1, 1024, 64]             [1, 1024, 64]             --                        True
│    │    │    └─Linear (0): 4-5                        [1, 1024, 64]             [1, 1024, 128]            8,320                     True
│    │    │    └─GELU (1): 4-6                          [1, 1024, 128]            [1, 1024, 128]            --                        --
│    │    │    └─Dropout (2): 4-7                       [1, 1024, 128]            [1, 1024, 128]            --                        --
│    │    │    └─Linear (3): 4-8                        [1, 1024, 128]            [1, 1024, 64]             8,256                     True
│    │    └─Dropout (drop2): 3-28                       [1, 1024, 64]             [1, 1024, 64]             --                        --
├─Linear (head): 1-3                                    [1, 64]                   [1, 24]                   1,560                     True
===========================================================================================================================================================
Total params: 96,024
Trainable params: 96,024
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 8.08
===========================================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 9.96
Params size (MB): 0.25
Estimated Total Size (MB): 10.23
===========================================================================================================================================================
```
