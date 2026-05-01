# Event2Vec Side-by-Side Alignment

Paper reference: [Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space](https://arxiv.org/abs/2504.15371)
Checked against the latest arXiv version available on February 5, 2026 (v5), including Figure 6 and Appendix A.2/Table 5.
Local config: `event2vec/default_config.json`

## Fast Read

- The Event2Vec embedding core in this project is close to the paper.
- The data front-end in this project is not the paper front-end.
- So the backbone is paper-like, but the full end-to-end pipeline is not a strict Figure 6 reproduction.

## Row-by-Row Comparison

| Stage | Paper Figure 6 / Appendix | Local project | Status | Why it matters |
| --- | --- | --- | --- | --- |
| Input representation | Raw events (x, y, t, p), with rho available when event clustering is used. | Raw DVS events are read first, but the model does not consume them directly. | Partial | Both start from events, but the local model only sees derived tokens after extra preprocessing. |
| Pre-token preprocessing | Figure 6 shows direct Event2Vec processing; appendix text mentions clustering for long streams, not frame/bin conversion. | Events are binned into 20 temporal frames, split into ON/OFF channels, average-pooled from 180x240 to 30x30, then encoded with 'rate'. | Different | This is the largest architectural difference. The project front-end is not the paper's direct raw-event path. |
| Token formation | Event tokens conceptually come from events themselves and include rho scaling when available. | Only nonzero encoded spikes become tokens [x, y, t_norm, p, rho], then token count is capped. | Different | The local token stream is sparser and already transformed before Event2Vec sees it. |
| Spatial embedding | Linear 3 -> D/4 -> ReLU -> Linear D/4 -> D/2 -> ReLU -> Linear D/2 -> D. | Same shape in `SpatialEmbedding`: Linear 3 -> 16 -> 32 -> 64, with LayerNorm after each linear. | Match | This part is very close to the Figure 6 reference. |
| Temporal embedding | Conv1d 1 -> D/4 -> ReLU -> Conv1d D/4 -> D/2 -> ReLU -> Conv1d D/2 -> D over delta-t. | Same channel progression in `TemporalEmbedding`: Conv1d 1 -> 16 -> 32 -> 64 over delta-t, with LayerNorm after each conv. | Match | This also closely follows Figure 6, although the local code uses standard Conv1d as noted in `e2v.py`. |
| Event2Vec fusion | V = (Vs + Vt) * (log rho + 1). | Same formula in `Event2Vec.forward`. | Match | The core Event2Vec token fusion is aligned. |
| Backbone block | Backbone x l with Self-Attention + FFN. For ASL-DVS: D=64, Df=128, heads=2, depth=2. | `SharedBidirectionalAttentionBlock` x 2: forward + reversed `nn.MultiheadAttention`, fusion linear, FFN 64->128->64. | Partial | The size matches the ASL-DVS paper preset, but the attention implementation differs from the appendix's FoX/GLA discussion. |
| Pooling in backbone | Optional; enabled for DVS Gesture, disabled for ASL-DVS and DVS-Lip. | `pool_after_each_block=[False, False]`. | Match | Your default config matches the ASL-DVS paper preset here. |
| Readout / head | Backbone output feeds a linear classification head. | Masked mean pooling across tokens, then Linear 64->24. | Partial | Functionally similar classifier output, but the local readout explicitly averages token features first. |
| Overall verdict | Direct Event2Vec classification pipeline from Figure 6 / Appendix A.2. | Paper-like Event2Vec encoder/backbone attached to a project-specific preprocessing front-end. | Different | Your project is best described as Event2Vec-inspired or partially paper-aligned, not a strict Figure 6 reproduction. |

## Local Defaults

- Encoding: `rate`
- Sensor size: `(180, 240)`
- Pool kernel: `(6, 8)`
- Time steps: `20`
- Max tokens: `1024`
- Model dims: `D=64`, `Df=128`, `heads=2`, `depth=2`
