# SpikeGen Preprocessing Example

This note walks through one real pooled cell from the local `Event2Vec + latency` preprocessing pipeline and shows exactly how `snntorch.spikegen.latency` turns that cell into a final Event2Vec token.

## Context

The local preprocessing path is:

1. Load raw ASL-DVS events `(x, y, ts, pol)`
2. Bin them into `20` time steps and `2` channels `(ON, OFF)`
3. Average-pool the `180 x 240` sensor into a `30 x 30` grid
4. Apply `spikegen.latency(...)`
5. Convert nonzero spikes into Event2Vec tokens `[x, y, t, p, rho]`

Relevant code:

- `event2vec/data.py`
- `event2vec/e2v.py`
- `snntorch/spikegen.py`

This example uses:

- sample: `data/ASLDVS/a/a_0001.mat`
- channel: `0` (ON)
- pooled location: `(y=12, x=22)`
- local `snntorch` version: `0.9.4`

## Step 1: One Cell's 20-Step History

After event binning and spatial pooling, this one pooled cell had the following normalized time history:

```text
t=0  0.0000
t=1  0.0000
t=2  0.0000
t=3  0.0104
t=4  0.0000
t=5  0.0104
t=6  0.0104
t=7  0.0000
t=8  0.0000
t=9  0.0000
t=10 0.0000
t=11 0.0000
t=12 0.0000
t=13 0.0000
t=14 0.0000
t=15 0.0000
t=16 0.0000
t=17 0.0000
t=18 0.0000
t=19 0.0000
```

Interpretation:

- this pooled cell was active only `3` times
- each nonzero entry came from one pooled event contribution after normalization

## Step 2: What Latency Encoding Changes

The local Event2Vec pipeline does **not** keep that 20-step history for latency mode.

Instead, it collapses time first:

```python
static_raw = pooled_raw.sum(dim=0)
static_norm = pooled_norm.sum(dim=0)
static_norm = static_norm / static_norm.max()
encoded = spikegen.latency(static_norm, num_steps=20, normalize=True, clip=True)
```

For this cell:

- `static_raw = 3.0`
- `static_norm = 0.0128205`

So the original local time pattern:

```text
[0, 0, 0, 0.0104, 0, 0.0104, 0.0104, 0, ...]
```

is replaced by a single scalar strength:

```text
0.0128205
```

This is the critical idea behind latency encoding in this project:

- time history is compressed
- each cell gets one strength value
- that strength determines one spike time

## Step 3: What `spikegen.latency` Computes

The installed `snntorch` implementation uses the logarithmic latency code:

```text
spike_time = tau * log(data / (data - threshold))
```

With the local defaults:

- `threshold = 0.01`
- `tau = 1`
- `normalize = True`
- `clip = True`

For this cell:

```text
data = 0.0128205
raw latency = log(0.0128205 / (0.0128205 - 0.01))
            = 1.5141
```

Because `normalize=True`, that raw latency is then rescaled relative to the slowest valid cell in the whole sample:

```text
max raw latency in this sample = 11.5071
normalized spike time = 1.5141 * 19 / 11.5071
                      = 2.5001
rounded step = 3
```

So this cell becomes a one-hot spike train:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## Step 4: Final Event2Vec Token

After latency encoding, the local preprocessing converts that spike into an Event2Vec token:

```text
[x=22, y=12, t=3/19, p=0, rho=3]
```

Numerically:

```text
[22.0000, 12.0000, 0.1579, 0.0000, 3.0000]
```

Meaning:

- `x = 22`: pooled x-coordinate
- `y = 12`: pooled y-coordinate
- `t = 0.1579`: normalized time index from spike step `3`
- `p = 0`: ON polarity channel
- `rho = 3`: total pooled raw strength for this cell

## Step 5: What The Purpose Is

Latency encoding is trying to express:

- stronger cells fire earlier
- weaker cells fire later
- cells below threshold do not fire at all

So the purpose is to compress the spatial-temporal event signal into a sparse rank-order code.

In this project specifically, that has three effects:

1. It makes the input sequence much sparser than keeping all active bins.
2. It removes fine-grained local timing and replaces it with a single spike time per cell.
3. It gives Event2Vec a smaller token set built from "importance-ranked" spatial locations.

## Step 6: What Gets Discarded

Latency mode does **not** preserve the original local timing pattern.

For this example, the model does **not** directly receive:

```text
activity at t=3, t=5, and t=6
```

Instead, it receives:

```text
this cell had total strength 3, so emit one spike at step 3
```

That is why latency encoding is best thought of as a compression and ranking scheme, not a direct replay of the original event stream.

## Step 7: Thresholding Behavior

Because the local call uses `clip=True`, cells below the latency threshold are removed.

Example from the same sample:

- some cells had `static_norm = 0.0043`
- those cells produced no spike token at all

So latency is also acting as a denoising threshold.

## One-Sentence Summary

In the local Event2Vec latency pipeline, each pooled cell's entire 20-step history is collapsed into one scalar strength, `spikegen.latency` converts that strength into a single spike time, and that spike becomes one Event2Vec token `[x, y, t, p, rho]`.
