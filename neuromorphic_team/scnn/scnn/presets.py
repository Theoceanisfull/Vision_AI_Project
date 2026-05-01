from __future__ import annotations

from .config import SNNConfig


def apply_encoding_preset(cfg: SNNConfig) -> None:
    """Apply encoding-specific training defaults for comparison runs.

    The sweep scripts start from a shared base config, then specialize it per
    encoding so each run uses a more appropriate objective and, for delta, a
    milder operating point.
    """

    encoding = cfg.data.encoding
    cfg.train.loss_kwargs = {}

    if encoding == "rate":
        cfg.train.loss = "ce_rate_loss"
        cfg.train.accuracy_fn = "accuracy_rate"
        return

    if encoding == "latency":
        cfg.train.loss = "ce_temporal_loss"
        cfg.train.accuracy_fn = "accuracy_temporal"
        return

    if encoding == "delta":
        cfg.train.loss = "ce_count_loss"
        cfg.train.accuracy_fn = "accuracy_rate"

        # Sparse signed delta inputs were collapsing under the shared defaults.
        cfg.data.delta_threshold = min(cfg.data.delta_threshold, 0.05)
        cfg.train.lr = min(cfg.train.lr, 5e-4)
        if cfg.train.grad_clip is None:
            cfg.train.grad_clip = 0.5
        else:
            cfg.train.grad_clip = min(cfg.train.grad_clip, 0.5)
        cfg.model.threshold = min(cfg.model.threshold, 0.75)
        cfg.model.dropout = min(cfg.model.dropout, 0.1)
        return

    raise ValueError(f"Unsupported encoding preset: {encoding}")


__all__ = ["apply_encoding_preset"]
