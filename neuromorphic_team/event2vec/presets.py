from __future__ import annotations

from .config import Event2VecConfig


def apply_encoding_preset(cfg: Event2VecConfig) -> None:
    encoding = cfg.data.encoding

    if encoding == "rate":
        cfg.data.max_tokens = 1024
        cfg.data.batch_size = 32
        cfg.train.lr = 3e-4
        return

    if encoding == "latency":
        cfg.data.max_tokens = 1024
        cfg.data.batch_size = 32
        cfg.train.lr = 3e-4
        return

    if encoding == "delta":
        cfg.data.max_tokens = 1280
        cfg.data.batch_size = 24
        cfg.train.lr = 2e-4
        cfg.train.grad_clip = 0.75
        return

    raise ValueError(f"Unsupported encoding preset: {encoding}")


__all__ = ["apply_encoding_preset"]
