from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat
from snntorch import spikegen
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import EncodingMode, resolve_project_path


@dataclass(frozen=True)
class SplitDatasets:
    train: Dataset
    val: Dataset
    test: Dataset


class ASLDVSEvent2VecDataset(Dataset):
    def __init__(
        self,
        sample_paths: Sequence[Path],
        class_to_idx: dict[str, int],
        *,
        encoding: EncodingMode = "rate",
        sensor_size: tuple[int, int] = (180, 240),
        pool_kernel: tuple[int, int] = (6, 8),
        num_steps: int = 20,
        max_tokens: int = 1024,
        delta_threshold: float = 0.1,
    ) -> None:
        if not sample_paths:
            raise ValueError("sample_paths is empty")
        if encoding not in ("rate", "latency", "delta"):
            raise ValueError(f"Unsupported encoding: {encoding}")
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")

        self.sample_paths = [Path(path) for path in sample_paths]
        self.class_to_idx = class_to_idx
        self.encoding = encoding
        self.height, self.width = sensor_size
        self.pool_kernel = tuple(int(v) for v in pool_kernel)
        self.num_steps = int(num_steps)
        self.max_tokens = int(max_tokens)
        self.delta_threshold = float(delta_threshold)

        pool_h, pool_w = self.pool_kernel
        if pool_h <= 0 or pool_w <= 0:
            raise ValueError("pool_kernel values must be > 0")
        self.pooled_height = self.height // pool_h
        self.pooled_width = self.width // pool_w
        if self.pooled_height <= 0 or self.pooled_width <= 0:
            raise ValueError("pool_kernel is larger than the sensor size")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path = self.sample_paths[index]
        label = self.class_to_idx[path.parent.name]

        x, y, ts, pol = self._load_events(path)
        raw_frames = self._events_to_temporal_channels(x, y, ts, pol)
        pooled_raw, pooled_norm = self._pool_frames(raw_frames)
        encoded, rho_source = self._encode_frames(pooled_raw, pooled_norm)
        tokens = self._spikes_to_tokens(encoded, rho_source)

        if tokens.shape[0] == 0:
            tokens = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

        token_count = int(tokens.shape[0])
        return tokens.to(torch.float32), label, token_count

    def _load_events(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = loadmat(path)
        x = mat["x"].reshape(-1).astype(np.int64)
        y = mat["y"].reshape(-1).astype(np.int64)
        ts = mat["ts"].reshape(-1).astype(np.int64)
        pol = mat["pol"].reshape(-1).astype(np.int8)

        y = self.height - 1 - y

        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
        )
        if not np.any(valid):
            return (
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int8),
            )

        return x[valid], y[valid], ts[valid], pol[valid]

    def _events_to_temporal_channels(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        pol: np.ndarray,
    ) -> torch.Tensor:
        frames = np.zeros((self.num_steps, 2, self.height, self.width), dtype=np.float32)

        t_min = int(ts.min())
        t_max = int(ts.max())
        if t_max <= t_min:
            bins = np.zeros_like(ts, dtype=np.int64)
        else:
            edges = np.linspace(t_min, t_max + 1, self.num_steps + 1, dtype=np.int64)
            bins = np.searchsorted(edges, ts, side="right") - 1
            bins = np.clip(bins, 0, self.num_steps - 1)

        on_mask = pol > 0
        off_mask = ~on_mask

        np.add.at(frames, (bins[on_mask], 0, y[on_mask], x[on_mask]), 1.0)
        np.add.at(frames, (bins[off_mask], 1, y[off_mask], x[off_mask]), 1.0)

        return torch.from_numpy(frames)

    def _pool_frames(self, raw_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pool_h, pool_w = self.pool_kernel
        kernel_area = float(pool_h * pool_w)

        raw_frames = raw_frames.to(torch.float32)
        max_val = float(raw_frames.max())
        if max_val > 0:
            norm_frames = raw_frames / max_val
        else:
            norm_frames = raw_frames.clone()

        pooled_raw = F.avg_pool2d(raw_frames, kernel_size=self.pool_kernel, stride=self.pool_kernel) * kernel_area
        pooled_norm = F.avg_pool2d(norm_frames, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        return pooled_raw, pooled_norm

    def _encode_frames(
        self,
        pooled_raw: torch.Tensor,
        pooled_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoding == "rate":
            encoded = spikegen.rate(pooled_norm, time_var_input=True)
            rho_source = pooled_raw
            return encoded.to(torch.float32), rho_source.to(torch.float32)

        if self.encoding == "latency":
            static_raw = pooled_raw.sum(dim=0)
            static_norm = pooled_norm.sum(dim=0)
            max_val = float(static_norm.max())
            if max_val > 0:
                static_norm = static_norm / max_val
            encoded = spikegen.latency(
                static_norm,
                num_steps=self.num_steps,
                normalize=True,
                clip=True,
            )
            rho_source = static_raw.unsqueeze(0).repeat(self.num_steps, 1, 1, 1)
            return encoded.to(torch.float32), rho_source.to(torch.float32)

        encoded = spikegen.delta(
            pooled_norm,
            threshold=self.delta_threshold,
            padding=True,
            off_spike=True,
        )
        delta_strength = torch.zeros_like(pooled_raw)
        delta_strength[0] = pooled_raw[0].abs()
        delta_strength[1:] = (pooled_raw[1:] - pooled_raw[:-1]).abs()
        return encoded.to(torch.float32), delta_strength.to(torch.float32)

    def _spikes_to_tokens(
        self,
        encoded: torch.Tensor,
        rho_source: torch.Tensor,
    ) -> torch.Tensor:
        nonzero = encoded != 0
        if not bool(nonzero.any()):
            return torch.zeros((0, 5), dtype=torch.float32)

        time_idx, channel_idx, y_idx, x_idx = nonzero.nonzero(as_tuple=True)
        values = encoded[time_idx, channel_idx, y_idx, x_idx]
        rho = rho_source[time_idx, channel_idx, y_idx, x_idx].clamp_min(1.0)

        if self.encoding == "delta":
            polarity = torch.where(values > 0, channel_idx, 1 - channel_idx)
        else:
            polarity = channel_idx

        tokens = torch.stack(
            [
                x_idx.to(torch.float32),
                y_idx.to(torch.float32),
                time_idx.to(torch.float32) / float(max(self.num_steps - 1, 1)),
                polarity.to(torch.float32),
                rho.to(torch.float32),
            ],
            dim=-1,
        )

        return self._limit_tokens(tokens=tokens, time_idx=time_idx, rho=rho)

    def _limit_tokens(
        self,
        *,
        tokens: torch.Tensor,
        time_idx: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[0] <= self.max_tokens:
            return tokens

        per_step_cap = max(1, math.ceil(self.max_tokens / self.num_steps))
        kept_indices: list[torch.Tensor] = []

        for step in range(self.num_steps):
            step_indices = torch.nonzero(time_idx == step, as_tuple=False).squeeze(1)
            if step_indices.numel() == 0:
                continue
            if step_indices.numel() > per_step_cap:
                scores = rho[step_indices]
                topk = torch.topk(scores, k=per_step_cap, largest=True, sorted=False).indices
                step_indices = step_indices[topk]
            kept_indices.append(step_indices)

        selected = torch.cat(kept_indices, dim=0) if kept_indices else torch.arange(tokens.shape[0])
        if selected.numel() > self.max_tokens:
            scores = rho[selected]
            topk = torch.topk(scores, k=self.max_tokens, largest=True, sorted=False).indices
            selected = selected[topk]

        order = torch.argsort(time_idx[selected] * (self.pooled_height * self.pooled_width * 2) + selected)
        return tokens[selected[order]]


def collate_event_sequences(
    batch: Sequence[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    events_list, labels, token_counts = zip(*batch)
    batch_size = len(events_list)
    max_len = max(events.shape[0] for events in events_list)

    events = torch.zeros((batch_size, max_len, 5), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for idx, sample in enumerate(events_list):
        length = sample.shape[0]
        events[idx, :length] = sample
        padding_mask[idx, :length] = False

    targets = torch.tensor(labels, dtype=torch.long)
    counts = torch.tensor(token_counts, dtype=torch.float32)
    return events, padding_mask, targets, counts


def _collect_samples(data_root: Path) -> tuple[list[Path], dict[str, int]]:
    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {data_root}")

    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
    sample_paths: list[Path] = []

    for class_dir in class_dirs:
        sample_paths.extend(sorted(class_dir.glob("*.mat")))

    if not sample_paths:
        raise FileNotFoundError(f"No .mat files found under {data_root}")

    return sample_paths, class_to_idx


def _stratified_split(
    sample_paths: Sequence[Path],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Path]] = {}
    for sample_path in sample_paths:
        by_class.setdefault(sample_path.parent.name, []).append(sample_path)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for class_name, class_paths in sorted(by_class.items()):
        class_paths = list(class_paths)
        rng.shuffle(class_paths)

        total = len(class_paths)
        num_train = int(total * train_ratio)
        num_val = int(total * val_ratio)

        if total >= 3:
            num_train = max(1, min(num_train, total - 2))
            num_val = max(1, min(num_val, total - num_train - 1))

        train.extend(class_paths[:num_train])
        val.extend(class_paths[num_train : num_train + num_val])
        test.extend(class_paths[num_train + num_val :])

    return train, val, test


def build_asldvs_event2vec_splits(
    data_root: str | Path = "data/ASLDVS",
    *,
    encoding: EncodingMode = "rate",
    sensor_size: tuple[int, int] = (180, 240),
    pool_kernel: tuple[int, int] = (6, 8),
    num_steps: int = 20,
    max_tokens: int = 1024,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    delta_threshold: float = 0.1,
) -> SplitDatasets:
    root = resolve_project_path(data_root).resolve()
    sample_paths, class_to_idx = _collect_samples(root)
    train_paths, val_paths, test_paths = _stratified_split(
        sample_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    kwargs = {
        "class_to_idx": class_to_idx,
        "encoding": encoding,
        "sensor_size": sensor_size,
        "pool_kernel": pool_kernel,
        "num_steps": num_steps,
        "max_tokens": max_tokens,
        "delta_threshold": delta_threshold,
    }

    return SplitDatasets(
        train=ASLDVSEvent2VecDataset(train_paths, **kwargs),
        val=ASLDVSEvent2VecDataset(val_paths, **kwargs),
        test=ASLDVSEvent2VecDataset(test_paths, **kwargs),
    )


def build_asldvs_event2vec_dataloaders(
    data_root: str | Path = "data/ASLDVS",
    *,
    encoding: EncodingMode = "rate",
    sensor_size: tuple[int, int] = (180, 240),
    pool_kernel: tuple[int, int] = (6, 8),
    num_steps: int = 20,
    max_tokens: int = 1024,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    delta_threshold: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> dict[str, DataLoader]:
    splits = build_asldvs_event2vec_splits(
        data_root=data_root,
        encoding=encoding,
        sensor_size=sensor_size,
        pool_kernel=pool_kernel,
        num_steps=num_steps,
        max_tokens=max_tokens,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        delta_threshold=delta_threshold,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_event_sequences,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    return {
        "train": DataLoader(
            splits.train,
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            splits.val,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            splits.test,
            shuffle=False,
            **loader_kwargs,
        ),
    }


__all__ = [
    "ASLDVSEvent2VecDataset",
    "SplitDatasets",
    "build_asldvs_event2vec_dataloaders",
    "build_asldvs_event2vec_splits",
    "collate_event_sequences",
]
