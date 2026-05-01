from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_history(history: dict, out_dir: str | Path) -> None:
    out = _ensure_dir(out_dir)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc", []), label="train")
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(epochs, history.get("train_spike_count", []), label="train")
    if "val_spike_count" in history and history["val_spike_count"]:
        axes[2].plot(epochs, history["val_spike_count"], label="val")
    axes[2].set_title("Avg Output Spike Count")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: torch.Tensor,
    out_dir: str | Path,
    *,
    class_names: list[str] | None = None,
) -> None:
    out = _ensure_dir(out_dir)
    cm = confusion.detach().cpu().numpy().astype(np.float64)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    cm_norm = cm / denom

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Recall per class")
    ax.set_title("Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if class_names and len(class_names) == cm.shape[0]:
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)

    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_output_spike_raster(spk_rec: torch.Tensor, out_dir: str | Path) -> None:
    """Plot output spikes [T, B, C] for first sample in batch."""
    out = _ensure_dir(out_dir)
    if spk_rec.ndim != 3:
        return

    spikes = spk_rec[:, 0, :].detach().cpu().numpy()  # [T, C]
    t_idx, n_idx = np.where(spikes > 0)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.scatter(t_idx, n_idx, s=8)
    ax.set_title("Output Spike Raster (Sample 0)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Output neuron")
    fig.savefig(out / "output_spike_raster.png", dpi=160)
    plt.close(fig)


def save_metrics_text(metrics: dict, out_dir: str | Path) -> None:
    out = _ensure_dir(out_dir)
    lines = []
    for key in sorted(metrics):
        val = metrics[key]
        if isinstance(val, float):
            lines.append(f"{key}: {val:.6f}")
        else:
            lines.append(f"{key}: {val}")
    (out / "metrics.txt").write_text("\n".join(lines) + "\n")


__all__ = [
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_output_spike_raster",
    "save_metrics_text",
]
