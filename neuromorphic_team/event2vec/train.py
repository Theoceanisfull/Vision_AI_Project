from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
import random
import subprocess
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

try:
    from models.scnn.result import plot_confusion_matrix, plot_training_history, save_metrics_text
except ModuleNotFoundError:
    from scnn.scnn.result import plot_confusion_matrix, plot_training_history, save_metrics_text

from .config import Event2VecConfig, default_config, resolve_project_path
from .data import build_asldvs_event2vec_dataloaders
from .e2v import Event2VecClassifier


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            try:
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.free,name",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                candidates: list[tuple[int, int, str]] = []
                for line in query.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",", maxsplit=2)]
                    if len(parts) != 3:
                        continue
                    index_str, free_mib_str, gpu_name = parts
                    candidates.append((int(index_str), int(free_mib_str), gpu_name))

                if candidates:
                    best_index, best_free_mib, gpu_name = max(candidates, key=lambda item: item[1])
                    free_gib = best_free_mib / 1024
                    print(f"Auto-selected GPU {best_index} ({gpu_name}) with {free_gib:.1f} GiB free")
                    return torch.device(f"cuda:{best_index}")
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                pass

            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device_cfg)


def build_model(cfg: Event2VecConfig, *, pooled_sensor_size: tuple[int, int]) -> nn.Module:
    height, width = pooled_sensor_size
    return Event2VecClassifier(
        num_classes=cfg.model.num_classes,
        height=height,
        width=width,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        pool_after_each_block=cfg.model.pool_after_each_block,
    )


def build_optimizer(cfg: Event2VecConfig, model: nn.Module) -> torch.optim.Optimizer:
    kwargs = {"lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_criterion(cfg: Event2VecConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)


def get_autocast_context(cfg: Event2VecConfig, device: torch.device):
    if not cfg.train.amp or device.type != "cuda":
        return nullcontext()

    dtype = torch.bfloat16 if cfg.train.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def update_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    for target, pred in zip(targets.view(-1), preds.view(-1)):
        confusion[int(target), int(pred)] += 1


def _iter_limited(
    loader,
    *,
    max_batches: int | None,
) -> Iterator:
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Event2VecConfig,
    *,
    max_batches: int | None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_token_count = 0.0
    total_batches = 0
    confusion = torch.zeros((cfg.model.num_classes, cfg.model.num_classes), dtype=torch.int64)
    for events, padding_mask, targets, token_counts in _iter_limited(loader, max_batches=max_batches):
        events = events.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with get_autocast_context(cfg, device):
            logits = model(events, padding_mask=padding_mask)
            loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean()

        update_confusion(confusion, preds.cpu(), targets.cpu())
        total_loss += float(loss.detach().cpu())
        total_acc += float(acc.detach().cpu())
        total_token_count += float(token_counts.mean().item())
        total_batches += 1

    if total_batches == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "token_count": 0.0,
            "confusion": confusion,
        }

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "token_count": total_token_count / total_batches,
        "confusion": confusion,
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Event2VecConfig,
) -> dict:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_token_count = 0.0
    total_batches = 0

    for batch_idx, (events, padding_mask, targets, token_counts) in enumerate(loader):
        if cfg.train.max_train_batches is not None and batch_idx >= cfg.train.max_train_batches:
            break

        events = events.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(cfg, device):
            logits = model(events, padding_mask=padding_mask)
            loss = criterion(logits, targets)

        loss.backward()
        if cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        preds = logits.detach().argmax(dim=1)
        acc = (preds == targets).float().mean()

        total_loss += float(loss.detach().cpu())
        total_acc += float(acc.detach().cpu())
        total_token_count += float(token_counts.mean().item())
        total_batches += 1

        if cfg.train.log_every > 0 and (batch_idx + 1) % cfg.train.log_every == 0:
            print(
                f"  train batch {batch_idx + 1}: "
                f"loss={total_loss / total_batches:.4f}, "
                f"acc={total_acc / total_batches:.4f}, "
                f"tokens={total_token_count / total_batches:.2f}"
            )

    if total_batches == 0:
        return {"loss": 0.0, "acc": 0.0, "token_count": 0.0}

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "token_count": total_token_count / total_batches,
    }


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _is_better_epoch(current_val: dict, best_val: dict | None) -> bool:
    if best_val is None:
        return True
    if current_val["acc"] > best_val["acc"]:
        return True
    if current_val["acc"] < best_val["acc"]:
        return False
    if current_val["loss"] < best_val["loss"]:
        return True
    if current_val["loss"] > best_val["loss"]:
        return False
    return current_val["token_count"] < best_val["token_count"]


def run_training(cfg: Event2VecConfig) -> dict:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_random_seed(cfg.data.seed)

    run_dir = resolve_project_path(cfg.result.out_dir) / cfg.result.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.train.device)
    print(f"Device: {device}")

    loaders = build_asldvs_event2vec_dataloaders(
        data_root=cfg.data.data_root,
        encoding=cfg.data.encoding,
        sensor_size=cfg.data.sensor_size,
        pool_kernel=cfg.data.pool_kernel,
        num_steps=cfg.data.num_steps,
        max_tokens=cfg.data.max_tokens,
        batch_size=cfg.data.batch_size,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        delta_threshold=cfg.data.delta_threshold,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
    )

    class_to_idx = loaders["train"].dataset.class_to_idx  # type: ignore[attr-defined]
    cfg.model.num_classes = len(class_to_idx)
    class_names = sorted(class_to_idx, key=lambda key: class_to_idx[key])

    pooled_sensor_size = (
        loaders["train"].dataset.pooled_height,  # type: ignore[attr-defined]
        loaders["train"].dataset.pooled_width,  # type: ignore[attr-defined]
    )

    model = build_model(cfg, pooled_sensor_size=pooled_sensor_size).to(device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_spike_count": [],
        "val_spike_count": [],
        "train_token_count": [],
        "val_token_count": [],
    }
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: dict | None = None

    for epoch in range(cfg.train.epochs):
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            cfg,
        )

        val_metrics = evaluate_model(
            model,
            loaders["val"],
            criterion,
            device,
            cfg,
            max_batches=cfg.train.max_val_batches,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_spike_count"].append(train_metrics["token_count"])
        history["val_spike_count"].append(val_metrics["token_count"])
        history["train_token_count"].append(train_metrics["token_count"])
        history["val_token_count"].append(val_metrics["token_count"])

        print(
            "  "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"train_tokens={train_metrics['token_count']:.2f}, val_tokens={val_metrics['token_count']:.2f}"
        )

        if _is_better_epoch(val_metrics, best_val_metrics):
            best_epoch = epoch + 1
            best_val_metrics = {
                "loss": val_metrics["loss"],
                "acc": val_metrics["acc"],
                "token_count": val_metrics["token_count"],
            }
            best_state_dict = _clone_state_dict_to_cpu(model)
            print(
                "  "
                f"new_best epoch={best_epoch}, val_acc={val_metrics['acc']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}"
            )

    last_test_metrics = evaluate_model(
        model,
        loaders["test"],
        criterion,
        device,
        cfg,
        max_batches=cfg.train.max_test_batches,
    )
    last_state_dict = _clone_state_dict_to_cpu(model)

    last_summary = {
        "test_loss": last_test_metrics["loss"],
        "test_acc": last_test_metrics["acc"],
        "test_spike_count": last_test_metrics["token_count"],
        "test_token_count": last_test_metrics["token_count"],
        "num_classes": cfg.model.num_classes,
        "device": str(device),
        "checkpoint": "last",
        "selected_epoch": cfg.train.epochs,
        "pooled_sensor_size": list(pooled_sensor_size),
    }

    if best_state_dict is None or best_val_metrics is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_val_metrics = {
            "loss": 0.0,
            "acc": 0.0,
            "token_count": 0.0,
        }
        best_epoch = cfg.train.epochs

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_model(
        model,
        loaders["test"],
        criterion,
        device,
        cfg,
        max_batches=cfg.train.max_test_batches,
    )

    summary = {
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_spike_count": test_metrics["token_count"],
        "test_token_count": test_metrics["token_count"],
        "num_classes": cfg.model.num_classes,
        "device": str(device),
        "checkpoint": "best",
        "selected_epoch": best_epoch,
        "best_val_loss": best_val_metrics["loss"],
        "best_val_acc": best_val_metrics["acc"],
        "best_val_spike_count": best_val_metrics["token_count"],
        "best_val_token_count": best_val_metrics["token_count"],
        "last_test_loss": last_summary["test_loss"],
        "last_test_acc": last_summary["test_acc"],
        "last_test_spike_count": last_summary["test_spike_count"],
        "last_test_token_count": last_summary["test_token_count"],
        "pooled_sensor_size": list(pooled_sensor_size),
    }

    cfg.save_json(run_dir / "config_resolved.json")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    if cfg.result.save_checkpoint:
        torch.save(
            {
                "model_state_dict": last_state_dict,
                "config": cfg.to_dict(),
                "history": history,
                "test_summary": last_summary,
                "class_to_idx": class_to_idx,
            },
            run_dir / "checkpoint_last.pt",
        )
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "config": cfg.to_dict(),
                "history": history,
                "test_summary": summary,
                "class_to_idx": class_to_idx,
                "selection": {
                    "checkpoint": "best",
                    "criterion": "val_acc_then_val_loss",
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                },
            },
            run_dir / "checkpoint_best.pt",
        )

    save_metrics_text(summary, run_dir)
    if cfg.result.save_plots:
        plot_training_history(history, run_dir)
        plot_confusion_matrix(test_metrics["confusion"], run_dir, class_names=class_names)

    print(
        f"Selected checkpoint=best (epoch {best_epoch}); "
        f"Test: loss={summary['test_loss']:.4f}, acc={summary['test_acc']:.4f}"
    )
    return {
        "history": history,
        "test": summary,
        "test_last": last_summary,
        "selection": {
            "checkpoint": "best",
            "criterion": "val_acc_then_val_loss",
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
        },
        "run_dir": str(run_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Event2Vec on ASL-DVS")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument(
        "--write-default-config",
        type=str,
        default=None,
        help="Write the default JSON config to the given path and exit",
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Override the config for a tiny smoke test run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_default_config:
        cfg = default_config()
        cfg.save_json(args.write_default_config)
        print(f"Wrote default config to {args.write_default_config}")
        return

    cfg = Event2VecConfig.from_json(args.config) if args.config else default_config()

    if args.quick_smoke:
        cfg.result.run_name = f"{cfg.result.run_name}_smoke"
        cfg.train.epochs = 1
        cfg.data.batch_size = 4
        cfg.data.max_tokens = min(cfg.data.max_tokens, 256)
        cfg.train.max_train_batches = 2
        cfg.train.max_val_batches = 1
        cfg.train.max_test_batches = 1
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False

    run_training(cfg)


if __name__ == "__main__":
    main()
