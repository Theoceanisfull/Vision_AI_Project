from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch.backprop as snn_backprop
import snntorch.functional as SF

from .config import SNNConfig, default_config, resolve_project_path
from .model import build_model
from .result import (
    plot_confusion_matrix,
    plot_output_spike_raster,
    plot_training_history,
    save_metrics_text,
)
from .spike_data import build_asldvs_dataloaders


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")

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
                idx_str, free_mib_str, gpu_name = parts
                candidates.append((int(idx_str), int(free_mib_str), gpu_name))

            if candidates:
                best_index, best_free_mib, gpu_name = max(candidates, key=lambda item: item[1])
                free_gib = best_free_mib / 1024
                print(f"Auto-selected GPU {best_index} ({gpu_name}) with {free_gib:.1f} GiB free")
                return torch.device(f"cuda:{best_index}")
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass

        return torch.device("cuda")
    return torch.device(device_cfg)


def build_optimizer(cfg: SNNConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    kwargs = {"lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_criterion(cfg: SNNConfig) -> Callable:
    try:
        fn = getattr(SF, cfg.train.loss)
    except AttributeError as exc:
        raise ValueError(f"Unsupported loss function: {cfg.train.loss}") from exc
    return fn(**cfg.train.loss_kwargs)


def build_regularizer(cfg: SNNConfig) -> Callable | None:
    if cfg.train.regularizer == "none":
        return None
    try:
        fn = getattr(SF, cfg.train.regularizer)
    except AttributeError as exc:
        raise ValueError(f"Unsupported regularizer: {cfg.train.regularizer}") from exc
    return fn(**cfg.train.regularizer_kwargs)


def build_accuracy_fn(cfg: SNNConfig) -> Callable:
    if cfg.train.accuracy_fn == "accuracy_rate":
        return SF.accuracy_rate
    if cfg.train.accuracy_fn == "accuracy_temporal":
        return SF.accuracy_temporal
    raise ValueError(f"Unsupported accuracy function: {cfg.train.accuracy_fn}")


def criterion_uses_membrane(criterion: Callable) -> bool:
    name = getattr(criterion, "__name__", "")
    return "membrane" in name


def compute_loss(
    criterion: Callable,
    spk_rec: torch.Tensor,
    mem_rec: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if criterion_uses_membrane(criterion):
        return criterion(mem_rec, targets)
    return criterion(spk_rec, targets)


def update_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion[int(t), int(p)] += 1


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    accuracy_fn: Callable,
    device: torch.device,
    *,
    max_batches: int | None,
    num_classes: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_spikes = 0.0
    total_batches = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    last_spk_rec = None

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        spk_rec, mem_rec = model(inputs, time_first=False, reset=True)
        loss = compute_loss(criterion, spk_rec, mem_rec, targets)
        acc = accuracy_fn(spk_rec, targets)

        preds = spk_rec.sum(dim=0).argmax(dim=1)
        update_confusion(confusion, preds.cpu(), targets.cpu())

        total_loss += float(loss.detach().cpu())
        total_acc += float(acc)
        total_spikes += float(spk_rec.detach().sum().cpu() / targets.numel())
        total_batches += 1
        last_spk_rec = spk_rec.detach().cpu()

    if total_batches == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "spike_count": 0.0,
            "confusion": confusion,
            "last_spk_rec": last_spk_rec,
        }

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "spike_count": total_spikes / total_batches,
        "confusion": confusion,
        "last_spk_rec": last_spk_rec,
    }


def train_one_epoch_manual(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    regularizer: Callable | None,
    accuracy_fn: Callable,
    device: torch.device,
    *,
    max_batches: int | None,
    grad_clip: float | None,
    reg_weight: float,
    log_every: int,
) -> dict:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_spikes = 0.0
    total_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        spk_rec, mem_rec = model(inputs, time_first=False, reset=True)

        loss = compute_loss(criterion, spk_rec, mem_rec, targets)
        if regularizer is not None:
            loss = loss + reg_weight * regularizer(spk_rec)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc = accuracy_fn(spk_rec.detach(), targets)
        spike_count = float(spk_rec.detach().sum().cpu() / targets.numel())

        total_loss += float(loss.detach().cpu())
        total_acc += float(acc)
        total_spikes += spike_count
        total_batches += 1

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            print(
                f"  train batch {batch_idx + 1}: "
                f"loss={total_loss / total_batches:.4f}, "
                f"acc={total_acc / total_batches:.4f}, "
                f"spikes={total_spikes / total_batches:.2f}"
            )

    if total_batches == 0:
        return {"loss": 0.0, "acc": 0.0, "spike_count": 0.0}

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "spike_count": total_spikes / total_batches,
    }


def train_one_epoch_backprop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    regularizer: Callable | None,
    device: torch.device,
    *,
    mode: str,
    tbptt_k: int,
) -> float:
    reg = regularizer if regularizer is not None else False

    if mode == "backprop_bptt":
        loss = snn_backprop.BPTT(
            model,
            loader,
            optimizer,
            criterion,
            time_var=True,
            time_first=False,
            regularization=reg,
            device=str(device),
        )
    elif mode == "backprop_rtrl":
        loss = snn_backprop.RTRL(
            model,
            loader,
            optimizer,
            criterion,
            time_var=True,
            time_first=False,
            regularization=reg,
            device=str(device),
        )
    elif mode == "backprop_tbptt":
        loss = snn_backprop.TBPTT(
            model,
            loader,
            optimizer,
            criterion,
            time_var=True,
            time_first=False,
            regularization=reg,
            device=str(device),
            K=tbptt_k,
        )
    else:
        raise ValueError(f"Unsupported backprop mode: {mode}")

    return float(loss.detach().cpu())


def _limited_loader(loader: DataLoader, max_batches: int | None) -> DataLoader:
    if max_batches is None:
        return loader

    class _Wrapper:
        def __iter__(self_inner):
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                yield batch

        def __len__(self_inner):
            return min(len(loader), max_batches)

    return _Wrapper()  # type: ignore[return-value]


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _is_better_epoch(
    current_val: dict,
    best_val: dict | None,
) -> bool:
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

    return current_val["spike_count"] < best_val["spike_count"]


def run_training(cfg: SNNConfig) -> dict:
    run_dir = resolve_project_path(cfg.result.out_dir) / cfg.result.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.train.device)
    print(f"Device: {device}")

    loaders = build_asldvs_dataloaders(
        data_root=cfg.data.data_root,
        encoding=cfg.data.encoding,
        sensor_size=cfg.data.sensor_size,
        num_steps=cfg.data.num_steps,
        batch_size=cfg.data.batch_size,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        delta_threshold=cfg.data.delta_threshold,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    class_to_idx = loaders["train"].dataset.class_to_idx  # type: ignore[attr-defined]
    cfg.model.num_classes = len(class_to_idx)
    class_names = sorted(class_to_idx, key=lambda k: class_to_idx[k])

    model = build_model(cfg.model, sensor_size=cfg.data.sensor_size).to(device)
    optimizer = build_optimizer(cfg, model)
    criterion = build_criterion(cfg)
    regularizer = build_regularizer(cfg)
    accuracy_fn = build_accuracy_fn(cfg)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_spike_count": [],
        "val_spike_count": [],
    }
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_metrics: dict | None = None

    for epoch in range(cfg.train.epochs):
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        if cfg.train.mode == "manual":
            train_metrics = train_one_epoch_manual(
                model,
                loaders["train"],
                optimizer,
                criterion,
                regularizer,
                accuracy_fn,
                device,
                max_batches=cfg.train.max_train_batches,
                grad_clip=cfg.train.grad_clip,
                reg_weight=cfg.train.regularizer_weight,
                log_every=cfg.train.log_every,
            )
        else:
            train_loader_limited = _limited_loader(loaders["train"], cfg.train.max_train_batches)
            train_loss = train_one_epoch_backprop(
                model,
                train_loader_limited,
                optimizer,
                criterion,
                regularizer,
                device,
                mode=cfg.train.mode,
                tbptt_k=cfg.train.tbptt_k,
            )
            # Accuracy for backprop mode is estimated on a short training subset.
            train_eval = evaluate_model(
                model,
                loaders["train"],
                criterion,
                accuracy_fn,
                device,
                max_batches=cfg.train.max_train_batches if cfg.train.max_train_batches is not None else 10,
                num_classes=cfg.model.num_classes,
            )
            train_metrics = {
                "loss": train_loss,
                "acc": train_eval["acc"],
                "spike_count": train_eval["spike_count"],
            }

        val_metrics = evaluate_model(
            model,
            loaders["val"],
            criterion,
            accuracy_fn,
            device,
            max_batches=cfg.train.max_val_batches,
            num_classes=cfg.model.num_classes,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_spike_count"].append(train_metrics["spike_count"])
        history["val_spike_count"].append(val_metrics["spike_count"])

        print(
            "  "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"train_spikes={train_metrics['spike_count']:.2f}, val_spikes={val_metrics['spike_count']:.2f}"
        )

        if _is_better_epoch(val_metrics, best_val_metrics):
            best_epoch = epoch + 1
            best_val_metrics = {
                "loss": val_metrics["loss"],
                "acc": val_metrics["acc"],
                "spike_count": val_metrics["spike_count"],
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
        accuracy_fn,
        device,
        max_batches=cfg.train.max_test_batches,
        num_classes=cfg.model.num_classes,
    )

    last_summary = {
        "test_loss": last_test_metrics["loss"],
        "test_acc": last_test_metrics["acc"],
        "test_spike_count": last_test_metrics["spike_count"],
        "num_classes": cfg.model.num_classes,
        "device": str(device),
        "checkpoint": "last",
        "selected_epoch": cfg.train.epochs,
    }

    last_state_dict = _clone_state_dict_to_cpu(model)

    if best_state_dict is None or best_val_metrics is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_val_metrics = {
            "loss": 0.0,
            "acc": 0.0,
            "spike_count": 0.0,
        }
        best_epoch = cfg.train.epochs

    model.load_state_dict(best_state_dict)

    test_metrics = evaluate_model(
        model,
        loaders["test"],
        criterion,
        accuracy_fn,
        device,
        max_batches=cfg.train.max_test_batches,
        num_classes=cfg.model.num_classes,
    )

    summary = {
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_spike_count": test_metrics["spike_count"],
        "num_classes": cfg.model.num_classes,
        "device": str(device),
        "checkpoint": "best",
        "selected_epoch": best_epoch,
        "best_val_loss": best_val_metrics["loss"],
        "best_val_acc": best_val_metrics["acc"],
        "best_val_spike_count": best_val_metrics["spike_count"],
        "last_test_loss": last_summary["test_loss"],
        "last_test_acc": last_summary["test_acc"],
        "last_test_spike_count": last_summary["test_spike_count"],
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
        if test_metrics["last_spk_rec"] is not None:
            plot_output_spike_raster(test_metrics["last_spk_rec"], run_dir)

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
    parser = argparse.ArgumentParser(description="Train Conv-SNN on ASL-DVS")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument(
        "--write-default-config",
        type=str,
        default=None,
        help="Write default JSON config to path and exit",
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Override config for a tiny smoke test run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_default_config:
        cfg = default_config()
        cfg.save_json(args.write_default_config)
        print(f"Wrote default config to {args.write_default_config}")
        return

    cfg = SNNConfig.from_json(args.config) if args.config else default_config()

    if args.quick_smoke:
        cfg.result.run_name = cfg.result.run_name + "_smoke"
        cfg.train.epochs = 1
        cfg.data.batch_size = 4
        cfg.data.num_steps = 8
        cfg.train.max_train_batches = 2
        cfg.train.max_val_batches = 1
        cfg.train.max_test_batches = 1

    run_training(cfg)


if __name__ == "__main__":
    main()
