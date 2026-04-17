from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTGAN_CODE_ROOT = PROJECT_ROOT / "EventGAN" / "EventGAN"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVENTGAN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENTGAN_CODE_ROOT))

from event2vec.config import Event2VecConfig, default_config, resolve_project_path
from event2vec.data import ASLDVSEvent2VecDataset, collate_event_sequences
from event2vec.train import build_model, resolve_device, run_training

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def chunked(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def discover_event2vec_checkpoint(run_name: str | None = None) -> Path | None:
    runs_root = PROJECT_ROOT / "runs" / "event2vec"
    if not runs_root.exists():
        return None

    if run_name:
        candidate = runs_root / run_name / "checkpoint_best.pt"
        if candidate.exists():
            return candidate

    candidates = [
        path
        for path in runs_root.glob("*/checkpoint_best.pt")
        if "smoke" not in path.parent.name.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def train_event2vec_if_needed(args: argparse.Namespace) -> Path:
    if args.event2vec_checkpoint:
        checkpoint = resolve_project_path(args.event2vec_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Event2Vec checkpoint not found: {checkpoint}")
        return checkpoint

    existing = discover_event2vec_checkpoint(args.event2vec_run_name)
    if existing and not args.force_train_event2vec:
        print(f"Using Event2Vec checkpoint: {existing}")
        return existing

    cfg = default_config()
    cfg.data.encoding = args.encoding
    cfg.data.batch_size = args.event2vec_batch_size
    cfg.data.max_tokens = args.event2vec_max_tokens
    cfg.data.num_workers = args.event2vec_workers
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = args.event2vec_workers > 0
    cfg.train.epochs = args.event2vec_epochs
    cfg.train.max_train_batches = args.event2vec_max_train_batches
    cfg.train.max_val_batches = args.event2vec_max_val_batches
    cfg.train.max_test_batches = args.event2vec_max_test_batches
    cfg.train.device = args.device
    cfg.train.amp = False
    cfg.train.log_every = args.event2vec_log_every
    cfg.result.run_name = args.event2vec_run_name

    print(
        "Training Event2Vec because no reusable checkpoint was found. "
        f"run={cfg.result.run_name}, epochs={cfg.train.epochs}, "
        f"max_train_batches={cfg.train.max_train_batches}"
    )
    result = run_training(cfg)
    checkpoint = Path(result["run_dir"]) / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Event2Vec training finished without {checkpoint}")
    return checkpoint


def load_event2vec(checkpoint_path: Path, device_name: str) -> tuple[torch.nn.Module, Event2VecConfig, dict[str, int], torch.device]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = Event2VecConfig.from_dict(checkpoint["config"])
    class_to_idx = dict(checkpoint["class_to_idx"])
    cfg.model.num_classes = len(class_to_idx)

    if "pooled_sensor_size" in checkpoint.get("test_summary", {}):
        pooled_sensor_size = tuple(checkpoint["test_summary"]["pooled_sensor_size"])
    else:
        sensor_h, sensor_w = cfg.data.sensor_size
        pool_h, pool_w = cfg.data.pool_kernel
        pooled_sensor_size = (sensor_h // pool_h, sensor_w // pool_w)

    device = resolve_device(device_name)
    model = build_model(cfg, pooled_sensor_size=pooled_sensor_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, class_to_idx, device


def load_eventgan(checkpoint_dir: Path) -> Any:
    if not EVENTGAN_CODE_ROOT.exists():
        raise FileNotFoundError(
            f"EventGAN code was not found under {EVENTGAN_CODE_ROOT}. "
            "Clone the alexzzhu/EventGAN repository into Vision_AI_Project/EventGAN first."
        )

    from models.eventgan_base import EventGANBase

    options = SimpleNamespace(
        n_image_channels=1,
        n_time_bins=9,
        sn=True,
        checkpoint_dir=str(checkpoint_dir.resolve()),
    )
    return EventGANBase(options)


def motion_for_image(path: Path, shift_px: int, seed: int) -> tuple[float, float, float, float]:
    stem = path.stem.lower()
    if "_left_" in stem:
        return -shift_px, 0.0, 0.0, 1.0
    if "_right_" in stem:
        return shift_px, 0.0, 0.0, 1.0
    if "_top_" in stem:
        return 0.0, -shift_px, 0.0, 1.0
    if "_bot_" in stem or "_bottom_" in stem:
        return 0.0, shift_px, 0.0, 1.0

    choices = [
        (shift_px, -shift_px, 2.5, 1.02),
        (-shift_px, shift_px, -2.5, 1.02),
        (shift_px, shift_px, 0.0, 0.98),
        (-shift_px, -shift_px, 0.0, 0.98),
    ]
    index = stable_seed(str(path), seed) % len(choices)
    return choices[index]


def image_pair_tensor(
    path: Path,
    *,
    sensor_size: tuple[int, int],
    motion_shift: int,
    seed: int,
) -> np.ndarray:
    height, width = sensor_size
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    dx, dy, angle, scale = motion_for_image(path, motion_shift, seed)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    moved = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    pair = np.stack([image, moved]).astype(np.float32)
    return (pair / 255.0 - 0.5) * 2.0


def collect_image_paths(
    image_root: Path,
    class_names: list[str],
    *,
    samples_per_class: int | None,
    seed: int,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    image_paths: list[Path] = []
    for class_name in class_names:
        class_dir = image_root / class_name
        if not class_dir.exists():
            continue
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if samples_per_class is not None and len(paths) > samples_per_class:
            selected = rng.choice(len(paths), size=samples_per_class, replace=False)
            paths = [paths[int(index)] for index in sorted(selected)]
        image_paths.extend(paths)
    return image_paths


def resolve_image_root_for_classes(image_root: Path, class_names: list[str]) -> Path:
    candidates = [image_root]
    candidates.extend(path for path in image_root.rglob("*") if path.is_dir() and path != image_root)

    best_path = image_root
    best_score = -1
    for candidate in candidates:
        score = sum(1 for class_name in class_names if (candidate / class_name).is_dir())
        if score > best_score:
            best_score = score
            best_path = candidate
    return best_path


def collect_generated_paths(
    generated_root: Path,
    class_names: list[str],
    *,
    samples_per_class: int | None,
    seed: int,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    generated_paths: list[Path] = []
    for class_name in class_names:
        class_dir = generated_root / class_name
        if not class_dir.exists():
            continue
        paths = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mat")
        if samples_per_class is not None and len(paths) > samples_per_class:
            selected = rng.choice(len(paths), size=samples_per_class, replace=False)
            paths = [paths[int(index)] for index in sorted(selected)]
        generated_paths.extend(paths)
    return generated_paths


def event_volume_to_arrays(
    volume: torch.Tensor,
    *,
    target_events: int,
    duration_us: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    vol = volume.detach().float().cpu().numpy()
    vol = np.maximum(vol, 0.0)
    height, width = vol.shape[-2:]
    weights = vol.reshape(-1)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        x = rng.integers(0, width, size=1, dtype=np.int16)
        y = rng.integers(0, height, size=1, dtype=np.int16)
        y_raw = (height - 1 - y).astype(np.int16)
        return {
            "x": x.reshape(-1, 1),
            # Store y in the same bottom-origin convention as the ASL-DVS .mat files.
            "y": y_raw.reshape(-1, 1),
            "ts": np.zeros((1, 1), dtype=np.int32),
            "pol": np.ones((1, 1), dtype=np.uint8),
        }

    event_count = max(1, int(target_events))
    probabilities = weights / total
    flat_indices = rng.choice(weights.size, size=event_count, replace=True, p=probabilities)
    channel, y, x = np.unravel_index(flat_indices, vol.shape)
    y_raw = height - 1 - y

    n_bins = vol.shape[0] // 2
    polarity = (channel < n_bins).astype(np.uint8)
    t_bin = np.where(channel < n_bins, channel, channel - n_bins)
    ts = ((t_bin + rng.random(event_count)) / n_bins * duration_us).astype(np.int32)

    order = np.argsort(ts, kind="stable")
    return {
        "x": x[order].astype(np.int16).reshape(-1, 1),
        "y": y_raw[order].astype(np.int16).reshape(-1, 1),
        "ts": ts[order].astype(np.int32).reshape(-1, 1),
        "pol": polarity[order].astype(np.uint8).reshape(-1, 1),
    }


def generate_eventgan_dataset(
    *,
    image_paths: list[Path],
    image_root: Path,
    output_root: Path,
    eventgan: Any | None,
    sensor_size: tuple[int, int],
    batch_size: int,
    motion_shift: int,
    target_events: int,
    duration_us: int,
    seed: int,
    regenerate: bool,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    to_generate: list[Path] = []

    for image_path in image_paths:
        rel = image_path.relative_to(image_root)
        out_path = output_root / rel.parent / f"{image_path.stem}.mat"
        generated_paths.append(out_path)
        if regenerate or not out_path.exists():
            to_generate.append(image_path)

    if not to_generate:
        print(f"Using existing generated events under {output_root}")
        return generated_paths

    if eventgan is None:
        raise FileNotFoundError(
            "Some EventGAN outputs are missing and no usable EventGAN model was loaded. "
            f"Expected checkpoints under {output_root} or EventGAN weights under the configured checkpoint dir."
        )

    print(f"Generating EventGAN event data for {len(to_generate)} image pairs")
    eventgan.generator.eval()
    with torch.no_grad():
        for batch_paths in chunked(to_generate, batch_size):
            batch = np.stack(
                [
                    image_pair_tensor(
                        path,
                        sensor_size=sensor_size,
                        motion_shift=motion_shift,
                        seed=seed,
                    )
                    for path in batch_paths
                ]
            )
            batch_tensor = torch.from_numpy(batch).to(eventgan.device)
            event_volume = eventgan.forward(batch_tensor, is_train=False)[-1]
            if eventgan.device.type == "mps":
                torch.mps.synchronize()

            for path, volume in zip(batch_paths, event_volume):
                rel = path.relative_to(image_root)
                out_path = output_root / rel.parent / f"{path.stem}.mat"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                arrays = event_volume_to_arrays(
                    volume,
                    target_events=target_events,
                    duration_us=duration_us,
                    rng=np.random.default_rng(stable_seed(str(path), seed)),
                )
                arrays["source_image"] = np.array(str(path))
                savemat(out_path, arrays)

    return generated_paths


def build_generated_dataset(
    sample_paths: list[Path],
    class_to_idx: dict[str, int],
    cfg: Event2VecConfig,
) -> ASLDVSEvent2VecDataset:
    supported_paths = [path for path in sample_paths if path.parent.name in class_to_idx and path.exists()]
    if not supported_paths:
        raise FileNotFoundError("No generated .mat files matched the Event2Vec classes")
    return ASLDVSEvent2VecDataset(
        supported_paths,
        class_to_idx,
        encoding=cfg.data.encoding,
        sensor_size=tuple(cfg.data.sensor_size),
        pool_kernel=tuple(cfg.data.pool_kernel),
        num_steps=cfg.data.num_steps,
        max_tokens=cfg.data.max_tokens,
        delta_threshold=cfg.data.delta_threshold,
    )


def evaluate_generated_events(
    *,
    model: torch.nn.Module,
    dataset: ASLDVSEvent2VecDataset,
    idx_to_class: dict[int, str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_event_sequences,
    )
    rows: list[dict] = []
    confusion = np.zeros((len(idx_to_class), len(idx_to_class)), dtype=np.int64)
    offset = 0

    with torch.no_grad():
        for events, padding_mask, targets, token_counts in loader:
            batch_paths = dataset.sample_paths[offset : offset + int(targets.shape[0])]
            offset += int(targets.shape[0])

            logits = model(events.to(device), padding_mask=padding_mask.to(device))
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            preds = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            for path, target, pred, confidence, token_count in zip(
                batch_paths,
                targets.tolist(),
                preds.tolist(),
                confidences.tolist(),
                token_counts.tolist(),
            ):
                confusion[target, pred] += 1
                rows.append(
                    {
                        "path": str(path),
                        "class": idx_to_class[target],
                        "prediction": idx_to_class[pred],
                        "correct": idx_to_class[target] == idx_to_class[pred],
                        "confidence": float(confidence),
                        "token_count": float(token_count),
                    }
                )

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    per_class = {}
    for class_name in sorted({row["class"] for row in rows}):
        class_rows = [row for row in rows if row["class"] == class_name]
        class_correct = sum(1 for row in class_rows if row["correct"])
        per_class[class_name] = {
            "total": len(class_rows),
            "correct": class_correct,
            "accuracy": class_correct / len(class_rows) if class_rows else 0.0,
        }

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }
    return rows, summary


def save_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "class", "prediction", "correct", "confidence", "token_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = summary["per_class"]
    classes = sorted(per_class)
    accuracies = [per_class[class_name]["accuracy"] for class_name in classes]

    fig_width = max(10, len(classes) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    ax.bar(classes, accuracies, color="#276fbf")
    ax.axhline(summary["accuracy"], color="#d1495b", linestyle="--", linewidth=2, label="overall")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("ASL class")
    ax.set_title("EventGAN ASL image events classified by Event2Vec")
    ax.tick_params(axis="x", rotation=0)
    ax.legend()
    ax.text(
        0.01,
        0.97,
        f"overall: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_raw_events(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat = loadmat(path)
    return (
        mat["x"].reshape(-1).astype(np.int64),
        mat["y"].reshape(-1).astype(np.int64),
        mat["ts"].reshape(-1).astype(np.int64),
        mat["pol"].reshape(-1).astype(np.int8),
    )


def tokens_from_prefix(
    dataset: ASLDVSEvent2VecDataset,
    x: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    pol: np.ndarray,
) -> torch.Tensor:
    if len(x) == 0:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    y_model = dataset.height - 1 - y
    valid = (
        (x >= 0)
        & (x < dataset.width)
        & (y_model >= 0)
        & (y_model < dataset.height)
    )
    if not np.any(valid):
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    raw_frames = dataset._events_to_temporal_channels(x[valid], y_model[valid], ts[valid], pol[valid])
    pooled_raw, pooled_norm = dataset._pool_frames(raw_frames)
    encoded, rho_source = dataset._encode_frames(pooled_raw, pooled_norm)
    tokens = dataset._spikes_to_tokens(encoded, rho_source)
    if tokens.shape[0] == 0:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    return tokens.to(torch.float32)


def predict_tokens(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[str, float]:
    with torch.no_grad():
        events = tokens.unsqueeze(0).to(device)
        padding_mask = torch.zeros((1, tokens.shape[0]), dtype=torch.bool, device=device)
        logits = model(events, padding_mask=padding_mask)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()[0]
    pred_idx = int(probabilities.argmax().item())
    return idx_to_class[pred_idx], float(probabilities[pred_idx].item())


def event_frame(
    x: np.ndarray,
    y: np.ndarray,
    pol: np.ndarray,
    *,
    sensor_size: tuple[int, int],
) -> np.ndarray:
    height, width = sensor_size
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    if len(x) == 0:
        return frame

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    pol = pol[valid]

    pos = pol > 0
    neg = ~pos
    pos_counts = np.zeros((height, width), dtype=np.float32)
    neg_counts = np.zeros((height, width), dtype=np.float32)
    np.add.at(pos_counts, (y[pos], x[pos]), 1.0)
    np.add.at(neg_counts, (y[neg], x[neg]), 1.0)

    if pos_counts.max() > 0:
        pos_counts = np.sqrt(pos_counts / pos_counts.max())
    if neg_counts.max() > 0:
        neg_counts = np.sqrt(neg_counts / neg_counts.max())

    frame[..., 1] = np.maximum(frame[..., 1], (pos_counts * 210).astype(np.uint8))
    frame[..., 2] = np.maximum(frame[..., 2], (pos_counts * 245).astype(np.uint8))
    frame[..., 0] = np.maximum(frame[..., 0], (neg_counts * 245).astype(np.uint8))
    frame[..., 2] = np.maximum(frame[..., 2], (neg_counts * 80).astype(np.uint8))
    return frame


def select_video_rows(rows: list[dict], max_samples: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row["class"] in seen:
            continue
        selected.append(row)
        seen.add(row["class"])
        if len(selected) >= max_samples:
            return selected
    return selected[:max_samples]


def select_showcase_rows(rows: list[dict], max_samples: int) -> list[dict]:
    by_class: dict[str, dict] = {}
    for row in rows:
        best = by_class.get(row["class"])
        row_score = (int(bool(row["correct"])), float(row["confidence"]), float(row["token_count"]))
        if best is None:
            by_class[row["class"]] = row
            continue
        best_score = (int(bool(best["correct"])), float(best["confidence"]), float(best["token_count"]))
        if row_score > best_score:
            by_class[row["class"]] = row

    selected = [by_class[class_name] for class_name in sorted(by_class)]
    return selected[:max_samples]


def render_realtime_video(
    *,
    rows: list[dict],
    dataset: ASLDVSEvent2VecDataset,
    model: torch.nn.Module,
    idx_to_class: dict[int, str],
    device: torch.device,
    output_path: Path,
    sensor_size: tuple[int, int],
    max_samples: int,
    frames_per_sample: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_video_rows(rows, max_samples)
    if not selected:
        return

    video_width = 960
    event_height = 720
    panel_height = 120
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_width, event_height + panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    running_total = 0
    running_correct = 0
    try:
        for sample_index, row in enumerate(selected, start=1):
            path = Path(row["path"])
            x, y, ts, pol = load_raw_events(path)
            t_min = int(ts.min()) if len(ts) else 0
            t_max = int(ts.max()) if len(ts) else 1
            if t_max <= t_min:
                t_max = t_min + 1

            final_prediction = None
            final_confidence = 0.0
            for frame_index, cutoff in enumerate(np.linspace(t_min, t_max, frames_per_sample), start=1):
                mask = ts <= cutoff
                tokens = tokens_from_prefix(dataset, x[mask], y[mask], ts[mask], pol[mask])
                pred, confidence = predict_tokens(model, tokens, idx_to_class, device)
                final_prediction = pred
                final_confidence = confidence

                canvas = event_frame(x[mask], y[mask], pol[mask], sensor_size=sensor_size)
                canvas = cv2.resize(canvas, (video_width, event_height), interpolation=cv2.INTER_NEAREST)
                panel = np.full((panel_height, video_width, 3), 245, dtype=np.uint8)

                elapsed_pct = frame_index / frames_per_sample
                truth = row["class"]
                correct_text = "correct" if pred == truth else "wrong"
                cv2.putText(
                    panel,
                    f"sample {sample_index}/{len(selected)}  true: {truth}  live pred: {pred}  confidence: {confidence:.2f}",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    f"time: {elapsed_pct:5.1%}  status: {correct_text}  running accuracy: "
                    f"{running_correct}/{running_total if running_total else 1}",
                    (24, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    (65, 65, 65),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(np.vstack([panel, canvas]))

            running_total += 1
            if final_prediction == row["class"]:
                running_correct += 1

            hold_panel = np.full((panel_height, video_width, 3), 245, dtype=np.uint8)
            status = "correct" if final_prediction == row["class"] else "wrong"
            cv2.putText(
                hold_panel,
                f"final  true: {row['class']}  pred: {final_prediction}  confidence: {final_confidence:.2f}  {status}",
                (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            hold_frame = np.vstack([hold_panel, canvas])
            for _ in range(max(1, fps // 2)):
                writer.write(hold_frame)
    finally:
        writer.release()


def source_image_for_event_path(path: Path) -> str | None:
    mat = loadmat(path)
    raw = mat.get("source_image")
    if raw is None:
        return None
    value = np.asarray(raw).reshape(-1)
    if value.size == 0:
        return None
    first = value[0]
    if isinstance(first, bytes):
        return first.decode("utf-8", errors="replace")
    return str(first)


def render_eventgan2vec_video(
    *,
    rows: list[dict],
    summary: dict,
    output_path: Path,
    checkpoint_path: Path,
    frames_per_sample: int,
    fps: int,
    max_samples: int,
) -> None:
    from PIL import Image, ImageDraw

    from export_asldvs_demo_videos import (
        ACCENT,
        BG,
        CANVAS_SIZE,
        CARD,
        FONTS,
        GREEN,
        HEADER,
        MUTED,
        RED,
        TEXT,
        draw_multiline,
        draw_progress_bar,
        event_matrix_to_rgb,
        format_time_window,
        make_time_frames,
        render_intro_card,
        write_mp4,
    )

    selected_rows = select_showcase_rows(rows, max_samples=max_samples)
    if not selected_rows:
        return

    checkpoint_label = checkpoint_path.parent.parent.name.replace("_", " ")
    checkpoint_encoding = checkpoint_path.parent.name

    def render_frame(
        row: dict,
        *,
        sample_index: int,
        total_samples: int,
        frame_index: int,
        total_frames: int,
        frame_image: Image.Image,
        frame_start_us: int,
        frame_end_us: int,
    ) -> np.ndarray:
        canvas = Image.new("RGB", CANVAS_SIZE, BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, CANVAS_SIZE[0], 78), fill=HEADER)
        draw.text((44, 18), "EventGAN2Vec", font=FONTS["title"], fill=(255, 255, 255))
        draw.text(
            (48, 92),
            "EventGAN-generated events classified with the Event2Vec 50e latency checkpoint",
            font=FONTS["subtitle"],
            fill=MUTED,
        )

        panel = (48, 128, 854, 675)
        draw.rounded_rectangle(panel, radius=26, fill=CARD)
        frame_resized = frame_image.resize((760, 570), resample=Image.Resampling.NEAREST)
        canvas.paste(frame_resized, (72, 146))

        info = (882, 128, 1236, 675)
        draw.rounded_rectangle(info, radius=26, fill=CARD)
        badge_box = (906, 154, 1210, 224)
        draw.rounded_rectangle(badge_box, radius=20, fill=(255, 241, 228))
        draw.text(
            (930, 173),
            f"{summary['accuracy'] * 100:.2f}% accuracy",
            font=FONTS["badge"],
            fill=ACCENT,
        )

        pred_color = GREEN if row["correct"] else RED
        source_image = source_image_for_event_path(Path(row["path"]))
        source_label = Path(source_image).name if source_image else "source image unavailable"
        lines = [
            ("Ground truth", FONTS["card_title"], MUTED),
            (row["class"].upper(), FONTS["card_title"], TEXT),
            ("Prediction", FONTS["card_title"], MUTED),
            (row["prediction"].upper(), FONTS["letter"], pred_color),
            (f"Confidence: {row['confidence'] * 100:.1f}%", FONTS["body"], TEXT),
            (f"Tokens: {int(round(row['token_count']))}", FONTS["body"], TEXT),
            (f"Checkpoint: {checkpoint_label} / {checkpoint_encoding}", FONTS["small"], TEXT),
            (f"Time bin: {format_time_window(frame_start_us, frame_end_us)}", FONTS["small"], MUTED),
            (f"Clip {sample_index + 1:02d} / {total_samples:02d}", FONTS["small"], MUTED),
            (Path(row["path"]).name, FONTS["tiny"], MUTED),
            (source_label, FONTS["tiny"], MUTED),
            ("Blue = OFF events  |  Red = ON events", FONTS["tiny"], MUTED),
        ]
        draw_multiline(draw, x=910, y=254, lines=lines, gap=7)
        draw_progress_bar(
            draw,
            x=910,
            y=628,
            width=280,
            height=14,
            fraction=(sample_index * total_frames + frame_index + 1) / float(total_samples * total_frames),
            fill=ACCENT,
        )
        return np.asarray(canvas, dtype=np.uint8)

    def iter_frames() -> Iterable[np.ndarray]:
        yield from render_intro_card(
            "EventGAN2Vec",
            f"EventGAN outputs scored {summary['accuracy'] * 100:.2f}% over {summary['total']} generated samples.",
            "Each clip shows a generated event sequence and the final latency-model prediction.",
        )

        for sample_index, row in enumerate(selected_rows):
            mat = loadmat(row["path"])
            events = {key: np.asarray(mat[key]).reshape(-1) for key in ("x", "y", "ts", "pol")}
            frames, edges = make_time_frames(events, n_frames=frames_per_sample)
            clip_limit = float(np.abs(frames).max())
            clip_limit = clip_limit if clip_limit > 0 else 1.0

            for frame_index in range(frames_per_sample):
                event_image = event_matrix_to_rgb(frames[frame_index], limit=clip_limit)
                start_us = int(edges[frame_index])
                end_us = int(edges[frame_index + 1])
                yield render_frame(
                    row,
                    sample_index=sample_index,
                    total_samples=len(selected_rows),
                    frame_index=frame_index,
                    total_frames=frames_per_sample,
                    frame_image=event_image,
                    frame_start_us=start_us,
                    frame_end_us=end_us,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(output_path, iter_frames(), fps=fps)


def write_manifest(image_paths: list[Path], generated_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "image_path", "event_path"])
        writer.writeheader()
        for image_path, event_path in zip(image_paths, generated_paths):
            writer.writerow(
                {
                    "class": image_path.parent.name,
                    "image_path": str(image_path),
                    "event_path": str(event_path),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EventGAN ASL events and classify with Event2Vec")
    parser.add_argument("--image-root", default="data/asl-data")
    parser.add_argument("--generated-root", default="EventGAN/generated_events/asl_data_eventgan")
    parser.add_argument("--output-dir", default="Analytics/eventgan2vec")
    parser.add_argument("--analytics-video-path", default="Analytics/EventGAN2Vec.mp4")
    parser.add_argument("--eventgan-checkpoint-dir", default="EventGAN/logs/EventGAN/checkpoints")
    parser.add_argument("--event2vec-checkpoint", default="runs/event2vec_50_epochs/latency/checkpoint_best.pt")
    parser.add_argument("--event2vec-run-name", default="eventgan_latency_50e")
    parser.add_argument("--force-train-event2vec", action="store_true")
    parser.add_argument("--encoding", choices=["rate", "latency", "delta"], default="latency")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--eventgan-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--sensor-size", type=int, nargs=2, default=[180, 240], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--motion-shift", type=int, default=8)
    parser.add_argument("--target-events", type=int, default=12000)
    parser.add_argument("--duration-us", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--event2vec-epochs", type=int, default=2)
    parser.add_argument("--event2vec-batch-size", type=int, default=16)
    parser.add_argument("--event2vec-max-tokens", type=int, default=512)
    parser.add_argument("--event2vec-max-train-batches", type=int, default=80)
    parser.add_argument("--event2vec-max-val-batches", type=int, default=20)
    parser.add_argument("--event2vec-max-test-batches", type=int, default=20)
    parser.add_argument("--event2vec-workers", type=int, default=0)
    parser.add_argument("--event2vec-log-every", type=int, default=20)
    parser.add_argument("--video-max-samples", type=int, default=24)
    parser.add_argument("--video-frames-per-sample", type=int, default=16)
    parser.add_argument("--video-fps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = resolve_project_path(args.image_root)
    generated_root = resolve_project_path(args.generated_root)
    output_dir = resolve_project_path(args.output_dir)
    analytics_video_path = resolve_project_path(args.analytics_video_path)
    checkpoint_dir = resolve_project_path(args.eventgan_checkpoint_dir)
    sensor_size = tuple(args.sensor_size)

    event2vec_checkpoint = train_event2vec_if_needed(args)
    model, cfg, class_to_idx, device = load_event2vec(event2vec_checkpoint, args.device)
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    image_paths = []
    if image_root.exists():
        image_root = resolve_image_root_for_classes(image_root, class_names)
        image_paths = collect_image_paths(
            image_root,
            class_names,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )

    if image_paths:
        planned_generated_paths = [
            generated_root / image_path.relative_to(image_root).parent / f"{image_path.stem}.mat"
            for image_path in image_paths
        ]
        needs_generation = args.regenerate or any(not path.exists() for path in planned_generated_paths)
        eventgan = load_eventgan(checkpoint_dir) if needs_generation else None
        generated_paths = generate_eventgan_dataset(
            image_paths=image_paths,
            image_root=image_root,
            output_root=generated_root,
            eventgan=eventgan,
            sensor_size=sensor_size,
            batch_size=args.eventgan_batch_size,
            motion_shift=args.motion_shift,
            target_events=args.target_events,
            duration_us=args.duration_us,
            seed=args.seed,
            regenerate=args.regenerate,
        )
        write_manifest(image_paths, generated_paths, output_dir / "generation_manifest.csv")
    else:
        generated_paths = collect_generated_paths(
            generated_root,
            class_names,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        if not generated_paths:
            raise FileNotFoundError(
                "No source images or generated EventGAN events were found. "
                f"Checked image root {image_root} and generated root {generated_root}."
            )

    dataset = build_generated_dataset(generated_paths, class_to_idx, cfg)
    rows, summary = evaluate_generated_events(
        model=model,
        dataset=dataset,
        idx_to_class=idx_to_class,
        device=device,
        batch_size=args.eval_batch_size,
    )

    save_rows_csv(rows, output_dir / "classification_results.csv")
    (output_dir / "classification_summary.json").write_text(json.dumps(summary, indent=2))
    plot_accuracy(summary, output_dir / "classification_accuracy.png")
    render_realtime_video(
        rows=rows,
        dataset=dataset,
        model=model,
        idx_to_class=idx_to_class,
        device=device,
        output_path=output_dir / "eventgan2vec_realtime.mp4",
        sensor_size=sensor_size,
        max_samples=args.video_max_samples,
        frames_per_sample=args.video_frames_per_sample,
        fps=args.video_fps,
    )
    render_eventgan2vec_video(
        rows=rows,
        summary=summary,
        output_path=analytics_video_path,
        checkpoint_path=event2vec_checkpoint,
        frames_per_sample=args.video_frames_per_sample,
        fps=args.video_fps,
        max_samples=args.video_max_samples,
    )

    print(json.dumps({
        "event2vec_checkpoint": str(event2vec_checkpoint),
        "generated_root": str(generated_root),
        "output_dir": str(output_dir),
        "samples": summary["total"],
        "accuracy": summary["accuracy"],
        "accuracy_chart": str(output_dir / "classification_accuracy.png"),
        "realtime_video": str(output_dir / "eventgan2vec_realtime.mp4"),
        "analytics_video": str(analytics_video_path),
    }, indent=2))


if __name__ == "__main__":
    main()
