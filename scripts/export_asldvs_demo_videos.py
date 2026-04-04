#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from scipy.io import loadmat
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event2vec.config import Event2VecConfig  # noqa: E402
from event2vec.data import build_asldvs_event2vec_splits, collate_event_sequences  # noqa: E402
from event2vec.e2v import Event2VecClassifier  # noqa: E402

SENSOR_HEIGHT = 180
SENSOR_WIDTH = 240
CANVAS_SIZE = (1280, 720)
FPS = 10
FRAMES_PER_SAMPLE = 12
INTRO_FRAMES = 12
SEARCH_LIMIT_PER_CLASS = 12
TARGET_CORRECT_CANDIDATES = 3
HEADER_HEIGHT = 78
COOLWARM = colormaps["coolwarm"]

BG = (243, 240, 234)
CARD = (255, 255, 255)
HEADER = (27, 41, 51)
TEXT = (25, 31, 36)
MUTED = (92, 104, 114)
ACCENT = (240, 138, 36)
BLUE = (67, 109, 198)
GREEN = (38, 128, 78)
RED = (180, 65, 70)


@dataclass
class DemoSample:
    class_name: str
    dataset_index: int
    sample_path: str
    predicted_class: str
    confidence: float
    token_count: int
    correct: bool
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ASL-DVS event-data and Event2Vec latency demo MP4s."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/event2vec_50_epochs/latency/checkpoint_best.pt",
        help="Path to the Event2Vec latency checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/demo_videos",
        help="Directory for the generated MP4s and manifest JSON.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use for sample selection, e.g. auto, cpu, cuda.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=FRAMES_PER_SAMPLE,
        help="Number of display frames to render per selected letter sample.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def flip_y_for_display(raw_y: np.ndarray) -> np.ndarray:
    return SENSOR_HEIGHT - 1 - raw_y.astype(np.int64)


def load_asldvs_mat(path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(path)
    return {
        key: mat[key].reshape(-1)
        for key in ("x", "y", "ts", "pol")
    }


def make_time_frames(
    event_dict: dict[str, np.ndarray],
    *,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = event_dict["x"].astype(np.int64)
    y = flip_y_for_display(event_dict["y"])
    ts = event_dict["ts"].astype(np.int64)
    signed_pol = np.where(event_dict["pol"] > 0, 1, -1).astype(np.int8)

    valid = (
        (x >= 0)
        & (x < SENSOR_WIDTH)
        & (y >= 0)
        & (y < SENSOR_HEIGHT)
    )

    x = x[valid]
    y = y[valid]
    ts = ts[valid]
    signed_pol = signed_pol[valid]

    if ts.size == 0:
        frames = np.zeros((n_frames, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
        edges = np.zeros(n_frames + 1, dtype=np.int64)
        return frames, edges

    edges = np.linspace(int(ts.min()), int(ts.max()) + 1, n_frames + 1, dtype=np.int64)
    frames = np.zeros((n_frames, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)

    for frame_idx in range(n_frames):
        mask = (ts >= edges[frame_idx]) & (ts < edges[frame_idx + 1])
        np.add.at(frames[frame_idx], (y[mask], x[mask]), signed_pol[mask])

    return frames, edges


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": load_font(42, bold=True),
    "subtitle": load_font(24),
    "card_title": load_font(28, bold=True),
    "body": load_font(24),
    "small": load_font(20),
    "tiny": load_font(18),
    "letter": load_font(138, bold=True),
    "badge": load_font(30, bold=True),
}


def build_model_from_checkpoint(checkpoint: dict) -> tuple[Event2VecClassifier, Event2VecConfig]:
    cfg = Event2VecConfig.from_dict(checkpoint["config"])
    sensor_h, sensor_w = (int(v) for v in cfg.data.sensor_size)
    pool_h, pool_w = (int(v) for v in cfg.data.pool_kernel)
    pooled_height = sensor_h // pool_h
    pooled_width = sensor_w // pool_w

    model = Event2VecClassifier(
        num_classes=int(cfg.model.num_classes),
        height=pooled_height,
        width=pooled_width,
        d_model=int(cfg.model.d_model),
        depth=int(cfg.model.depth),
        num_heads=int(cfg.model.num_heads),
        ffn_dim=int(cfg.model.ffn_dim),
        dropout=float(cfg.model.dropout),
        pool_after_each_block=tuple(bool(v) for v in cfg.model.pool_after_each_block),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, cfg


@torch.no_grad()
def select_demo_samples(
    model: Event2VecClassifier,
    cfg: Event2VecConfig,
    device: torch.device,
) -> tuple[list[DemoSample], dict]:
    splits = build_asldvs_event2vec_splits(
        data_root=cfg.data.data_root,
        encoding=cfg.data.encoding,
        sensor_size=tuple(int(v) for v in cfg.data.sensor_size),
        pool_kernel=tuple(int(v) for v in cfg.data.pool_kernel),
        num_steps=int(cfg.data.num_steps),
        max_tokens=int(cfg.data.max_tokens),
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
        delta_threshold=float(cfg.data.delta_threshold),
    )
    test_dataset = splits.test
    idx_to_class = {idx: class_name for class_name, idx in test_dataset.class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    model = model.to(device)
    model.eval()

    class_to_indices: dict[str, list[int]] = {class_name: [] for class_name in class_names}
    for dataset_index, sample_path in enumerate(test_dataset.sample_paths):
        class_to_indices[sample_path.parent.name].append(dataset_index)

    selected: list[DemoSample] = []
    searched_samples = 0

    for class_name in class_names:
        indices = class_to_indices[class_name]
        if not indices:
            raise RuntimeError(f"No test samples found for class {class_name!r}")

        candidates: list[DemoSample] = []
        correct_candidates: list[DemoSample] = []

        for examined_idx, dataset_index in enumerate(indices):
            tokens, label, token_count = test_dataset[dataset_index]
            events, padding_mask, targets, token_counts = collate_event_sequences([(tokens, label, token_count)])
            events = events.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            logits = model(events, padding_mask=padding_mask)
            probabilities = logits.softmax(dim=1).cpu()[0]
            pred_idx = int(probabilities.argmax().item())
            confidence = float(probabilities[pred_idx].item())
            target_idx = int(targets[0].item())
            sample = DemoSample(
                class_name=class_name,
                dataset_index=dataset_index,
                sample_path=str(test_dataset.sample_paths[dataset_index]),
                predicted_class=idx_to_class[pred_idx],
                confidence=confidence,
                token_count=int(round(float(token_counts[0].item()))),
                correct=pred_idx == target_idx,
                selection_reason="candidate",
            )
            candidates.append(sample)
            searched_samples += 1
            if sample.correct:
                correct_candidates.append(sample)

            enough_correct = len(correct_candidates) >= TARGET_CORRECT_CANDIDATES
            hit_search_limit = examined_idx + 1 >= SEARCH_LIMIT_PER_CLASS and correct_candidates
            if enough_correct or hit_search_limit:
                break

        if correct_candidates:
            best = max(correct_candidates, key=lambda sample: (sample.confidence, sample.token_count))
            best.selection_reason = "highest-confidence correct test sample from targeted search"
        else:
            best = max(candidates, key=lambda sample: (sample.confidence, sample.token_count))
            best.selection_reason = "fallback highest-confidence test sample from targeted search"
        selected.append(best)
        print(
            f"Selected {class_name.upper()}: {Path(best.sample_path).name} "
            f"-> {best.predicted_class.upper()} ({best.confidence * 100:.1f}%)",
            flush=True,
        )

    summary = {
        "split": "test",
        "searched_samples": searched_samples,
        "selection_mode": (
            "targeted per-class search; keep up to "
            f"{TARGET_CORRECT_CANDIDATES} correct candidates or {SEARCH_LIMIT_PER_CLASS} samples per class"
        ),
    }
    return selected, summary


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    lines: Iterable[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]],
    gap: int = 10,
) -> None:
    cursor_y = y
    for text, font, fill in lines:
        draw.text((x, cursor_y), text, font=font, fill=fill)
        _, height = text_size(draw, text, font)
        cursor_y += height + gap


def draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    fraction: float,
    fill: tuple[int, int, int] = ACCENT,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill=(223, 221, 216))
    clamped = max(0.0, min(1.0, fraction))
    active_width = max(height, int(width * clamped))
    draw.rounded_rectangle((x, y, x + active_width, y + height), radius=height // 2, fill=fill)


def event_matrix_to_rgb(frame: np.ndarray, *, limit: float) -> Image.Image:
    limit = max(limit, 1.0)
    clipped = np.clip(frame, -limit, limit)
    normalized = (clipped + limit) / (2.0 * limit)
    rgba = COOLWARM(normalized)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def format_time_window(start_us: int, end_us: int) -> str:
    return f"{start_us / 1000.0:.1f} - {end_us / 1000.0:.1f} ms"


def render_intro_card(title: str, subtitle: str, footer: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for _ in range(INTRO_FRAMES):
        canvas = Image.new("RGB", CANVAS_SIZE, BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, CANVAS_SIZE[0], HEADER_HEIGHT), fill=HEADER)
        draw.text((48, 20), title, font=FONTS["title"], fill=(255, 255, 255))
        draw.text((48, 150), subtitle, font=FONTS["subtitle"], fill=TEXT)
        draw.rounded_rectangle((48, 250, 1228, 430), radius=28, fill=CARD)
        draw.text((82, 290), footer, font=FONTS["card_title"], fill=ACCENT)
        draw.text(
            (82, 350),
            "Blue = OFF events  |  Red = ON events  |  display y = 179 - raw y",
            font=FONTS["body"],
            fill=MUTED,
        )
        frames.append(np.asarray(canvas, dtype=np.uint8))
    return frames


def render_event_video_frame(
    sample: DemoSample,
    *,
    letter_idx: int,
    total_letters: int,
    frame_idx: int,
    total_frames: int,
    frame_image: Image.Image,
    frame_start_us: int,
    frame_end_us: int,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CANVAS_SIZE[0], HEADER_HEIGHT), fill=HEADER)
    draw.text((44, 18), "ASL-DVS Event Data Showcase", font=FONTS["title"], fill=(255, 255, 255))
    draw.text((48, 92), "Notebook-style display flip applied for export", font=FONTS["subtitle"], fill=MUTED)

    panel = (48, 128, 854, 675)
    draw.rounded_rectangle(panel, radius=26, fill=CARD)
    frame_resized = frame_image.resize((760, 570), resample=Image.Resampling.NEAREST)
    canvas.paste(frame_resized, (72, 146))

    info = (890, 128, 1236, 675)
    draw.rounded_rectangle(info, radius=26, fill=CARD)
    draw.text((920, 158), "Letter", font=FONTS["card_title"], fill=MUTED)
    draw.text((920, 196), sample.class_name.upper(), font=FONTS["letter"], fill=TEXT)

    lines = [
        (f"Clip {letter_idx + 1:02d} / {total_letters:02d}", FONTS["body"], TEXT),
        (f"Frame {frame_idx + 1:02d} / {total_frames:02d}", FONTS["body"], TEXT),
        (format_time_window(frame_start_us, frame_end_us), FONTS["body"], TEXT),
        (Path(sample.sample_path).name, FONTS["small"], MUTED),
        ("Display flip: y -> 179 - raw y", FONTS["small"], MUTED),
        ("Blue = OFF  |  Red = ON", FONTS["small"], MUTED),
    ]
    draw_multiline(draw, x=920, y=378, lines=lines, gap=9)
    draw_progress_bar(
        draw,
        x=920,
        y=628,
        width=280,
        height=14,
        fraction=(letter_idx * total_frames + frame_idx + 1) / float(total_letters * total_frames),
        fill=BLUE,
    )

    return np.asarray(canvas, dtype=np.uint8)


def render_classification_video_frame(
    sample: DemoSample,
    *,
    letter_idx: int,
    total_letters: int,
    frame_idx: int,
    total_frames: int,
    frame_image: Image.Image,
    frame_start_us: int,
    frame_end_us: int,
    reported_accuracy: float,
    best_epoch: int,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CANVAS_SIZE[0], HEADER_HEIGHT), fill=HEADER)
    draw.text((44, 18), "Event2Vec Latency Classification", font=FONTS["title"], fill=(255, 255, 255))
    draw.text((48, 92), "Best latency checkpoint on the ASL-DVS test split", font=FONTS["subtitle"], fill=MUTED)

    panel = (48, 128, 854, 675)
    draw.rounded_rectangle(panel, radius=26, fill=CARD)
    frame_resized = frame_image.resize((760, 570), resample=Image.Resampling.NEAREST)
    canvas.paste(frame_resized, (72, 146))

    info = (882, 128, 1236, 675)
    draw.rounded_rectangle(info, radius=26, fill=CARD)
    badge_box = (906, 154, 1210, 224)
    draw.rounded_rectangle(badge_box, radius=20, fill=(255, 241, 228))
    draw.text((930, 173), f"{reported_accuracy * 100:.2f}% test accuracy", font=FONTS["badge"], fill=ACCENT)

    pred_color = GREEN if sample.correct else RED
    lines = [
        ("Ground truth", FONTS["card_title"], MUTED),
        (sample.class_name.upper(), FONTS["card_title"], TEXT),
        ("Prediction", FONTS["card_title"], MUTED),
        (sample.predicted_class.upper(), FONTS["letter"], pred_color),
        (f"Confidence: {sample.confidence * 100:.1f}%", FONTS["body"], TEXT),
        (f"Tokens: {sample.token_count}", FONTS["body"], TEXT),
        (f"Epoch: {best_epoch}", FONTS["body"], TEXT),
        (f"Time bin: {format_time_window(frame_start_us, frame_end_us)}", FONTS["small"], MUTED),
        (f"Clip {letter_idx + 1:02d} / {total_letters:02d}", FONTS["small"], MUTED),
        (Path(sample.sample_path).name, FONTS["tiny"], MUTED),
        ("Video shows raw event bins; model uses latency-coded pooled input.", FONTS["tiny"], MUTED),
    ]
    draw_multiline(draw, x=910, y=254, lines=lines, gap=7)
    draw_progress_bar(
        draw,
        x=910,
        y=628,
        width=280,
        height=14,
        fraction=(letter_idx * total_frames + frame_idx + 1) / float(total_letters * total_frames),
        fill=ACCENT,
    )

    return np.asarray(canvas, dtype=np.uint8)


def write_mp4(path: Path, frames: Iterable[np.ndarray], *, fps: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    width, height = CANVAS_SIZE
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(f"Frame has shape {frame.shape}, expected {(height, width, 3)}")
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}: {stderr}")
    finally:
        if process.poll() is None:
            process.kill()


def iter_event_video_frames(
    selected_samples: list[DemoSample],
    *,
    frames_per_sample: int,
):
    yield from render_intro_card(
        "ASL-DVS Event Data Showcase",
        "24 letter clips in alphabetical order with the same vertical flip used in the viewer notebook.",
        "Built from raw event frames for a PowerPoint-ready MP4.",
    )

    for letter_idx, sample in enumerate(selected_samples):
        print(f"Rendering event clip {letter_idx + 1:02d}/{len(selected_samples):02d}: {sample.class_name.upper()}", flush=True)
        events = load_asldvs_mat(Path(sample.sample_path))
        frames, edges = make_time_frames(events, n_frames=frames_per_sample)
        clip_limit = float(np.abs(frames).max())
        clip_limit = clip_limit if clip_limit > 0 else 1.0

        for frame_idx in range(frames_per_sample):
            event_image = event_matrix_to_rgb(frames[frame_idx], limit=clip_limit)
            start_us = int(edges[frame_idx])
            end_us = int(edges[frame_idx + 1])
            yield render_event_video_frame(
                sample,
                letter_idx=letter_idx,
                total_letters=len(selected_samples),
                frame_idx=frame_idx,
                total_frames=frames_per_sample,
                frame_image=event_image,
                frame_start_us=start_us,
                frame_end_us=end_us,
            )


def iter_classification_video_frames(
    selected_samples: list[DemoSample],
    *,
    frames_per_sample: int,
    reported_accuracy: float,
    best_epoch: int,
):
    yield from render_intro_card(
        "Event2Vec Latency Classification",
        f"Using the best latency checkpoint from epoch {best_epoch} with {reported_accuracy * 100:.2f}% reported test accuracy.",
        "Each letter clip shows the model prediction selected from the test split.",
    )

    for letter_idx, sample in enumerate(selected_samples):
        print(
            f"Rendering classification clip {letter_idx + 1:02d}/{len(selected_samples):02d}: {sample.class_name.upper()}",
            flush=True,
        )
        events = load_asldvs_mat(Path(sample.sample_path))
        frames, edges = make_time_frames(events, n_frames=frames_per_sample)
        clip_limit = float(np.abs(frames).max())
        clip_limit = clip_limit if clip_limit > 0 else 1.0

        for frame_idx in range(frames_per_sample):
            event_image = event_matrix_to_rgb(frames[frame_idx], limit=clip_limit)
            start_us = int(edges[frame_idx])
            end_us = int(edges[frame_idx + 1])
            yield render_classification_video_frame(
                sample,
                letter_idx=letter_idx,
                total_letters=len(selected_samples),
                frame_idx=frame_idx,
                total_frames=frames_per_sample,
                frame_image=event_image,
                frame_start_us=start_us,
                frame_end_us=end_us,
                reported_accuracy=reported_accuracy,
                best_epoch=best_epoch,
            )


def main() -> None:
    args = parse_args()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, cfg = build_model_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    print(f"Selecting demo samples on device: {device}", flush=True)
    selected_samples, selection_summary = select_demo_samples(model, cfg, device)

    test_summary = checkpoint.get("test_summary", {})
    selection = checkpoint.get("selection", {})
    best_epoch = int(selection.get("best_epoch", test_summary.get("selected_epoch", 0)))
    reported_accuracy = float(test_summary.get("test_acc", 0.0))

    event_video_path = output_dir / "asldvs_all_letters_events.mp4"
    classification_video_path = output_dir / "asldvs_all_letters_event2vec_latency.mp4"
    manifest_path = output_dir / "asldvs_demo_manifest.json"

    print(f"Writing {event_video_path.name}", flush=True)
    write_mp4(
        event_video_path,
        iter_event_video_frames(
            selected_samples,
            frames_per_sample=args.frames_per_sample,
        ),
        fps=args.fps,
    )
    print(f"Writing {classification_video_path.name}", flush=True)
    write_mp4(
        classification_video_path,
        iter_classification_video_frames(
            selected_samples,
            frames_per_sample=args.frames_per_sample,
            reported_accuracy=reported_accuracy,
            best_epoch=best_epoch,
        ),
        fps=args.fps,
    )

    manifest = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "reported_test_summary": test_summary,
        "selection_summary": selection_summary,
        "video_settings": {
            "fps": args.fps,
            "frames_per_sample": args.frames_per_sample,
            "canvas_size": list(CANVAS_SIZE),
            "orientation": "display y = 179 - raw y",
        },
        "outputs": {
            "event_video": str(event_video_path.relative_to(PROJECT_ROOT)),
            "classification_video": str(classification_video_path.relative_to(PROJECT_ROOT)),
        },
        "selected_samples": [asdict(sample) for sample in selected_samples],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote manifest: {manifest_path}", flush=True)
    print("Selected samples:", flush=True)
    for sample in selected_samples:
        print(
            f"  {sample.class_name.upper()}: {Path(sample.sample_path).name} "
            f"-> {sample.predicted_class.upper()} ({sample.confidence * 100:.1f}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
