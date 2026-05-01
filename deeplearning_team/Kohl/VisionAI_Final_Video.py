"""
VisionAI Final Video — ASL Sign Language Video Recognition
============================================================
A production-quality video classification pipeline for recognizing
American Sign Language signs from short video clips.

Supported architectures (set MODEL_TYPE):
    "mvit_v2_s"  — MViTv2-S (Multiscale Vision Transformer v2, Small)
        80.76% top-1 on Kinetics-400.  Transformer-based architecture
        with multiscale pooling attention.  Requires 224x224 input.
        Best available video model in torchvision.

    "r2plus1d_18" — R(2+1)D-18
        67.46% top-1 on Kinetics-400.  Decomposes 3D convolutions into
        separate spatial (2D) and temporal (1D) convolutions.
        Requires 112x112 input.  Faster, lower memory.

Data pipeline:
    Each video is decoded into frames, then we sample a fixed-length
    clip (CLIP_LEN frames spaced FRAME_STRIDE apart) with temporal
    jitter during training.  At test time we sample MULTI_CLIP_K
    evenly-spaced clips per video and average their predictions
    (multi-clip evaluation) for more robust results.

Synthetic data mode:
    Because real ASL video datasets are large and require special
    download steps, this script ships with a *synthetic data generator*
    that creates short videos of colored shapes performing distinct
    motion patterns (one pattern per "sign class").  Set
    GENERATE_SYNTHETIC = True to run the full pipeline end-to-end
    without any external data.  Then swap in a real dataset when ready.

Requirements:
    pip install torch torchvision numpy matplotlib Pillow

    For REAL video data you also need ONE of:
        pip install av            # PyAV — fastest
        pip install opencv-python # OpenCV fallback
"""

import time
import json
import csv
import random
import sys
import os
import struct
import datetime
import shutil
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.models.video as video_models
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
#  CONFIGURATION
# ============================================================================

# --- Paths ---
VIDEO_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\asl_videos"
OUTPUT_DIR = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "visionai_video_v4"  # v4: upgrade to MViTv2-S backbone

# --- Model selection ---
# "mvit_v2_s"  — MViTv2-S: 80.76% Kinetics-400, 224x224, transformer-based (recommended)
# "r2plus1d_18" — R(2+1)D-18: 67.46% Kinetics-400, 112x112, 3D-CNN (lighter, fallback)
MODEL_TYPE = "mvit_v2_s"

# --- Synthetic data (set True to auto-generate toy data for pipeline test) ---
GENERATE_SYNTHETIC = False
SYNTHETIC_NUM_CLASSES = 12       # number of "sign" classes to generate
SYNTHETIC_VIDEOS_PER_CLASS = 60  # videos per class (train+val+test)
SYNTHETIC_FRAME_H = 224 if MODEL_TYPE == "mvit_v2_s" else 112
SYNTHETIC_FRAME_W = 224 if MODEL_TYPE == "mvit_v2_s" else 112
SYNTHETIC_NUM_FRAMES = 32        # frames per synthetic video

# --- Video loading ---
CLIP_LEN = 16                    # frames per clip fed to the model
FRAME_STRIDE = 2                 # sample every Nth frame (temporal receptive field = CLIP_LEN * FRAME_STRIDE)
FRAME_SIZE = 224 if MODEL_TYPE == "mvit_v2_s" else 112  # MViTv2 needs 224; R(2+1)D needs 112
DATA_SUBSET_FRACTION = 1.0       # 0.0-1.0; set < 1.0 for quick test runs
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15                 # test = 1 - TRAIN_RATIO - VAL_RATIO
SEED = 42
NUM_WORKERS = 0                  # 0 for Windows compatibility; increase on Linux

# --- Training ---
BATCH_SIZE = 2 if MODEL_TYPE == "mvit_v2_s" else 4  # MViTv2 at 224x224 is memory-heavy
NUM_EPOCHS = 35
LEARNING_RATE = 5e-5 if MODEL_TYPE == "mvit_v2_s" else 1e-4  # lower for larger pretrained model
HEAD_LR_MULTIPLIER = 10          # classification head gets LR * this multiplier
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = "cosine"        # "cosine" or "step"
STEP_LR_STEP_SIZE = 8
STEP_LR_GAMMA = 0.1

# --- Model ---
FREEZE_BACKBONE_STAGES = 10 if MODEL_TYPE == "mvit_v2_s" else 2  # MViTv2: freeze first 10/16 blocks; R(2+1)D: stem+layer1..2
DROPOUT_RATE = 0.4
HIDDEN_DIM = 256

# --- Multi-clip evaluation ---
MULTI_CLIP_K = 5                 # clips per video at test time

# --- Early stopping ---
EARLY_STOPPING_PATIENCE = 8

# --- Augmentation (spatial) ---
# IMPORTANT: horizontal flips are DISABLED for sign language — mirroring changes
# the meaning of signs (e.g. left-hand vs right-hand signs are different).
AUGMENT_HFLIP = False
AUGMENT_COLOR_JITTER = 0.15

# --- Normalization (model-specific) ---
if MODEL_TYPE == "mvit_v2_s":
    NORMALIZE_MEAN = [0.45, 0.45, 0.45]
    NORMALIZE_STD = [0.225, 0.225, 0.225]
else:
    # R(2+1)D Kinetics-400 normalization
    NORMALIZE_MEAN = [0.43216, 0.394666, 0.37645]
    NORMALIZE_STD = [0.22803, 0.22145, 0.216989]

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
#  LOGGER — dual output to console + file
# ============================================================================

class DualLogger:
    """Writes every message to both stdout and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.stdout.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()

    def log(self, msg: str = ""):
        line = f"{msg}\n"
        self.write(line)

    def raw(self, msg: str = ""):
        """Write without extra newline (caller controls formatting)."""
        self.write(msg + "\n")

    def close(self):
        self.log_file.close()


# ============================================================================
#  SYNTHETIC DATA GENERATOR
# ============================================================================

# We create short videos of coloured shapes performing class-specific motion
# patterns.  This lets the model learn temporal dynamics (not just appearance)
# and allows the full pipeline to run without downloading external data.

SIGN_NAMES = [
    "hello", "thanks", "yes", "no", "please",
    "sorry", "help", "love", "friend", "learn",
    "eat", "drink", "sleep", "play", "work",
    "good", "bad", "more", "stop", "go",
]

SHAPE_COLORS = [
    (220, 60, 60),   (60, 180, 60),   (60, 80, 220),  (220, 180, 40),
    (180, 60, 200),  (40, 200, 200),  (240, 140, 40),  (120, 200, 80),
    (200, 80, 140),  (80, 140, 220),  (180, 200, 60),  (100, 60, 160),
    (220, 120, 120), (60, 160, 140),  (160, 120, 80),  (80, 200, 160),
    (200, 160, 200), (140, 100, 60),  (60, 120, 100),  (200, 200, 200),
]


def _motion_trajectory(cls_id, num_frames, w, h, radius=18):
    """Return list of (cx, cy) positions for each frame — one pattern per class."""
    cx0, cy0 = w // 2, h // 2
    amp_x, amp_y = w // 3, h // 3
    patterns = [
        # 0: circle CW
        lambda t: (cx0 + int(amp_x * np.cos(2 * np.pi * t / num_frames)),
                   cy0 + int(amp_y * np.sin(2 * np.pi * t / num_frames))),
        # 1: circle CCW
        lambda t: (cx0 + int(amp_x * np.cos(-2 * np.pi * t / num_frames)),
                   cy0 + int(amp_y * np.sin(-2 * np.pi * t / num_frames))),
        # 2: horizontal bounce
        lambda t: (cx0 + int(amp_x * np.sin(2 * np.pi * t / num_frames)), cy0),
        # 3: vertical bounce
        lambda t: (cx0, cy0 + int(amp_y * np.sin(2 * np.pi * t / num_frames))),
        # 4: diagonal TL-BR
        lambda t: (int(radius + (w - 2 * radius) * t / num_frames),
                   int(radius + (h - 2 * radius) * t / num_frames)),
        # 5: diagonal TR-BL
        lambda t: (int(w - radius - (w - 2 * radius) * t / num_frames),
                   int(radius + (h - 2 * radius) * t / num_frames)),
        # 6: figure-8
        lambda t: (cx0 + int(amp_x * np.sin(2 * np.pi * t / num_frames)),
                   cy0 + int(amp_y * np.sin(4 * np.pi * t / num_frames))),
        # 7: shrink-grow (stay centred, vary radius — handled specially)
        lambda t: (cx0, cy0),
        # 8: zigzag horizontal
        lambda t: (int(radius + (w - 2 * radius) * t / num_frames),
                   cy0 + int(amp_y * 0.6 * ((-1) ** int(4 * t / num_frames)))),
        # 9: spiral outward
        lambda t: (cx0 + int(amp_x * (t / num_frames) * np.cos(3 * 2 * np.pi * t / num_frames)),
                   cy0 + int(amp_y * (t / num_frames) * np.sin(3 * 2 * np.pi * t / num_frames))),
        # 10: stationary + blink (alpha modulation)
        lambda t: (cx0, cy0),
        # 11: L-shape path
        lambda t: (int(radius + (w - 2 * radius) * min(1, 2 * t / num_frames)),
                   int(radius + (h - 2 * radius) * max(0, 2 * t / num_frames - 1))),
    ]
    idx = cls_id % len(patterns)
    coords = [patterns[idx](t) for t in range(num_frames)]
    return coords, idx


def _draw_frame(cls_id, pattern_idx, frame_t, num_frames, cx, cy, w, h, base_radius=18):
    """Draw a single frame with shape + motion pattern."""
    img = Image.new("RGB", (w, h), (20, 20, 30))  # dark background
    draw = ImageDraw.Draw(img)

    color = SHAPE_COLORS[cls_id % len(SHAPE_COLORS)]
    r = base_radius

    # Pattern 7 = shrink-grow
    if pattern_idx == 7:
        r = int(base_radius * (0.5 + abs(np.sin(2 * np.pi * frame_t / num_frames))))
    # Pattern 10 = blink
    if pattern_idx == 10 and (frame_t // (num_frames // 6)) % 2 == 1:
        return img  # blank frame for blink effect

    # Alternate between circle and rectangle based on class
    if cls_id % 3 == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif cls_id % 3 == 1:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        # triangle
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(pts, fill=color)

    # Small trailing dot for motion cue
    if frame_t > 0:
        alpha_color = tuple(c // 3 for c in color)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=alpha_color)

    return img


def generate_synthetic_dataset(root: Path, num_classes: int, vids_per_class: int,
                               num_frames: int, h: int, w: int, logger):
    """
    Create a folder-of-folders dataset: root/<class_name>/vid_XXXX/frame_XXXX.png
    Each "video" is a folder of PNG frames (simple, no codec dependency).
    """
    logger.log(f"\n  Generating synthetic video dataset at: {root}")
    logger.log(f"  Classes: {num_classes}, Videos/class: {vids_per_class}, "
               f"Frames/video: {num_frames}, Size: {h}x{w}")

    class_names = SIGN_NAMES[:num_classes]
    rng = random.Random(SEED)

    total_vids = 0
    for cls_id, cls_name in enumerate(class_names):
        cls_dir = root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        coords, pattern_idx = _motion_trajectory(cls_id, num_frames, w, h)

        for vid_i in range(vids_per_class):
            vid_dir = cls_dir / f"vid_{vid_i:04d}"
            vid_dir.mkdir(exist_ok=True)

            # Add per-video randomness: offset + noise
            x_off = rng.randint(-8, 8)
            y_off = rng.randint(-8, 8)
            t_off = rng.randint(0, num_frames - 1)

            for t in range(num_frames):
                t_shifted = (t + t_off) % num_frames
                cx, cy = coords[t_shifted]
                cx = max(4, min(w - 4, cx + x_off + rng.randint(-3, 3)))
                cy = max(4, min(h - 4, cy + y_off + rng.randint(-3, 3)))
                frame_img = _draw_frame(cls_id, pattern_idx, t_shifted, num_frames,
                                        cx, cy, w, h)
                frame_img.save(vid_dir / f"frame_{t:04d}.png")
            total_vids += 1

    logger.log(f"  Created {total_vids} synthetic videos "
               f"({total_vids * num_frames} frames total)\n")
    return class_names


# ============================================================================
#  DATASET — loads video clips from frame-folder or video-file structure
# ============================================================================

class ASLVideoDataset(Dataset):
    """
    Expects either:
      A) root/<class>/<vid_id>/frame_XXXX.png  (frame-folder format)
      B) root/<class>/<vid_id>.{mp4,avi,mov}   (video-file format, needs cv2/av)

    Each sample returns (clip_tensor, label) where clip_tensor has shape
    (C, T, H, W) — channels-first, time second (torchvision video convention).
    """

    def __init__(self, root: str, clip_len: int, frame_stride: int,
                 frame_size: int, spatial_transform=None, is_train: bool = True):
        super().__init__()
        self.root = Path(root)
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.spatial_transform = spatial_transform
        self.is_train = is_train

        # Discover classes and videos
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []   # list of (video_path_or_dir, label)
        self.targets = []   # flat label list for stratified split

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            label = self.class_to_idx[cls_name]

            for item in sorted(cls_dir.iterdir()):
                if item.is_dir():
                    # frame-folder format
                    frames = sorted(item.glob("*.png")) + sorted(item.glob("*.jpg"))
                    if len(frames) >= clip_len:
                        self.samples.append((item, label, "frames"))
                        self.targets.append(label)
                elif item.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    self.samples.append((item, label, "video"))
                    self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def _load_frames_from_folder(self, folder: Path):
        """Load all PNG/JPG frames from a folder, return list of PIL Images."""
        paths = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
        return [Image.open(p).convert("RGB") for p in paths]

    def _load_frames_from_video(self, video_path: Path):
        """Load frames from a video file using available backend."""
        frames = []
        try:
            import av
            # Use bytes mode to avoid UTF-8 path encoding issues on Windows
            with open(video_path, "rb") as f:
                container = av.open(f)
                for frame in container.decode(video=0):
                    frames.append(frame.to_image().convert("RGB"))
                container.close()
        except ImportError:
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                while cap.isOpened():
                    ret, bgr = cap.read()
                    if not ret:
                        break
                    rgb = bgr[:, :, ::-1]
                    frames.append(Image.fromarray(rgb))
                cap.release()
            except ImportError:
                raise RuntimeError(
                    "Video file loading requires either 'av' (PyAV) or "
                    "'cv2' (opencv-python). Install one:\n"
                    "  pip install av\n  pip install opencv-python"
                )
        return frames

    def _sample_clip(self, frames, multi_clip_idx=None):
        """
        Sample CLIP_LEN frames spaced FRAME_STRIDE apart.
        Training: random temporal offset (jitter).
        Eval: centred clip (or multi-clip with even spacing).
        """
        total = len(frames)
        span = self.clip_len * self.frame_stride  # total frames the clip covers

        if span > total:
            # Not enough frames — sample with reduced stride or repeat
            indices = np.linspace(0, total - 1, self.clip_len, dtype=int)
        elif self.is_train:
            # Random start within valid range
            max_start = total - span
            start = random.randint(0, max_start)
            indices = list(range(start, start + span, self.frame_stride))
        else:
            if multi_clip_idx is not None and MULTI_CLIP_K > 1:
                max_start = total - span
                starts = np.linspace(0, max_start, MULTI_CLIP_K, dtype=int)
                start = int(starts[multi_clip_idx])
            else:
                start = (total - span) // 2  # centre crop
            indices = list(range(start, start + span, self.frame_stride))

        return [frames[i] for i in indices[:self.clip_len]]

    def _frames_to_tensor(self, frame_list):
        """
        Convert list of PIL images to tensor (C, T, H, W).
        Applies spatial transform per frame, then stacks along time.
        """
        tensors = []
        for img in frame_list:
            img = img.resize((self.frame_size, self.frame_size), Image.BILINEAR)
            if self.spatial_transform:
                img = self.spatial_transform(img)
            else:
                img = transforms.ToTensor()(img)
            tensors.append(img)
        # Stack along new time dimension: (T, C, H, W) -> (C, T, H, W)
        clip = torch.stack(tensors, dim=1)
        return clip

    def __getitem__(self, idx):
        # Try loading the requested video; on failure, fall back to a random valid sample
        for attempt_idx in [idx] + [random.randint(0, len(self) - 1) for _ in range(5)]:
            try:
                path, label, fmt = self.samples[attempt_idx]
                if fmt == "frames":
                    all_frames = self._load_frames_from_folder(path)
                else:
                    all_frames = self._load_frames_from_video(path)
                if len(all_frames) < 2:
                    continue  # too few frames, skip
                clip_frames = self._sample_clip(all_frames)
                clip_tensor = self._frames_to_tensor(clip_frames)
                return clip_tensor, label
            except Exception:
                continue
        # Last resort: return a black clip with label 0
        clip = torch.zeros(3, self.clip_len, self.frame_size, self.frame_size)
        return clip, 0

    def get_multi_clip(self, idx, k=None):
        """Return K clips for multi-clip evaluation. Returns (K, C, T, H, W)."""
        k = k or MULTI_CLIP_K
        path, label, fmt = self.samples[idx]
        try:
            if fmt == "frames":
                all_frames = self._load_frames_from_folder(path)
            else:
                all_frames = self._load_frames_from_video(path)
            if len(all_frames) < 2:
                raise ValueError("Too few frames")
        except Exception:
            # Return black clips on failure
            clip = torch.zeros(k, 3, self.clip_len, self.frame_size, self.frame_size)
            return clip, label

        clips = []
        for ci in range(k):
            clip_frames = self._sample_clip(all_frames, multi_clip_idx=ci)
            clip_tensor = self._frames_to_tensor(clip_frames)
            clips.append(clip_tensor)
        return torch.stack(clips), label  # (K, C, T, H, W)


# ============================================================================
#  MODEL — MViTv2-S / R(2+1)D-18 with custom classification head
# ============================================================================

class ASLVideoNet(nn.Module):
    """
    Video classification model with two architecture options:

    MViTv2-S (MODEL_TYPE="mvit_v2_s"):
        Multiscale Vision Transformer v2, Small variant.
        80.76% top-1 on Kinetics-400.  Uses multiscale pooling attention
        with hierarchical feature maps.  Feature dim = 768.
        Input: (B, C=3, T=16, H=224, W=224).

    R(2+1)D-18 (MODEL_TYPE="r2plus1d_18"):
        Factorised 3D CNN (spatial 2D + temporal 1D convolutions).
        67.46% top-1 on Kinetics-400.  Feature dim = 512.
        Input: (B, C=3, T=16, H=112, W=112).

    Both use:
      - Kinetics-400 pretrained weights
      - Optional backbone stage freezing (transfer learning)
      - Custom classification head with dropout
    """

    def __init__(self, num_classes: int, model_type: str = "mvit_v2_s",
                 dropout_rate: float = 0.4, hidden_dim: int = 256,
                 freeze_stages: int = 10):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.freeze_stages = freeze_stages

        if model_type == "mvit_v2_s":
            self._build_mvit(num_classes, dropout_rate, hidden_dim, freeze_stages)
        elif model_type == "r2plus1d_18":
            self._build_r2plus1d(num_classes, dropout_rate, hidden_dim, freeze_stages)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}. "
                             f"Choose 'mvit_v2_s' or 'r2plus1d_18'.")

    # ---- MViTv2-S builder ----

    def _build_mvit(self, num_classes, dropout_rate, hidden_dim, freeze_stages):
        """Build MViTv2-S backbone with custom classification head."""
        weights = video_models.MViT_V2_S_Weights.KINETICS400_V1
        self.backbone = video_models.mvit_v2_s(weights=weights)

        # MViTv2-S structure:
        #   .conv_proj     — convolutional patch projection (stem)
        #   .pos_encoding  — positional encoding
        #   .blocks        — nn.ModuleList of 16 transformer blocks
        #   .norm          — final LayerNorm
        #   .head          — Sequential(Dropout, Linear(768, 400))
        #
        # Replace .head with our custom head (feature dim = 768)
        self.backbone.head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Freeze patch embedding, positional encoding, and early transformer blocks
        self._freeze_mvit_stages(freeze_stages)

    def _freeze_mvit_stages(self, n_stages):
        """Freeze patch embedding and first n_stages transformer blocks."""
        if n_stages <= 0:
            return

        # Always freeze convolutional projection and positional encoding
        for param in self.backbone.conv_proj.parameters():
            param.requires_grad = False
        for param in self.backbone.pos_encoding.parameters():
            param.requires_grad = False

        # Freeze first n_stages transformer blocks
        num_blocks = len(self.backbone.blocks)
        blocks_to_freeze = min(n_stages, num_blocks)
        for i in range(blocks_to_freeze):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = False

    # ---- R(2+1)D-18 builder ----

    def _build_r2plus1d(self, num_classes, dropout_rate, hidden_dim, freeze_stages):
        """Build R(2+1)D-18 backbone with custom classification head."""
        weights = video_models.R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = video_models.r2plus1d_18(weights=weights)

        # The backbone has: stem, layer1, layer2, layer3, layer4, avgpool, fc
        # We replace fc with our own head
        backbone_out_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Freeze early stages
        self._freeze_r2plus1d_stages(freeze_stages)

    def _freeze_r2plus1d_stages(self, n_stages):
        """Freeze the stem + first n_stages residual layers."""
        if n_stages <= 0:
            return

        # Always freeze stem if n_stages >= 1
        for param in self.backbone.stem.parameters():
            param.requires_grad = False

        stage_names = ["layer1", "layer2", "layer3", "layer4"]
        for i in range(min(n_stages, len(stage_names))):
            layer = getattr(self.backbone, stage_names[i])
            for param in layer.parameters():
                param.requires_grad = False

    # ---- Forward pass ----

    def forward(self, x):
        """
        x: (B, C, T, H, W) — batch of video clips
        Returns: (B, num_classes) logits
        """
        return self.backbone(x)


# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(dataset, train_ratio, val_ratio, seed):
    """Split dataset into train/val/test using stratified sampling."""
    rng = random.Random(seed)
    targets = dataset.targets
    class_indices = {}
    for idx, label in enumerate(targets):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls][:]
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def compute_gradient_norm(model):
    """Compute total L2 gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
#  TRAINING AND EVALUATION LOOPS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, logger, epoch):
    """Train for one epoch, return (avg_loss, accuracy, grad_norm)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_norm = 0.0
    num_batches = 0

    for batch_idx, (clips, labels) in enumerate(loader):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()

        grad_norm += compute_gradient_norm(model)
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            logger.raw(f"    Batch {batch_idx + 1}/{len(loader)}  "
                       f"loss={loss.item():.4f}  "
                       f"acc={100 * correct / total:.1f}%")

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    avg_grad = grad_norm / max(num_batches, 1)
    return avg_loss, accuracy, avg_grad


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataloader, return (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips)
        loss = criterion(logits, labels)

        running_loss += loss.item() * clips.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def multi_clip_evaluate(model, dataset, indices, device, k=None, logger=None):
    """
    Multi-clip evaluation: sample K clips per video, average logits, take argmax.
    This is the standard evaluation protocol for video classification and
    typically improves accuracy by 2-5% over single-clip.
    """
    k = k or MULTI_CLIP_K
    model.eval()
    all_preds = []
    all_labels = []

    for count, idx in enumerate(indices):
        clips, label = dataset.get_multi_clip(idx, k=k)
        # clips: (K, C, T, H, W)
        clips = clips.to(device)
        logits = model(clips)  # (K, num_classes)
        avg_logits = logits.mean(dim=0)  # (num_classes,)
        pred = avg_logits.argmax().item()

        all_preds.append(pred)
        all_labels.append(label)

        if logger and (count + 1) % max(1, len(indices) // 5) == 0:
            logger.raw(f"    Multi-clip eval: {count + 1}/{len(indices)}")

    return np.array(all_preds), np.array(all_labels)


# ============================================================================
#  VISUALIZATION
# ============================================================================

def plot_training_curves(history, output_dir, prefix):
    """Plot loss and accuracy curves for train/val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs_range, history["train_loss"], "b-o", markersize=3, label="Train")
    axes[0].plot(epochs_range, history["val_loss"], "r-o", markersize=3, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "b-o", markersize=3, label="Train")
    axes[1].plot(epochs_range, history["val_acc"], "r-o", markersize=3, label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"{prefix}_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, class_names, output_dir, prefix, title="Confusion Matrix"):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.7),
                                    max(6, len(class_names) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True label",
           xlabel="Predicted label",
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Add counts in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = output_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gradient_norms(history, output_dir, prefix):
    """Plot gradient norm over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs_range = range(1, len(history["grad_norm"]) + 1)
    ax.plot(epochs_range, history["grad_norm"], "g-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Gradient Norm")
    ax.set_title("Gradient Norm per Epoch")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{prefix}_gradient_norms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sample_clips(dataset, indices, class_names, preds, labels,
                      output_dir, prefix, n_samples=8):
    """
    Visualise sample video clips as horizontal frame strips.
    Shows 4 evenly-spaced frames from each clip with predicted/true labels.
    """
    n_samples = min(n_samples, len(indices), len(preds), len(labels))
    if n_samples < 1:
        return output_dir / f"{prefix}_sample_clips.png"  # nothing to plot
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, n_samples * 1.8))
    if n_samples == 1:
        axes = [axes]

    row = 0
    for sample_i in range(min(n_samples + 10, len(indices))):
        if row >= n_samples:
            break
        idx = indices[sample_i]
        path, label, fmt = dataset.samples[idx]

        try:
            if fmt == "frames":
                all_frames = dataset._load_frames_from_folder(path)
            else:
                all_frames = dataset._load_frames_from_video(path)
            if len(all_frames) < 2:
                continue
        except Exception:
            continue  # skip corrupt videos

        # Pick 4 evenly-spaced frames
        frame_indices = np.linspace(0, len(all_frames) - 1, 4, dtype=int)
        for col, fi in enumerate(frame_indices):
            ax = axes[row][col] if n_samples > 1 else axes[col]
            ax.imshow(all_frames[fi].resize((FRAME_SIZE, FRAME_SIZE)))
            ax.axis("off")
            if col == 0:
                pred_name = class_names[preds[sample_i]]
                true_name = class_names[labels[sample_i]]
                colour = "green" if preds[sample_i] == labels[sample_i] else "red"
                ax.set_title(f"P:{pred_name}\nT:{true_name}",
                             fontsize=7, color=colour, fontweight="bold")
        row += 1

    plt.suptitle("Sample Video Clips (4 frames each)", fontsize=11, y=1.02)
    plt.tight_layout()
    path = output_dir / f"{prefix}_sample_clips.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_class_accuracy(cm, class_names, output_dir, prefix):
    """Horizontal bar chart of per-class accuracy."""
    per_class_acc = []
    for i in range(len(class_names)):
        total_i = cm[i].sum()
        acc_i = cm[i, i] / total_i if total_i > 0 else 0
        per_class_acc.append(acc_i)

    fig, ax = plt.subplots(figsize=(8, max(4, len(class_names) * 0.35)))
    colors = ["#2ecc71" if a >= 0.8 else "#e74c3c" if a < 0.5 else "#f39c12"
              for a in per_class_acc]
    y_pos = range(len(class_names))
    ax.barh(y_pos, per_class_acc, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    path = output_dir / f"{prefix}_per_class_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_temporal_sensitivity(model, dataset, indices, device, class_names,
                              output_dir, prefix, n_samples=6):
    """
    Temporal occlusion sensitivity: for each frame position in a clip, zero it
    out and measure the drop in correct-class logit.  A large drop means that
    frame position is critical for the prediction.

    This reveals which moments in the sign the model attends to most.
    """
    model.eval()
    n_samples = min(n_samples, len(indices))
    if n_samples < 1:
        return output_dir / f"{prefix}_temporal_sensitivity.png"  # nothing to plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 2.2))
    if n_samples == 1:
        axes = [axes]

    row = 0
    for sample_i in range(min(n_samples + 10, len(indices))):
        if row >= n_samples:
            break
        idx = indices[sample_i]
        try:
            clip, label = dataset[idx]
        except Exception:
            continue
        clip = clip.unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.no_grad():
            base_logits = model(clip)
            base_score = base_logits[0, label].item()

        drops = []
        T = clip.shape[2]
        for t in range(T):
            occluded = clip.clone()
            occluded[:, :, t, :, :] = 0  # zero out frame t
            with torch.no_grad():
                occ_logits = model(occluded)
                occ_score = occ_logits[0, label].item()
            drops.append(base_score - occ_score)

        ax = axes[row]
        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in drops]
        ax.bar(range(T), drops, color=colors, alpha=0.8)
        ax.set_ylabel("Logit drop", fontsize=7)
        ax.set_title(f"Sign: {class_names[label]} — Temporal Sensitivity",
                     fontsize=8)
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"f{t}" for t in range(T)], fontsize=6)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        row += 1

    axes[-1].set_xlabel("Frame position in clip")
    plt.suptitle("Temporal Occlusion Sensitivity\n(red = removing frame hurts prediction)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    path = output_dir / f"{prefix}_temporal_sensitivity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
#  MAIN
# ============================================================================

def main():
    start_time = time.time()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = DualLogger(output_dir / f"{OUTPUT_PREFIX}_log.txt")

    model_display = {
        "mvit_v2_s": "MViTv2-S (Kinetics-400 pretrained, 80.76% top-1)",
        "r2plus1d_18": "R(2+1)D-18 (Kinetics-400 pretrained, 67.46% top-1)",
    }

    logger.log("=" * 70)
    logger.log("  VisionAI Final — ASL Video Sign Recognition")
    logger.log(f"  Model: {model_display.get(MODEL_TYPE, MODEL_TYPE)}")
    logger.log("=" * 70)
    logger.log(f"  Date/time   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.log(f"  Device      : {DEVICE}")
    if torch.cuda.is_available():
        logger.log(f"  GPU         : {torch.cuda.get_device_name(0)}")
        logger.log(f"  GPU memory  : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    logger.log(f"  Model type  : {MODEL_TYPE}")
    logger.log(f"  Clip length : {CLIP_LEN} frames x stride {FRAME_STRIDE} "
               f"= {CLIP_LEN * FRAME_STRIDE} frame span")
    logger.log(f"  Frame size  : {FRAME_SIZE}x{FRAME_SIZE}")
    logger.log(f"  Batch size  : {BATCH_SIZE}")
    logger.log(f"  Epochs      : {NUM_EPOCHS}")
    logger.log(f"  LR (backbone): {LEARNING_RATE}")
    logger.log(f"  LR (head)   : {LEARNING_RATE * HEAD_LR_MULTIPLIER}")
    logger.log(f"  Head LR mult: {HEAD_LR_MULTIPLIER}x")
    logger.log(f"  Freeze stages: {FREEZE_BACKBONE_STAGES}")
    logger.log(f"  HFlip aug   : {AUGMENT_HFLIP}  (disabled for sign language)")
    logger.log(f"  Normalize   : mean={NORMALIZE_MEAN}, std={NORMALIZE_STD}")
    logger.log(f"  Multi-clip K : {MULTI_CLIP_K}")
    logger.log(f"  Seed        : {SEED}")
    logger.log("")

    # ---- Seed ----
    set_seed(SEED)

    # ---- Synthetic data generation ----
    video_root = Path(VIDEO_ROOT)
    if GENERATE_SYNTHETIC:
        logger.log("=" * 70)
        logger.log("  SYNTHETIC DATA GENERATION")
        logger.log("=" * 70)
        class_names = generate_synthetic_dataset(
            video_root, SYNTHETIC_NUM_CLASSES, SYNTHETIC_VIDEOS_PER_CLASS,
            SYNTHETIC_NUM_FRAMES, SYNTHETIC_FRAME_H, SYNTHETIC_FRAME_W, logger
        )
    else:
        class_names = None  # will be read from dataset

    # ---- Transforms ----
    # Spatial transforms (applied per-frame)
    train_transforms_list = []
    if AUGMENT_HFLIP:
        train_transforms_list.append(transforms.RandomHorizontalFlip())
    train_transforms_list.extend([
        transforms.ColorJitter(brightness=AUGMENT_COLOR_JITTER,
                               contrast=AUGMENT_COLOR_JITTER,
                               saturation=AUGMENT_COLOR_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    train_spatial = transforms.Compose(train_transforms_list)

    eval_spatial = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # ---- Dataset ----
    logger.log("=" * 70)
    logger.log("  LOADING DATASET")
    logger.log("=" * 70)

    full_dataset = ASLVideoDataset(
        root=str(video_root),
        clip_len=CLIP_LEN,
        frame_stride=FRAME_STRIDE,
        frame_size=FRAME_SIZE,
        spatial_transform=train_spatial,
        is_train=True,
    )

    if class_names is None:
        class_names = full_dataset.classes
    num_classes = len(class_names)

    logger.log(f"  Total videos  : {len(full_dataset)}")
    logger.log(f"  Classes       : {num_classes}")
    logger.log(f"  Class names   : {', '.join(class_names)}")

    # Subset for quick testing
    if DATA_SUBSET_FRACTION < 1.0:
        n_keep = max(num_classes, int(len(full_dataset) * DATA_SUBSET_FRACTION))
        rng = random.Random(SEED)
        keep_idx = rng.sample(range(len(full_dataset)), n_keep)
        full_dataset = Subset(full_dataset, keep_idx)
        full_dataset.targets = [full_dataset.dataset.targets[i] for i in keep_idx]
        logger.log(f"  Subset        : {n_keep} videos ({DATA_SUBSET_FRACTION:.0%})")

    # ---- Split ----
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, TRAIN_RATIO, VAL_RATIO, SEED
    )
    logger.log(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # Create separate dataset views with appropriate transforms
    # Training set uses train_spatial, eval sets use eval_spatial
    train_set = Subset(full_dataset, train_idx)

    # For val/test we need eval transforms — create a parallel dataset
    eval_dataset = ASLVideoDataset(
        root=str(video_root),
        clip_len=CLIP_LEN,
        frame_stride=FRAME_STRIDE,
        frame_size=FRAME_SIZE,
        spatial_transform=eval_spatial,
        is_train=False,
    )
    val_set = Subset(eval_dataset, val_idx)
    test_set = Subset(eval_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Class distribution
    train_labels = [full_dataset.targets[i] if hasattr(full_dataset, 'targets')
                    else full_dataset.dataset.targets[i] for i in train_idx]
    dist = Counter(train_labels)
    logger.log("\n  Training class distribution:")
    for cls_idx in sorted(dist.keys()):
        logger.log(f"    {class_names[cls_idx]:>12s}: {dist[cls_idx]:4d} videos")

    # ---- Model ----
    logger.log("\n" + "=" * 70)
    logger.log("  MODEL ARCHITECTURE")
    logger.log("=" * 70)

    model = ASLVideoNet(
        num_classes=num_classes,
        model_type=MODEL_TYPE,
        dropout_rate=DROPOUT_RATE,
        hidden_dim=HIDDEN_DIM,
        freeze_stages=FREEZE_BACKBONE_STAGES,
    ).to(DEVICE)

    total_params, trainable_params = count_parameters(model)
    feature_dim = 768 if MODEL_TYPE == "mvit_v2_s" else 512

    if MODEL_TYPE == "mvit_v2_s":
        num_blocks = len(model.backbone.blocks)
        logger.log(f"  Architecture  : MViTv2-S + custom head")
        logger.log(f"  Total params  : {total_params:,}")
        logger.log(f"  Trainable     : {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        logger.log(f"  Transformer blocks: {num_blocks} (first {FREEZE_BACKBONE_STAGES} frozen)")
        logger.log(f"  Frozen        : patch_embed + cls_positional_encoding + blocks[0..{FREEZE_BACKBONE_STAGES-1}]")
    else:
        logger.log(f"  Architecture  : R(2+1)D-18 + custom head")
        logger.log(f"  Total params  : {total_params:,}")
        logger.log(f"  Trainable     : {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        logger.log(f"  Frozen stages : stem + layer1..{FREEZE_BACKBONE_STAGES}")
    logger.log(f"  Head          : Linear({feature_dim}->{HIDDEN_DIM})->ReLU->Dropout({DROPOUT_RATE})"
               f"->Linear({HIDDEN_DIM}->{num_classes})")

    # ---- Optimizer & Scheduler ----
    # Differential learning rates: backbone gets base LR, head gets HEAD_LR_MULTIPLIER * base LR
    criterion = nn.CrossEntropyLoss()

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "head" in name or "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE},
        {"params": head_params, "lr": LEARNING_RATE * HEAD_LR_MULTIPLIER},
    ], weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE,
                                               gamma=STEP_LR_GAMMA)

    logger.log(f"  Optimizer     : AdamW (wd={WEIGHT_DECAY})")
    logger.log(f"  Backbone LR   : {LEARNING_RATE}")
    logger.log(f"  Head LR       : {LEARNING_RATE * HEAD_LR_MULTIPLIER}")
    logger.log(f"  Backbone params (trainable): {len(backbone_params)}")
    logger.log(f"  Head params   : {len(head_params)}")
    logger.log(f"  Scheduler     : {SCHEDULER_TYPE}")
    logger.log(f"  Criterion     : CrossEntropyLoss")

    # ---- Training ----
    logger.log("\n" + "=" * 70)
    logger.log("  TRAINING")
    logger.log("=" * 70)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "grad_norm": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = output_dir / f"{OUTPUT_PREFIX}_best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(f"\n  Epoch {epoch}/{NUM_EPOCHS}  (lr={current_lr:.2e})")
        logger.log("  " + "-" * 50)

        # Train
        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, logger, epoch
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["grad_norm"].append(grad_norm)

        epoch_time = time.time() - epoch_start

        # ETA
        avg_epoch_time = (time.time() - start_time) / epoch
        eta_seconds = avg_epoch_time * (NUM_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        logger.log(f"  Train — loss: {train_loss:.4f}  acc: {100*train_acc:.2f}%")
        logger.log(f"  Val   — loss: {val_loss:.4f}  acc: {100*val_acc:.2f}%")
        logger.log(f"  Grad norm: {grad_norm:.4f}  Time: {epoch_time:.1f}s")
        logger.log(f"  ETA: {eta_str}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"  ** New best model saved (val_acc={100*val_acc:.2f}%) **")
        else:
            patience_counter += 1
            logger.log(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if EARLY_STOPPING_PATIENCE > 0 and patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.log(f"\n  Early stopping triggered at epoch {epoch}")
            break

    train_time = time.time() - start_time
    logger.log(f"\n  Training complete in {train_time:.1f}s")
    logger.log(f"  Best val accuracy: {100*best_val_acc:.2f}% at epoch {best_epoch}")

    # ---- Load best model ----
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    logger.log(f"  Loaded best model from epoch {best_epoch}")

    # ---- Single-clip test evaluation ----
    logger.log("\n" + "=" * 70)
    logger.log("  TEST EVALUATION — Single Clip")
    logger.log("=" * 70)

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    logger.log(f"  Test loss     : {test_loss:.4f}")
    logger.log(f"  Test accuracy : {100*test_acc:.2f}%")

    # ---- Multi-clip test evaluation ----
    logger.log("\n" + "=" * 70)
    logger.log(f"  TEST EVALUATION — Multi-Clip (K={MULTI_CLIP_K})")
    logger.log("=" * 70)

    mc_preds, mc_labels = multi_clip_evaluate(
        model, eval_dataset, test_idx, DEVICE, k=MULTI_CLIP_K, logger=logger
    )
    mc_correct = (mc_preds == mc_labels).sum()
    mc_acc = mc_correct / len(mc_labels)
    logger.log(f"  Multi-clip accuracy: {100*mc_acc:.2f}%  "
               f"(+{100*(mc_acc - test_acc):.2f}% vs single-clip)")

    # Use multi-clip results for final evaluation
    final_preds = mc_preds
    final_labels = mc_labels

    # ---- Confusion matrix ----
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(final_labels, final_preds):
        cm[t, p] += 1

    # ---- Per-class metrics ----
    logger.log("\n  Per-class results (multi-clip):")
    logger.log(f"  {'Class':>12s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}")
    logger.log("  " + "-" * 50)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                 if (precision[i] + recall[i]) > 0 else 0)
        logger.log(f"  {class_names[i]:>12s}  {precision[i]:6.3f}  {recall[i]:6.3f}  "
                   f"{f1[i]:6.3f}  {int(cm[i].sum()):7d}")

    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    logger.log("  " + "-" * 50)
    logger.log(f"  {'Macro avg':>12s}  {macro_p:6.3f}  {macro_r:6.3f}  {macro_f1:6.3f}")

    # ---- Visualizations ----
    logger.log("\n" + "=" * 70)
    logger.log("  SAVING VISUALIZATIONS")
    logger.log("=" * 70)

    p = plot_training_curves(history, output_dir, OUTPUT_PREFIX)
    logger.raw(f"  Saved: {p.name}")

    p = plot_confusion_matrix(cm, class_names, output_dir, OUTPUT_PREFIX,
                              title=f"ASL Video Confusion Matrix (acc={100*mc_acc:.1f}%)")
    logger.raw(f"  Saved: {p.name}")

    p = plot_gradient_norms(history, output_dir, OUTPUT_PREFIX)
    logger.raw(f"  Saved: {p.name}")

    p = plot_per_class_accuracy(cm, class_names, output_dir, OUTPUT_PREFIX)
    logger.raw(f"  Saved: {p.name}")

    p = plot_sample_clips(eval_dataset, test_idx, class_names,
                          final_preds, final_labels, output_dir, OUTPUT_PREFIX)
    logger.raw(f"  Saved: {p.name}")

    # Temporal sensitivity — the most interesting visualization for video
    logger.log("\n  Computing temporal sensitivity analysis...")
    p = plot_temporal_sensitivity(
        model, eval_dataset, test_idx[:6], DEVICE, class_names,
        output_dir, OUTPUT_PREFIX
    )
    logger.raw(f"  Saved: {p.name}")

    # ---- Save CSV history ----
    csv_path = output_dir / f"{OUTPUT_PREFIX}_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                         "lr", "grad_norm"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
                f"{history['lr'][i]:.8f}",
                f"{history['grad_norm'][i]:.6f}",
            ])
    logger.raw(f"  Saved: {csv_path.name}")

    # ---- Save JSON results ----
    total_time = time.time() - start_time
    results = {
        "model": model_display.get(MODEL_TYPE, MODEL_TYPE),
        "model_type": MODEL_TYPE,
        "task": "ASL Video Sign Recognition",
        "timestamp": str(datetime.datetime.now()),
        "device": str(DEVICE),
        "total_time_seconds": round(total_time, 2),
        "config": {
            "model_type": MODEL_TYPE,
            "clip_len": CLIP_LEN,
            "frame_stride": FRAME_STRIDE,
            "frame_size": FRAME_SIZE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "head_lr_multiplier": HEAD_LR_MULTIPLIER,
            "freeze_stages": FREEZE_BACKBONE_STAGES,
            "dropout_rate": DROPOUT_RATE,
            "hidden_dim": HIDDEN_DIM,
            "multi_clip_k": MULTI_CLIP_K,
            "augment_hflip": AUGMENT_HFLIP,
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
            "synthetic_data": GENERATE_SYNTHETIC,
        },
        "dataset": {
            "total_videos": len(full_dataset),
            "num_classes": num_classes,
            "class_names": class_names,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 6),
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "test_results": {
            "single_clip_accuracy": round(test_acc, 6),
            "single_clip_loss": round(test_loss, 6),
            "multi_clip_accuracy": round(float(mc_acc), 6),
            "multi_clip_improvement": round(float(mc_acc - test_acc), 6),
            "macro_precision": round(float(macro_p), 6),
            "macro_recall": round(float(macro_r), 6),
            "macro_f1": round(float(macro_f1), 6),
        },
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i in range(num_classes)
        },
        "history": history,
    }

    json_path = output_dir / f"{OUTPUT_PREFIX}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_results.json")

    logger.log("\n" + "=" * 70)
    logger.log("ALL DONE")
    logger.log("=" * 70)
    logger.log(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.log(f"  Single-clip test accuracy: {100*test_acc:.2f}%")
    logger.log(f"  Multi-clip test accuracy:  {100*mc_acc:.2f}%")
    logger.log(f"  Outputs saved to: {output_dir}")
    logger.close()


if __name__ == "__main__":
    main()
