#!/usr/bin/env python3
"""
VisionAI_Final_SNN.py
Spiking Neural Network (SNN) for ASL Sign Classification from DVS Event-Camera Data.

Uses snntorch for LIF neurons, tonic for full ASL-DVS dataset loading, and PyTorch
for training.  Supports two data backends:
  - USE_TONIC = True  -> tonic.datasets.ASLDVS (full 100,800-sample dataset, 24 classes)
  - USE_TONIC = False -> manual .aedat file loading (legacy 125-file subset)

Architecture v4: deeper 4-layer Conv + SNN with LIF neurons and adaptive global pooling.
"""

import os
import sys
import re
import json
import time
import copy
import struct
import random
import logging
import pathlib
import datetime
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    raise ImportError(
        "snntorch is required. Install via: pip install snntorch"
    )

# Tonic availability check -- if not installed, fall back to manual loading
try:
    import tonic
    import tonic.io
    import tonic.transforms as tonic_transforms
    from tonic import DiskCachedDataset
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False
    print("[WARN] tonic not available; will rely on manual AEDAT decode only.")


# ============================================================================
# CONFIGURATION  --  All tunable knobs live here
# ============================================================================

# --- Data backend ---
USE_TONIC = True  # True = use Tonic library for full ASL-DVS dataset, False = manual .aedat loading

# --- Paths ---
ASLDVS_ROOT    = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\ASL_DVS"
TONIC_DATA_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\ASL_DVS_tonic"
OUTPUT_DIR     = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX  = "VisionAI_Final_SNN_v4"

# --- DVS sensor ---
DVS_WIDTH          = 240
DVS_HEIGHT         = 180
SPATIAL_DOWNSAMPLE = 4       # factor to shrink spatial dims
NUM_TIME_BINS      = 16      # temporal bins for event frames
MAX_EVENTS         = 300_000 # cap per file (manual mode only)

# --- Data ---
DATA_SUBSET_FRACTION = 1.0   # fraction of dataset to use (1.0 = all)
TRAIN_RATIO          = 0.70  # stratified split
VAL_RATIO            = 0.15  # val portion; test = 1 - train - val
SEED                 = 42
NUM_WORKERS          = 0     # DataLoader workers (0 = main thread)

# --- Training ---
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
SCHEDULER_TYPE  = "cosine"   # "cosine" | "step" | "none"

# --- SNN ---
LIF_BETA        = 0.9
SURROGATE_SLOPE = 25
FC_HIDDEN       = 128        # legacy model only

# --- Architecture ---
NUM_CLASSES     = 24          # Full ASL-DVS: 24 classes (a-y, no j or z)

# --- Early stopping ---
EARLY_STOPPING_PATIENCE = 8  # 0 to disable

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Derived
FRAME_H = DVS_HEIGHT // SPATIAL_DOWNSAMPLE
FRAME_W = DVS_WIDTH  // SPATIAL_DOWNSAMPLE

# Override USE_TONIC if tonic not installed
if USE_TONIC and not TONIC_AVAILABLE:
    print("[WARN] USE_TONIC=True but tonic not installed. Falling back to manual loading.")
    USE_TONIC = False


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{prefix}.log")
    logger = logging.getLogger("SNN")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# AEDAT LOADING UTILITIES  (manual / legacy mode)
# ============================================================================

# ASL alphabet minus 'j' (requires motion) -> 25 classes for legacy mode
ASL_CLASSES_25 = [c for c in "abcdefghiklmnopqrstuvwxyz"]
LABEL_TO_IDX_25 = {c: i for i, c in enumerate(ASL_CLASSES_25)}

# Full ASL-DVS via Tonic: 24 classes (a-y, no j or z)
ASL_CLASSES_24 = [c for c in "abcdefghiklmnopqrstuvwxy"]
LABEL_TO_IDX_24 = {c: i for i, c in enumerate(ASL_CLASSES_24)}


def infer_label_from_filename(filepath):
    """Parse stem, split on non-alphanumeric, find single alpha char -> label."""
    stem = pathlib.Path(filepath).stem
    parts = re.split(r"[^a-zA-Z]+", stem)
    for part in parts:
        if len(part) == 1 and part.lower() in LABEL_TO_IDX_25:
            return part.lower()
    # fallback: first alpha char in stem
    for ch in stem:
        if ch.isalpha() and ch.lower() in LABEL_TO_IDX_25:
            return ch.lower()
    return None


def _davis240_manual_decode(filepath, max_events=MAX_EVENTS):
    """Manual AEDAT 2.0 decode for DAVIS240 as fallback."""
    events_x, events_y, events_t, events_p = [], [], [], []
    with open(filepath, "rb") as f:
        # skip header lines starting with '#'
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return None
            if not line.startswith(b"#"):
                f.seek(pos)
                break
        count = 0
        while count < max_events:
            buf = f.read(8)
            if len(buf) < 8:
                break
            addr, ts = struct.unpack(">II", buf)
            # DAVIS240: x in bits 12-21, y in bits 1-11 (22-bit address)
            x = (addr >> 12) & 0x3FF
            y = (addr >> 1)  & 0x7FF
            p = addr & 1
            if 0 <= x < DVS_WIDTH and 0 <= y < DVS_HEIGHT:
                events_x.append(x)
                events_y.append(y)
                events_t.append(ts)
                events_p.append(p)
                count += 1
    if len(events_x) == 0:
        return None
    return {
        "x": np.array(events_x, dtype=np.int16),
        "y": np.array(events_y, dtype=np.int16),
        "t": np.array(events_t, dtype=np.int64),
        "p": np.array(events_p, dtype=np.int8),
    }


def decode_aedat_file(filepath, max_events=MAX_EVENTS):
    """
    Decode an .aedat file using tonic.io (preferred) with DAVIS240 manual
    fallback.  Returns dict with keys x, y, t, p as numpy arrays or None.
    """
    events = None

    # --- tonic.io approach ---
    if TONIC_AVAILABLE:
        try:
            with open(filepath, "rb") as f:
                header_end, file_type = tonic.io.read_aedat2_header(f)
            with open(filepath, "rb") as f:
                f.seek(header_end)
                raw = tonic.io.read_aedat2(f)
            # tonic may return structured array or dict
            if hasattr(raw, "dtype") and raw.dtype.names is not None:
                names = raw.dtype.names
                if "x" in names and "y" in names:
                    ev = raw[:max_events]
                    events = {
                        "x": ev["x"].astype(np.int16),
                        "y": ev["y"].astype(np.int16),
                        "t": ev["t"].astype(np.int64) if "t" in names
                             else ev["timeStamp"].astype(np.int64),
                        "p": ev["p"].astype(np.int8) if "p" in names
                             else ev["polarity"].astype(np.int8),
                    }
                elif "address" in names and "timeStamp" in names:
                    ev = raw[:max_events]
                    addr = ev["address"].astype(np.int64)
                    x = ((addr >> 12) & 0x3FF).astype(np.int16)
                    y = ((addr >> 1)  & 0x7FF).astype(np.int16)
                    p = (addr & 1).astype(np.int8)
                    t = ev["timeStamp"].astype(np.int64)
                    mask = (x >= 0) & (x < DVS_WIDTH) & (y >= 0) & (y < DVS_HEIGHT)
                    events = {
                        "x": x[mask], "y": y[mask],
                        "t": t[mask], "p": p[mask],
                    }
        except Exception:
            events = None

    # --- manual fallback ---
    if events is None:
        events = _davis240_manual_decode(filepath, max_events)

    return events


def events_to_frames(events, num_bins=NUM_TIME_BINS,
                     frame_h=FRAME_H, frame_w=FRAME_W):
    """
    Bin events into tensor of shape (num_bins, 2, frame_h, frame_w).
    Channel 0 = ON polarity, Channel 1 = OFF polarity.
    Spatial downsampling applied via integer division.
    Normalizes each frame independently.
    """
    frames = np.zeros((num_bins, 2, frame_h, frame_w), dtype=np.float32)
    if events is None or len(events["x"]) == 0:
        return frames

    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.int64)
    p = events["p"].astype(np.int64)

    # spatial downsample
    x_ds = x // SPATIAL_DOWNSAMPLE
    y_ds = y // SPATIAL_DOWNSAMPLE
    x_ds = np.clip(x_ds, 0, frame_w - 1)
    y_ds = np.clip(y_ds, 0, frame_h - 1)

    # temporal binning
    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_bin = np.zeros_like(t)
    else:
        t_bin = ((t - t_min) * num_bins // (t_max - t_min + 1))
    t_bin = np.clip(t_bin, 0, num_bins - 1)

    for i in range(len(x)):
        pol_ch = 0 if p[i] > 0 else 1
        frames[int(t_bin[i]), pol_ch, int(y_ds[i]), int(x_ds[i])] += 1.0

    # normalise each frame
    for b in range(num_bins):
        for c in range(2):
            mx = frames[b, c].max()
            if mx > 0:
                frames[b, c] /= mx

    return frames


# ============================================================================
# LEGACY DATASET (manual .aedat loading)
# ============================================================================

class ASLDVSDataset(Dataset):
    """
    Loads all .aedat files from ASLDVS_ROOT, caches decoded frames in memory
    (the dataset is small enough to fit).
    """

    def __init__(self, root, logger=None):
        self.root = root
        self.file_list = []  # (filepath, label_idx)
        self.cache = {}

        # discover files -- recursively search all subdirectories
        aedat_paths = sorted(pathlib.Path(root).rglob("*.aedat"))
        if logger:
            logger.info(f"  Found {len(aedat_paths)} .aedat files (recursive search)")
        for fpath in aedat_paths:
            label = infer_label_from_filename(str(fpath))
            if label is None or label not in LABEL_TO_IDX_25:
                if logger:
                    logger.warning(f"Skipping {fpath.name}: cannot infer label")
                continue
            self.file_list.append((str(fpath), LABEL_TO_IDX_25[label]))

        # pre-load into cache
        total = len(self.file_list)
        for idx, (fpath, lbl) in enumerate(self.file_list):
            if logger and (idx % 10 == 0 or idx == total - 1):
                logger.info(f"  Pre-loading [{idx+1}/{total}] {os.path.basename(fpath)}")
            events = decode_aedat_file(fpath)
            frames = events_to_frames(events)
            self.cache[idx] = torch.from_numpy(frames)  # (T, 2, H, W)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        frames = self.cache[idx]          # (T, 2, H, W)
        label  = self.file_list[idx][1]
        return frames, label

    def label_counts(self):
        counts = defaultdict(int)
        for _, lbl in self.file_list:
            counts[lbl] += 1
        return counts

    def labels(self):
        return [lbl for _, lbl in self.file_list]


# ============================================================================
# TONIC DATASET WRAPPER
# ============================================================================

class TonicASLDVSWrapper(Dataset):
    """
    Wraps the Tonic ASLDVS DiskCachedDataset with spatial downsampling,
    normalisation, and optional data augmentation.
    """

    def __init__(self, cached_dataset, all_labels, indices, spatial_downsample=4,
                 augment=False):
        """
        Args:
            cached_dataset: DiskCachedDataset wrapping tonic.datasets.ASLDVS
            all_labels: list of int labels for the entire dataset
            indices: list of int indices into the dataset for this split
            spatial_downsample: factor to reduce spatial resolution
            augment: whether to apply data augmentation (training only)
        """
        self.dataset = cached_dataset
        self.all_labels = all_labels
        self.indices = indices
        self.spatial_ds = spatial_downsample
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, label = self.dataset[real_idx]

        # data is a numpy array from tonic ToFrame: (T, 2, H, W)
        if isinstance(data, np.ndarray):
            frames = torch.from_numpy(data.astype(np.float32))
        else:
            frames = data.float()

        # Spatial downsample using average pooling
        if self.spatial_ds > 1:
            T, C, H, W = frames.shape
            # Reshape for avg_pool2d: (T*C, 1, H, W)
            frames = F.avg_pool2d(frames.view(T * C, 1, H, W),
                                  kernel_size=self.spatial_ds,
                                  stride=self.spatial_ds)
            frames = frames.view(T, C, frames.shape[-2], frames.shape[-1])

        # Normalize per-frame per-channel
        for t in range(frames.shape[0]):
            for c in range(frames.shape[1]):
                mx = frames[t, c].max()
                if mx > 0:
                    frames[t, c] /= mx

        # Data augmentation for training
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[-1])
            # Random temporal shift (roll time bins)
            shift = random.randint(-2, 2)
            if shift != 0:
                frames = torch.roll(frames, shifts=shift, dims=0)
            # Random noise
            if random.random() < 0.3:
                noise = torch.randn_like(frames) * 0.05
                frames = (frames + noise).clamp(0, 1)

        return frames, int(label)


# ============================================================================
# SNN MODELS
# ============================================================================

class ASLDVS_SNN(nn.Module):
    """
    Legacy 3-layer convolutional SNN with LIF neurons + FC readout.
    Input: (batch, T, 2, H, W) -- T time steps fed sequentially.
    """

    def __init__(self, num_classes, frame_h=FRAME_H, frame_w=FRAME_W,
                 beta=LIF_BETA, slope=SURROGATE_SLOPE, fc_hidden=FC_HIDDEN):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # Conv block 1
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Conv block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Conv block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.lif3  = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Global pooling
        self.pool  = nn.AdaptiveAvgPool2d(1)

        # FC layers
        self.fc1  = nn.Linear(64, fc_hidden)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(fc_hidden, num_classes)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        """
        x: (batch, T, 2, H, W)
        Returns: summed membrane potential (batch, num_classes),
                 dict of spike recordings per layer per timestep.
        """
        batch, T = x.shape[0], x.shape[1]

        # init hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        mem_out_sum = torch.zeros(batch, self.fc2.out_features, device=x.device)

        # for spike raster recording
        spike_record = {
            "conv1": [], "conv2": [], "conv3": [],
            "fc1": [], "fc2": [],
        }

        for t in range(T):
            xt = x[:, t]  # (batch, 2, H, W)

            cur1 = self.bn1(self.conv1(xt))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            pooled = self.pool(spk3).flatten(1)  # (batch, 64)

            cur4 = self.fc1(pooled)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            mem_out_sum += mem5

            # record spike counts (sum over spatial dims, mean over batch)
            spike_record["conv1"].append(spk1.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["conv2"].append(spk2.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["conv3"].append(spk3.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["fc1"].append(spk4.detach().mean(0).cpu())
            spike_record["fc2"].append(spk5.detach().mean(0).cpu())

        return mem_out_sum, spike_record


class ASLDVS_SNN_V4(nn.Module):
    """
    Deeper Conv-SNN with 4 conv layers, global adaptive pooling, and LIF neurons.
    Input: (batch, T, 2, H, W) where H/W are post-spatial-downsample dims.
    """

    def __init__(self, num_classes=24, num_time_bins=16, beta=0.9,
                 surrogate_slope=25):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> 256 x 1 x 1

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # LIF neurons (one per layer)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

        self.num_time_bins = num_time_bins

    def forward(self, x):
        """
        x: (B, T, 2, H, W)
        Returns: summed membrane potential (B, num_classes),
                 dict of spike recordings per layer per timestep.
        """
        B, T = x.shape[0], x.shape[1]

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem_out = self.lif_out.init_leaky()

        out_sum = torch.zeros(B, self.fc2.out_features, device=x.device)

        spike_record = {
            "conv1": [], "conv2": [], "conv3": [], "conv4": [],
            "fc1": [], "fc_out": [],
        }

        for t in range(T):
            frame = x[:, t]  # (B, 2, H, W)

            h = self.pool1(self.bn1(self.conv1(frame)))
            spk1, mem1 = self.lif1(h, mem1)

            h = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(h, mem2)

            h = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(h, mem3)

            h = self.bn4(self.conv4(spk3))
            h = self.global_pool(h)
            spk4, mem4 = self.lif4(h.flatten(1), mem4)

            h = self.fc1(spk4)
            spk5, mem5 = self.lif5(h, mem5)

            spk_out, mem_out = self.lif_out(self.fc2(spk5), mem_out)
            out_sum += mem_out

            # Record spike counts
            spike_record["conv1"].append(spk1.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["conv2"].append(spk2.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["conv3"].append(spk3.detach().sum(dim=(2, 3)).mean(0).cpu())
            spike_record["conv4"].append(spk4.detach().mean(0).cpu())
            spike_record["fc1"].append(spk5.detach().mean(0).cpu())
            spike_record["fc_out"].append(spk_out.detach().mean(0).cpu())

        return out_sum, spike_record


# ============================================================================
# TRAIN / EVAL HELPERS
# ============================================================================

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []

    for frames, labels in loader:
        frames = frames.to(device)      # (B, T, 2, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        out, _ = model(frames)
        loss = criterion(out, labels)
        loss.backward()
        gn = compute_grad_norm(model)
        grad_norms.append(gn)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    avg_gn = np.mean(grad_norms) if grad_norms else 0.0
    return avg_loss, acc, avg_gn


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    spike_record_sample = None

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        out, sr = model(frames)
        loss = criterion(out, labels)

        running_loss += loss.item() * labels.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if spike_record_sample is None:
            spike_record_sample = sr

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels), spike_record_sample


# ============================================================================
# METRICS (no sklearn)
# ============================================================================

def compute_precision_recall_f1(all_labels, all_preds, num_classes):
    """Per-class precision, recall, F1 computed manually."""
    metrics = {}
    for c in range(num_classes):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1,
                       "tp": tp, "fp": fp, "fn": fn}
    return metrics


def confusion_matrix_manual(all_labels, all_preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pr in zip(all_labels, all_preds):
        cm[gt, pr] += 1
    return cm


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

def stratified_split_legacy(dataset, train_ratio, seed):
    """Return (train_indices, val_indices) with stratification for legacy dataset."""
    rng = np.random.RandomState(seed)
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(dataset.labels()):
        label_to_indices[lbl].append(idx)

    train_idx, val_idx = [], []
    for lbl in sorted(label_to_indices.keys()):
        indices = label_to_indices[lbl]
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:])
    return train_idx, val_idx


def stratified_split_tonic(all_labels, train_ratio, val_ratio, seed):
    """
    Return (train_indices, val_indices, test_indices) with stratification.
    all_labels: list of int labels for entire dataset.
    """
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(all_labels):
        label_to_indices[lbl].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def extract_tonic_labels(cached_dataset, dataset_len, tonic_data_root, logger=None):
    """
    Extract labels from the Tonic ASLDVS dataset.
    Uses a JSON cache file to avoid re-extracting on subsequent runs.
    Falls back to directory scanning if available, otherwise iterates the dataset.
    """
    labels_cache_path = os.path.join(tonic_data_root, "labels_cache.json")

    if os.path.exists(labels_cache_path):
        if logger:
            logger.info(f"  Loading cached labels from {labels_cache_path}")
        with open(labels_cache_path, "r", encoding="utf-8") as f:
            all_labels = json.load(f)
        if len(all_labels) == dataset_len:
            return all_labels
        else:
            if logger:
                logger.info(f"  Label cache size mismatch ({len(all_labels)} vs {dataset_len}), re-extracting...")

    # Try directory scanning approach first (much faster)
    data_dir = pathlib.Path(tonic_data_root) / "ASLDVS"
    if data_dir.exists():
        if logger:
            logger.info("  Attempting fast label extraction via directory scan...")
        class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        # The Tonic ASLDVS dataset orders by class directory alphabetically
        all_labels = []
        for ci, cls_name in enumerate(class_dirs):
            n_files = len(list((data_dir / cls_name).glob("*.aedat")))
            if logger:
                logger.info(f"    Class '{cls_name}' (idx {ci}): {n_files} files")
            all_labels.extend([ci] * n_files)

        if len(all_labels) == dataset_len:
            with open(labels_cache_path, "w", encoding="utf-8") as f:
                json.dump(all_labels, f)
            if logger:
                logger.info(f"  Directory scan successful: {len(all_labels)} labels extracted")
            return all_labels
        elif logger:
            logger.info(f"  Directory scan gave {len(all_labels)} labels but dataset has "
                        f"{dataset_len} samples. Falling back to iteration...")

    # Fallback: iterate the dataset to extract labels (slow on first run)
    if logger:
        logger.info(f"  Extracting labels by iterating {dataset_len} samples (this may take a while on first run)...")
    all_labels = []
    t0 = time.time()
    for i in range(dataset_len):
        _, label = cached_dataset[i]
        all_labels.append(int(label))
        if logger and (i % 5000 == 0 or i == dataset_len - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (dataset_len - i - 1) / rate if rate > 0 else 0
            logger.info(f"    [{i+1}/{dataset_len}] ({rate:.1f} samples/s, ETA: {eta:.0f}s)")

    with open(labels_cache_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f)
    if logger:
        logger.info(f"  Labels extracted and cached in {time.time() - t0:.1f}s")

    return all_labels


# ============================================================================
# PLOTS
# ============================================================================

def plot_training_curves(epoch_log, output_dir, prefix):
    """Loss + accuracy curves."""
    epochs = [e["epoch"] for e in epoch_log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [e["train_loss"] for e in epoch_log], label="Train Loss")
    ax1.plot(epochs, [e["val_loss"] for e in epoch_log], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss Curves")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, [e["train_acc"] for e in epoch_log], label="Train Acc")
    ax2.plot(epochs, [e["val_acc"] for e in epoch_log], label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.set_title("Accuracy Curves")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_lr_schedule(epoch_log, output_dir, prefix):
    epochs = [e["epoch"] for e in epoch_log]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [e["lr"] for e in epoch_log], marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_title("LR Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_lr_schedule.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(cm, class_names, output_dir, prefix):
    per_class_acc = []
    for i in range(len(class_names)):
        total_i = cm[i].sum()
        per_class_acc.append(cm[i, i] / total_i if total_i > 0 else 0.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(class_names)), per_class_acc, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_per_class_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_spike_raster(spike_record, output_dir, prefix, num_bins=NUM_TIME_BINS):
    """Spike counts per layer per timestep."""
    fig, axes = plt.subplots(len(spike_record), 1, figsize=(12, 2.5 * len(spike_record)),
                              sharex=True)
    if len(spike_record) == 1:
        axes = [axes]
    for ax, (layer_name, spikes_list) in zip(axes, spike_record.items()):
        # spikes_list: list of T tensors, each (num_channels,)
        T = min(len(spikes_list), num_bins)
        counts = []
        for t in range(T):
            counts.append(spikes_list[t].sum().item())
        ax.bar(range(T), counts, color="darkorange", alpha=0.8)
        ax.set_ylabel(f"{layer_name}\nSpike Count")
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Spike Raster (total spike counts per layer per timestep)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_spike_raster.png"), dpi=150)
    plt.close(fig)


def plot_dvs_event_frames(dataset, sample_idx, class_names, output_dir, prefix,
                           num_bins=NUM_TIME_BINS):
    """Visualise ON/OFF polarity channels across time bins for one sample."""
    frames, label = dataset[sample_idx]  # (T, 2, H, W)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    T = frames.shape[0]
    n_show = min(T, 8)
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for t in range(n_show):
        axes[0, t].imshow(frames[t, 0], cmap="Reds", vmin=0, vmax=1)
        axes[0, t].set_title(f"ON t={t}", fontsize=8)
        axes[0, t].axis("off")
        axes[1, t].imshow(frames[t, 1], cmap="Blues", vmin=0, vmax=1)
        axes[1, t].set_title(f"OFF t={t}", fontsize=8)
        axes[1, t].axis("off")
    class_name = class_names[label] if isinstance(label, int) and label < len(class_names) else str(label)
    fig.suptitle(f"DVS Event Frames  --  Label: '{class_name}'  (sample {sample_idx})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_dvs_event_frames.png"), dpi=150)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    logger = setup_logging(OUTPUT_DIR, OUTPUT_PREFIX)
    logger.info("=" * 70)
    logger.info("VisionAI Final SNN  --  ASL-DVS Classification")
    logger.info("=" * 70)

    # ---- Determine class set ----
    if USE_TONIC:
        asl_classes = ASL_CLASSES_24
        num_classes = NUM_CLASSES  # 24
    else:
        asl_classes = ASL_CLASSES_25
        num_classes = len(ASL_CLASSES_25)  # 25

    # ---- Log all config ----
    config = {
        "USE_TONIC": USE_TONIC,
        "ASLDVS_ROOT": ASLDVS_ROOT, "TONIC_DATA_ROOT": TONIC_DATA_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR, "OUTPUT_PREFIX": OUTPUT_PREFIX,
        "DVS_WIDTH": DVS_WIDTH, "DVS_HEIGHT": DVS_HEIGHT,
        "SPATIAL_DOWNSAMPLE": SPATIAL_DOWNSAMPLE,
        "NUM_TIME_BINS": NUM_TIME_BINS, "MAX_EVENTS": MAX_EVENTS,
        "FRAME_H": FRAME_H, "FRAME_W": FRAME_W,
        "DATA_SUBSET_FRACTION": DATA_SUBSET_FRACTION,
        "TRAIN_RATIO": TRAIN_RATIO, "VAL_RATIO": VAL_RATIO,
        "SEED": SEED, "NUM_WORKERS": NUM_WORKERS,
        "BATCH_SIZE": BATCH_SIZE, "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE, "WEIGHT_DECAY": WEIGHT_DECAY,
        "SCHEDULER_TYPE": SCHEDULER_TYPE,
        "LIF_BETA": LIF_BETA, "SURROGATE_SLOPE": SURROGATE_SLOPE,
        "NUM_CLASSES": num_classes,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "DEVICE": DEVICE,
    }
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k:30s} = {v}")

    # ---- Dataset ----
    logger.info("-" * 50)
    t0_data = time.time()

    if USE_TONIC:
        # ================================================================
        # TONIC PATH: Full ASL-DVS dataset via tonic library
        # ================================================================
        logger.info("Loading ASL-DVS dataset via Tonic library ...")
        logger.info(f"  Data root: {TONIC_DATA_ROOT}")
        logger.info("  (First run will download the full dataset -- this may take a while)")

        sensor_size = tonic.datasets.ASLDVS.sensor_size  # (240, 180, 2)
        logger.info(f"  Sensor size: {sensor_size}")

        # Transform: Denoise + convert events to frames
        event_transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=NUM_TIME_BINS),
        ])

        # Create raw tonic dataset (downloads if needed)
        raw_tonic_dataset = tonic.datasets.ASLDVS(
            save_to=TONIC_DATA_ROOT, transform=event_transform
        )
        dataset_len = len(raw_tonic_dataset)
        logger.info(f"  Tonic dataset created: {dataset_len} samples")

        # Wrap in DiskCachedDataset for fast subsequent loads
        cache_dir = os.path.join(TONIC_DATA_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_dataset = DiskCachedDataset(
            dataset=raw_tonic_dataset,
            cache_path=cache_dir,
        )
        logger.info(f"  DiskCachedDataset ready (cache dir: {cache_dir})")

        # Extract labels
        logger.info("  Extracting labels for stratified split ...")
        all_labels = extract_tonic_labels(cached_dataset, dataset_len,
                                          TONIC_DATA_ROOT, logger=logger)

        data_load_time = time.time() - t0_data
        logger.info(f"Dataset loaded in {data_load_time:.1f}s  |  Total samples: {dataset_len}")

        # Per-class counts
        label_counts = defaultdict(int)
        for lbl in all_labels:
            label_counts[lbl] += 1
        logger.info("Per-class counts:")
        for idx in sorted(label_counts.keys()):
            cls_name = asl_classes[idx] if idx < len(asl_classes) else f"?{idx}"
            logger.info(f"  {cls_name:>2s} (idx {idx:2d}): {label_counts[idx]} samples")

        # Stratified split: train / val / test
        train_idx, val_idx, test_idx = stratified_split_tonic(
            all_labels, TRAIN_RATIO, VAL_RATIO, SEED
        )
        logger.info(f"Stratified split: {len(train_idx)} train, {len(val_idx)} val, "
                     f"{len(test_idx)} test")

        # Create wrapped datasets with augmentation for training
        train_set = TonicASLDVSWrapper(cached_dataset, all_labels, train_idx,
                                       spatial_downsample=SPATIAL_DOWNSAMPLE,
                                       augment=True)
        val_set = TonicASLDVSWrapper(cached_dataset, all_labels, val_idx,
                                     spatial_downsample=SPATIAL_DOWNSAMPLE,
                                     augment=False)
        test_set = TonicASLDVSWrapper(cached_dataset, all_labels, test_idx,
                                      spatial_downsample=SPATIAL_DOWNSAMPLE,
                                      augment=False)

    else:
        # ================================================================
        # LEGACY PATH: Manual .aedat file loading
        # ================================================================
        logger.info("Loading ASL-DVS dataset (manual .aedat loading) ...")
        full_dataset = ASLDVSDataset(ASLDVS_ROOT, logger=logger)
        data_load_time = time.time() - t0_data
        dataset_len = len(full_dataset)
        logger.info(f"Dataset loaded in {data_load_time:.1f}s  |  Total files: {dataset_len}")

        lc = full_dataset.label_counts()
        logger.info("Per-class counts:")
        for idx in sorted(lc.keys()):
            logger.info(f"  {asl_classes[idx]:>2s} (idx {idx:2d}): {lc[idx]} samples")

        # Subset if requested
        if DATA_SUBSET_FRACTION < 1.0:
            n_keep = max(1, int(len(full_dataset) * DATA_SUBSET_FRACTION))
            rng = np.random.RandomState(SEED)
            keep_idx = sorted(rng.choice(len(full_dataset), n_keep, replace=False))
            full_dataset_sub = Subset(full_dataset, keep_idx)
            logger.info(f"Using subset: {n_keep}/{len(full_dataset)} samples "
                         f"({DATA_SUBSET_FRACTION*100:.0f}%)")
        else:
            full_dataset_sub = full_dataset

        # Stratified split (legacy: train/val only)
        train_idx, val_idx = stratified_split_legacy(full_dataset, TRAIN_RATIO, SEED)
        train_set = Subset(full_dataset, train_idx)
        val_set   = Subset(full_dataset, val_idx)
        test_set  = None
        test_idx  = []
        logger.info(f"Stratified split: {len(train_set)} train, {len(val_set)} val "
                    f"(ratio {TRAIN_RATIO:.2f})")

    # ---- DataLoaders ----
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    if USE_TONIC and test_set is not None:
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    else:
        test_loader = None

    # ---- Model ----
    logger.info("-" * 50)
    if USE_TONIC:
        model = ASLDVS_SNN_V4(
            num_classes=num_classes,
            num_time_bins=NUM_TIME_BINS,
            beta=LIF_BETA,
            surrogate_slope=SURROGATE_SLOPE,
        ).to(DEVICE)
    else:
        model = ASLDVS_SNN(num_classes=num_classes).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model Architecture:")
    logger.info(str(model))
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        scheduler = None

    # ---- CSV epoch log ----
    csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_epoch_log.csv")
    csv_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                  "lr", "grad_norm", "epoch_time", "best_val_acc",
                  "epochs_since_improve"]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ---- Training loop ----
    logger.info("-" * 50)
    logger.info("Starting training ...")
    best_val_acc = 0.0
    best_model_state = None
    epochs_since_improve = 0
    epoch_log = []
    total_train_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc, avg_gn = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # ETA
        avg_epoch_time = (time.time() - total_train_start) / epoch
        eta_seconds = avg_epoch_time * (NUM_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = (f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0
                   else f"{eta_min}m {eta_sec}s" if eta_min > 0
                   else f"{eta_sec}s")

        # track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
            # save checkpoint
            ckpt_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": config,
            }, ckpt_path)
        else:
            epochs_since_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "lr": f"{current_lr:.6f}",
            "grad_norm": f"{avg_gn:.4f}",
            "epoch_time": f"{epoch_time:.1f}",
            "best_val_acc": f"{best_val_acc:.4f}",
            "epochs_since_improve": epochs_since_improve,
        }
        csv_writer.writerow(row)
        csv_file.flush()

        epoch_log.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": current_lr, "grad_norm": avg_gn,
            "epoch_time": epoch_time,
            "best_val_acc": best_val_acc,
            "epochs_since_improve": epochs_since_improve,
        })

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"TrainLoss {train_loss:.4f}  TrainAcc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f}  ValAcc {val_acc:.4f} | "
            f"LR {current_lr:.6f}  GradNorm {avg_gn:.2f} | "
            f"Best {best_val_acc:.4f}  NoImpv {epochs_since_improve} | "
            f"{epoch_time:.1f}s | ETA: {eta_str}"
        )

        # early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_since_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch} "
                        f"(patience={EARLY_STOPPING_PATIENCE})")
            break

    csv_file.close()
    total_train_time = time.time() - total_train_start
    logger.info(f"Training complete. Total time: {total_train_time:.1f}s")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")

    # ---- Final evaluation with best model ----
    logger.info("-" * 50)

    # Determine which loader to use for final eval
    if test_loader is not None:
        final_eval_loader = test_loader
        final_eval_name = "test"
        final_eval_size = len(test_set)
    else:
        final_eval_loader = val_loader
        final_eval_name = "validation"
        final_eval_size = len(val_set)

    logger.info(f"Final evaluation on {final_eval_name} set ({final_eval_size} samples) with best model ...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    val_loss_final, val_acc_final, all_preds, all_labels, spike_rec = evaluate(
        model, final_eval_loader, criterion, DEVICE)

    logger.info(f"Final {final_eval_name.capitalize()} Loss: {val_loss_final:.4f}  |  "
                f"Final {final_eval_name.capitalize()} Acc (Top-1): {val_acc_final:.4f}")

    # Per-class metrics
    prf = compute_precision_recall_f1(all_labels, all_preds, num_classes)
    logger.info("Per-class Precision / Recall / F1:")
    logger.info(f"  {'Class':>6s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  "
                f"{'TP':>4s}  {'FP':>4s}  {'FN':>4s}")
    for c in range(num_classes):
        m = prf[c]
        cls_name = asl_classes[c] if c < len(asl_classes) else f"?{c}"
        logger.info(f"  {cls_name:>6s}  {m['precision']:7.4f}  {m['recall']:7.4f}  "
                     f"{m['f1']:7.4f}  {m['tp']:4d}  {m['fp']:4d}  {m['fn']:4d}")

    # Macro averages
    macro_prec = np.mean([prf[c]["precision"] for c in range(num_classes)])
    macro_rec  = np.mean([prf[c]["recall"] for c in range(num_classes)])
    macro_f1   = np.mean([prf[c]["f1"] for c in range(num_classes)])
    logger.info(f"Macro Avg -- Precision: {macro_prec:.4f}  Recall: {macro_rec:.4f}  F1: {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix_manual(all_labels, all_preds, num_classes)
    logger.info("Confusion matrix (rows=true, cols=pred):")
    header = "      " + " ".join(f"{c:>3s}" for c in asl_classes[:num_classes])
    logger.info(header)
    for i in range(num_classes):
        cls_name = asl_classes[i] if i < len(asl_classes) else f"?{i}"
        row_str = f"  {cls_name:>2s}  " + " ".join(f"{cm[i,j]:3d}" for j in range(num_classes))
        logger.info(row_str)

    logger.info(f"Total training + eval time: {time.time() - total_train_start:.1f}s")

    # ---- Plots ----
    logger.info("-" * 50)
    logger.info("Generating plots ...")

    plot_training_curves(epoch_log, OUTPUT_DIR, OUTPUT_PREFIX)
    plot_lr_schedule(epoch_log, OUTPUT_DIR, OUTPUT_PREFIX)
    plot_confusion_matrix(cm, asl_classes[:num_classes], OUTPUT_DIR, OUTPUT_PREFIX)
    plot_per_class_accuracy(cm, asl_classes[:num_classes], OUTPUT_DIR, OUTPUT_PREFIX)

    if spike_rec is not None:
        plot_spike_raster(spike_rec, OUTPUT_DIR, OUTPUT_PREFIX)

    # DVS event frame visualization for first val sample
    if len(val_set) > 0:
        plot_dvs_event_frames(val_set, 0, asl_classes, OUTPUT_DIR, OUTPUT_PREFIX)

    logger.info("Plots saved.")

    # ---- Full results JSON ----
    per_class_results = {}
    for c in range(num_classes):
        m = prf[c]
        per_class_acc_val = cm[c, c] / cm[c].sum() if cm[c].sum() > 0 else 0.0
        cls_name = asl_classes[c] if c < len(asl_classes) else f"class_{c}"
        per_class_results[cls_name] = {
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "accuracy": round(per_class_acc_val, 4),
            "support": int(cm[c].sum()),
        }

    dataset_info = {
        "total_samples": dataset_len,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "num_classes": num_classes,
        "class_names": list(asl_classes[:num_classes]),
        "data_backend": "tonic" if USE_TONIC else "manual_aedat",
    }
    if USE_TONIC and test_set is not None:
        dataset_info["test_size"] = len(test_set)

    results = {
        "config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in config.items()},
        "dataset": dataset_info,
        "model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "architecture": "ASLDVS_SNN_V4" if USE_TONIC else "ASLDVS_SNN",
        },
        "training": {
            "epochs_run": len(epoch_log),
            "total_time_seconds": round(total_train_time, 1),
            "best_val_acc": round(best_val_acc, 4),
            "final_eval_set": final_eval_name,
            "final_eval_loss": round(val_loss_final, 4),
            "final_eval_acc": round(val_acc_final, 4),
        },
        "metrics": {
            "top1_accuracy": round(val_acc_final, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "per_class": per_class_results,
        "epoch_log": epoch_log,
        "confusion_matrix": cm.tolist(),
    }

    json_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results JSON saved to {json_path}")

    logger.info("=" * 70)
    logger.info("Done.")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
