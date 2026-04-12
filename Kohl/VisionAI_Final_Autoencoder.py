"""
VisionAI Final Autoencoder - ASL Sign Language Classifier
Two-phase training:
  Phase 1: Convolutional autoencoder for unsupervised image reconstruction
  Phase 2: Freeze encoder, attach classification head, then fine-tune jointly
"""

import os
import sys
import time
import json
import csv
import math
import random
import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONFIG SECTION - All tunable knobs
# =============================================================================

# Paths
DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\archive\asl_dataset\asl_dataset"
DATASET_SIZE = "small"  # "small" = original 36-class dataset, "large" = 87k-image 29-class dataset
LARGER_DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\LargerASL_Extract\asl_alphabet_train\asl_alphabet_train"
OUTPUT_DIR = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "VisionAI_Final_Autoencoder_v2"  # v2: deeper encoder, larger latent, stronger aug, more epochs

# Data
IMG_SIZE = 128
DATA_SUBSET_FRACTION = 1.0          # 0.0-1.0, fraction of data to use
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15                    # test = 1 - TRAIN_RATIO - VAL_RATIO
SEED = 42
NUM_WORKERS = 0

# Phase 1: Autoencoder
AE_BATCH_SIZE = 32
AE_EPOCHS = 60                      # was 50; give encoder more time to learn
AE_LR = 1e-3
LATENT_DIM = 192                    # was 128; more representational capacity for 36 classes

# Phase 2: Classifier
CLS_BATCH_SIZE = 32
CLS_EPOCHS = 80                     # was 60; v1 was still improving at epoch 60
CLS_LR = 1e-3
CLS_FINETUNE_LR = 2e-4             # was 1e-4; slightly warmer encoder fine-tuning
CLS_FROZEN_EPOCHS_RATIO = 0.3      # was 0.4; spend more epochs fine-tuning jointly

# Model
DROPOUT_RATE = 0.25                 # was 0.3; slightly less dropout with stronger augmentation
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 12       # was 10; more room with 80 epochs

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =============================================================================
# Logger - writes to stdout AND log file simultaneously
# =============================================================================

class DualLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# =============================================================================
# Dataset
# =============================================================================

class ASLDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset_info(root):
    """Scan dataset directory and return paths, labels, class names."""
    class_names = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths = []
    labels = []
    for cls_name in class_names:
        cls_dir = os.path.join(root, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            ):
                paths.append(fpath)
                labels.append(class_to_idx[cls_name])

    return paths, labels, class_names, class_to_idx


def stratified_split(paths, labels, train_ratio, val_ratio, seed, subset_fraction=1.0):
    """Stratified train/val/test split ensuring proportional class representation."""
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for cls_lbl in sorted(class_indices.keys()):
        idxs = class_indices[cls_lbl][:]
        rng.shuffle(idxs)

        # Apply subset fraction
        if subset_fraction < 1.0:
            n_keep = max(1, int(len(idxs) * subset_fraction))
            idxs = idxs[:n_keep]

        n = len(idxs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = n - n_train
            n_test = 0

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# =============================================================================
# Model Architecture
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # v2: added 5th block for richer features
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(384, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 384 * 8 * 8)
        self.deconvs = nn.Sequential(
            # v2: added mirror of 5th encoder block
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 384, 8, 8)
        x = self.deconvs(x)
        x = nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                       mode="bilinear", align_corners=False)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, img_size)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim, num_classes, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # lighter dropout on last hidden
            nn.Linear(192, num_classes),
        )

    def forward(self, z):
        return self.net(z)


# =============================================================================
# Utility functions
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def topk_accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy manually (no sklearn)."""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        k_actual = min(k, output.size(1))
        correct_k = correct[:k_actual].reshape(-1).float().sum(0)
        results.append(correct_k.item() / batch_size * 100.0)
    return results


def compute_per_class_metrics(all_preds, all_labels, num_classes, class_names):
    """Compute per-class precision, recall, F1 manually (no sklearn)."""
    metrics = {}
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        metrics[class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for l in all_labels if l == c),
        }
    return metrics


def build_confusion_matrix(all_preds, all_labels, num_classes):
    """Build confusion matrix manually."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1
    return cm


# =============================================================================
# Training functions
# =============================================================================

def train_autoencoder_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    criterion = nn.MSELoss()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        loss = criterion(recon, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / n_samples


@torch.no_grad()
def eval_autoencoder(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    criterion = nn.MSELoss()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        loss = criterion(recon, imgs)
        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / n_samples


def train_classifier_epoch(encoder, cls_head, loader, optimizer, device):
    encoder.train()
    cls_head.train()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        z = encoder(imgs)
        logits = cls_head(z)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)

    grad_norm = compute_grad_norm(encoder) + compute_grad_norm(cls_head)
    return total_loss / n_samples, correct / n_samples * 100.0, grad_norm


@torch.no_grad()
def eval_classifier(encoder, cls_head, loader, device):
    encoder.eval()
    cls_head.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        z = encoder(imgs)
        logits = cls_head(z)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)

    return total_loss / n_samples, correct / n_samples * 100.0


@torch.no_grad()
def full_test_evaluation(encoder, cls_head, loader, device, num_classes, class_names):
    """Full evaluation: per-class metrics, top-1/top-5, confusion matrix."""
    encoder.eval()
    cls_head.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        z = encoder(imgs)
        logits = cls_head(z)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.tensor(all_labels)

    top1, top5 = topk_accuracy(all_logits, all_labels_tensor, topk=(1, 5))
    per_class = compute_per_class_metrics(all_preds, all_labels, num_classes, class_names)
    cm = build_confusion_matrix(all_preds, all_labels, num_classes)

    return {
        "top1_acc": top1,
        "top5_acc": top5,
        "per_class": per_class,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


@torch.no_grad()
def get_latent_vectors(encoder, loader, device):
    """Extract latent vectors and labels for t-SNE."""
    encoder.eval()
    latents = []
    labels = []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        z = encoder(imgs)
        latents.append(z.cpu().numpy())
        labels.extend(lbls.tolist())
    return np.concatenate(latents, axis=0), np.array(labels)


# =============================================================================
# Plotting functions
# =============================================================================

def plot_ae_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", linewidth=2)
    ax.plot(epochs, val_losses, label="Val MSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Phase 1: Autoencoder Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reconstructions(model, loader, device, save_path, n=8):
    model.eval()
    imgs_batch, _ = next(iter(loader))
    imgs_batch = imgs_batch[:n].to(device)
    with torch.no_grad():
        recon, _ = model(imgs_batch)

    imgs_np = imgs_batch.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = np.clip(recon_np, 0, 1)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(imgs_np[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)
        axes[1, i].imshow(recon_np[i])
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)
    fig.suptitle("Autoencoder Reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cls_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Classification Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Classification Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(epochs, history["lr"], label="Learning Rate", linewidth=2, color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    fig.suptitle("Phase 2: Classification Training Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path):
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title("Normalized Confusion Matrix", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(per_class_metrics, class_names, save_path):
    recalls = [per_class_metrics[c]["recall"] * 100 for c in class_names]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(class_names)), recalls, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (Recall %)")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, recalls):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne(latents, labels, class_names, save_path):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[INFO] sklearn not installed, skipping t-SNE plot.")
        return

    print("[INFO] Computing t-SNE embedding (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(latents) - 1))
    embeddings = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(12, 10))
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", num_classes)

    for c in range(num_classes):
        mask = labels == c
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[cmap(c)], label=class_names[c], s=15, alpha=0.7)

    ax.set_title("t-SNE of Latent Space")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6,
              ncol=2, markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    total_start = time.time()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logger
    log_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_training.log")
    logger = DualLogger(log_path)
    sys.stdout = logger

    print("=" * 80)
    print(f"  VisionAI Final Autoencoder - ASL Classifier")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Print all config
    print("\n--- CONFIGURATION ---")
    config = {
        "DATASET_ROOT": DATASET_ROOT,
        "DATASET_SIZE": DATASET_SIZE,
        "LARGER_DATASET_ROOT": LARGER_DATASET_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_PREFIX": OUTPUT_PREFIX,
        "IMG_SIZE": IMG_SIZE,
        "DATA_SUBSET_FRACTION": DATA_SUBSET_FRACTION,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS,
        "AE_BATCH_SIZE": AE_BATCH_SIZE,
        "AE_EPOCHS": AE_EPOCHS,
        "AE_LR": AE_LR,
        "LATENT_DIM": LATENT_DIM,
        "CLS_BATCH_SIZE": CLS_BATCH_SIZE,
        "CLS_EPOCHS": CLS_EPOCHS,
        "CLS_LR": CLS_LR,
        "CLS_FINETUNE_LR": CLS_FINETUNE_LR,
        "CLS_FROZEN_EPOCHS_RATIO": CLS_FROZEN_EPOCHS_RATIO,
        "DROPOUT_RATE": DROPOUT_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "DEVICE": DEVICE,
    }
    for k, v in config.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    print("\n--- LOADING DATASET ---")
    active_root = LARGER_DATASET_ROOT if DATASET_SIZE == "large" else DATASET_ROOT
    paths, labels, class_names, class_to_idx = load_dataset_info(active_root)
    num_classes = len(class_names)
    print(f"  Total images found: {len(paths)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {class_names}")

    # Per-class counts
    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1
    print("\n  Per-class counts:")
    for c in range(num_classes):
        print(f"    {class_names[c]}: {class_counts[c]}")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(
        paths, labels, TRAIN_RATIO, VAL_RATIO, SEED, DATA_SUBSET_FRACTION
    )
    print(f"\n  Split sizes (subset fraction={DATA_SUBSET_FRACTION}):")
    print(f"    Train: {len(train_idx)}")
    print(f"    Val:   {len(val_idx)}")
    print(f"    Test:  {len(test_idx)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),                # was 10
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # occlusion robustness
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset_train = ASLDataset(paths, labels, transform=train_transform)
    full_dataset_eval = ASLDataset(paths, labels, transform=eval_transform)

    train_loader = DataLoader(
        Subset(full_dataset_train, train_idx),
        batch_size=AE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        Subset(full_dataset_eval, val_idx),
        batch_size=AE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        Subset(full_dataset_eval, test_idx),
        batch_size=AE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    # -------------------------------------------------------------------------
    # Phase 1: Autoencoder Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 1: AUTOENCODER TRAINING")
    print("=" * 80)

    autoencoder = Autoencoder(LATENT_DIM, IMG_SIZE).to(DEVICE)
    ae_total_params = count_parameters(autoencoder)
    print(f"  Autoencoder parameters: {ae_total_params:,}")
    print(f"    Encoder: {count_parameters(autoencoder.encoder):,}")
    print(f"    Decoder: {count_parameters(autoencoder.decoder):,}")

    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=AE_LR, weight_decay=WEIGHT_DECAY)
    ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ae_optimizer, mode="min", factor=0.5, patience=5
    )

    ae_history = {"train_mse": [], "val_mse": [], "lr": [], "time": []}
    best_ae_val_mse = float("inf")
    best_ae_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_autoencoder.pt")

    # AE CSV log
    ae_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ae_epoch_log.csv")
    ae_csv_file = open(ae_csv_path, "w", newline="", encoding="utf-8")
    ae_csv_writer = csv.writer(ae_csv_file)
    ae_csv_writer.writerow(["epoch", "train_mse", "val_mse", "lr", "time_sec"])

    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}  {'Time':>8}  {'Status'}")
    print("  " + "-" * 65)

    phase1_start = time.time()
    for epoch in range(1, AE_EPOCHS + 1):
        ep_start = time.time()
        train_mse = train_autoencoder_epoch(autoencoder, train_loader, ae_optimizer, DEVICE)
        val_mse = eval_autoencoder(autoencoder, val_loader, DEVICE)
        ae_scheduler.step(val_mse)
        ep_time = time.time() - ep_start

        current_lr = ae_optimizer.param_groups[0]["lr"]
        ae_history["train_mse"].append(train_mse)
        ae_history["val_mse"].append(val_mse)
        ae_history["lr"].append(current_lr)
        ae_history["time"].append(ep_time)

        status = ""
        if val_mse < best_ae_val_mse:
            best_ae_val_mse = val_mse
            torch.save({
                "encoder_state": autoencoder.encoder.state_dict(),
                "decoder_state": autoencoder.decoder.state_dict(),
                "epoch": epoch,
                "val_mse": val_mse,
            }, best_ae_path)
            status = " *BEST*"

        # ETA
        avg_epoch_time = (time.time() - phase1_start) / epoch
        eta_seconds = avg_epoch_time * (AE_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        print(f"  {epoch:5d}  {train_mse:10.6f}  {val_mse:10.6f}  {current_lr:10.2e}  {ep_time:7.1f}s{status}  ETA: {eta_str}")
        ae_csv_writer.writerow([epoch, f"{train_mse:.6f}", f"{val_mse:.6f}",
                                f"{current_lr:.2e}", f"{ep_time:.1f}"])
        ae_csv_file.flush()

    ae_csv_file.close()
    print(f"\n  Best AE val MSE: {best_ae_val_mse:.6f}")
    print(f"  AE model saved to: {best_ae_path}")

    # Plot AE results
    plot_ae_loss_curves(
        ae_history["train_mse"], ae_history["val_mse"],
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ae_loss_curves.png")
    )

    # Load best AE weights for reconstruction plot
    best_ae_ckpt = torch.load(best_ae_path, map_location=DEVICE, weights_only=True)
    autoencoder.encoder.load_state_dict(best_ae_ckpt["encoder_state"])
    autoencoder.decoder.load_state_dict(best_ae_ckpt["decoder_state"])
    plot_reconstructions(
        autoencoder, val_loader, DEVICE,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_ae_reconstructions.png")
    )

    # -------------------------------------------------------------------------
    # Phase 2: Classification Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 2: CLASSIFICATION TRAINING")
    print("=" * 80)

    encoder = autoencoder.encoder
    cls_head = ClassificationHead(LATENT_DIM, num_classes, DROPOUT_RATE).to(DEVICE)

    cls_head_params = count_parameters(cls_head)
    print(f"  Classification head parameters: {cls_head_params:,}")

    # Rebuild loaders with classifier batch size
    cls_train_loader = DataLoader(
        Subset(full_dataset_train, train_idx),
        batch_size=CLS_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    cls_val_loader = DataLoader(
        Subset(full_dataset_eval, val_idx),
        batch_size=CLS_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    cls_test_loader = DataLoader(
        Subset(full_dataset_eval, test_idx),
        batch_size=CLS_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    frozen_epochs = int(CLS_EPOCHS * CLS_FROZEN_EPOCHS_RATIO)
    finetune_epochs = CLS_EPOCHS - frozen_epochs
    print(f"  Frozen encoder epochs: {frozen_epochs}")
    print(f"  Joint fine-tune epochs: {finetune_epochs}")

    # Start with frozen encoder
    for param in encoder.parameters():
        param.requires_grad = False

    cls_optimizer = optim.Adam(cls_head.parameters(), lr=CLS_LR, weight_decay=WEIGHT_DECAY)
    cls_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        cls_optimizer, mode="max", factor=0.5, patience=5
    )

    cls_history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "lr": [], "grad_norm": [], "time": [], "best_val_acc": [],
    }
    best_cls_val_acc = 0.0
    epochs_since_improvement = 0
    best_cls_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_classifier.pt")

    # CLS CSV log
    cls_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cls_epoch_log.csv")
    cls_csv_file = open(cls_csv_path, "w", newline="", encoding="utf-8")
    cls_csv_writer = csv.writer(cls_csv_file)
    cls_csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                             "lr", "grad_norm", "time_sec", "best_val_acc"])

    print(f"\n  {'Ep':>4}  {'Phase':>8}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  "
          f"{'VlAcc':>7}  {'LR':>9}  {'GNorm':>8}  {'Time':>6}  {'BestAcc':>8}  {'Stale':>5}")
    print("  " + "-" * 100)

    phase2_start = time.time()
    for epoch in range(1, CLS_EPOCHS + 1):
        ep_start = time.time()

        # Switch to fine-tuning after frozen phase
        if epoch == frozen_epochs + 1 and frozen_epochs > 0:
            print("\n  >>> Unfreezing encoder for joint fine-tuning <<<\n")
            for param in encoder.parameters():
                param.requires_grad = True

            # Rebuild optimizer with differential LR
            cls_optimizer = optim.Adam([
                {"params": encoder.parameters(), "lr": CLS_FINETUNE_LR},
                {"params": cls_head.parameters(), "lr": CLS_LR},
            ], weight_decay=WEIGHT_DECAY)
            cls_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                cls_optimizer, mode="max", factor=0.5, patience=5
            )

        phase_label = "frozen" if epoch <= frozen_epochs else "finetune"

        train_loss, train_acc, grad_norm = train_classifier_epoch(
            encoder, cls_head, cls_train_loader, cls_optimizer, DEVICE
        )
        val_loss, val_acc = eval_classifier(encoder, cls_head, cls_val_loader, DEVICE)
        cls_scheduler.step(val_acc)
        ep_time = time.time() - ep_start

        current_lr = cls_optimizer.param_groups[-1]["lr"]

        # Track best
        if val_acc > best_cls_val_acc:
            best_cls_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save({
                "encoder_state": encoder.state_dict(),
                "cls_head_state": cls_head.state_dict(),
                "optimizer_state": cls_optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": config,
                "class_names": class_names,
            }, best_cls_path)
            status_mark = " *"
        else:
            epochs_since_improvement += 1
            status_mark = ""

        cls_history["train_loss"].append(train_loss)
        cls_history["train_acc"].append(train_acc)
        cls_history["val_loss"].append(val_loss)
        cls_history["val_acc"].append(val_acc)
        cls_history["lr"].append(current_lr)
        cls_history["grad_norm"].append(grad_norm)
        cls_history["time"].append(ep_time)
        cls_history["best_val_acc"].append(best_cls_val_acc)

        # ETA
        avg_epoch_time = (time.time() - phase2_start) / epoch
        eta_seconds = avg_epoch_time * (CLS_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        print(f"  {epoch:4d}  {phase_label:>8}  {train_loss:8.4f}  {train_acc:6.2f}%  "
              f"{val_loss:8.4f}  {val_acc:6.2f}%  {current_lr:9.2e}  {grad_norm:8.2f}  "
              f"{ep_time:5.1f}s  {best_cls_val_acc:7.2f}%  {epochs_since_improvement:5d}{status_mark}  ETA: {eta_str}")

        cls_csv_writer.writerow([
            epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}",
            f"{val_acc:.2f}", f"{current_lr:.2e}", f"{grad_norm:.4f}",
            f"{ep_time:.1f}", f"{best_cls_val_acc:.2f}"
        ])
        cls_csv_file.flush()

        # Early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    cls_csv_file.close()
    print(f"\n  Best classifier val accuracy: {best_cls_val_acc:.2f}%")
    print(f"  Classifier saved to: {best_cls_path}")

    # -------------------------------------------------------------------------
    # Final Evaluation on Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    # Load best classifier
    best_cls_ckpt = torch.load(best_cls_path, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(best_cls_ckpt["encoder_state"])
    cls_head.load_state_dict(best_cls_ckpt["cls_head_state"])

    test_results = full_test_evaluation(
        encoder, cls_head, cls_test_loader, DEVICE, num_classes, class_names
    )

    print(f"\n  Top-1 Accuracy: {test_results['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy: {test_results['top5_acc']:.2f}%")

    print(f"\n  {'Class':>12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
    print("  " + "-" * 55)
    macro_p, macro_r, macro_f1 = 0, 0, 0
    for c in class_names:
        m = test_results["per_class"][c]
        print(f"  {c:>12}  {m['precision']:10.4f}  {m['recall']:10.4f}  "
              f"{m['f1']:10.4f}  {m['support']:8d}")
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f1 += m["f1"]

    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes
    print("  " + "-" * 55)
    print(f"  {'MACRO AVG':>12}  {macro_p:10.4f}  {macro_r:10.4f}  {macro_f1:10.4f}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    print("\n--- GENERATING PLOTS ---")

    plot_cls_curves(
        cls_history,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cls_training_curves.png")
    )
    print("  Classification training curves saved.")

    plot_confusion_matrix(
        test_results["confusion_matrix"], class_names,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_confusion_matrix.png")
    )
    print("  Confusion matrix saved.")

    plot_per_class_accuracy(
        test_results["per_class"], class_names,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_per_class_accuracy.png")
    )
    print("  Per-class accuracy chart saved.")

    # t-SNE
    latents, tsne_labels = get_latent_vectors(encoder, cls_test_loader, DEVICE)
    plot_tsne(
        latents, tsne_labels, class_names,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_tsne.png")
    )
    print("  t-SNE plot saved (or skipped if sklearn not available).")

    # -------------------------------------------------------------------------
    # Full Results JSON
    # -------------------------------------------------------------------------
    total_time = time.time() - total_start

    results_json = {
        "config": config,
        "dataset": {
            "total_images": len(paths),
            "num_classes": num_classes,
            "class_names": class_names,
            "subset_fraction": DATA_SUBSET_FRACTION,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "phase1_autoencoder": {
            "total_params": ae_total_params,
            "best_val_mse": best_ae_val_mse,
            "epochs_trained": len(ae_history["train_mse"]),
            "history": {
                "train_mse": ae_history["train_mse"],
                "val_mse": ae_history["val_mse"],
            },
        },
        "phase2_classifier": {
            "cls_head_params": cls_head_params,
            "frozen_epochs": frozen_epochs,
            "total_epochs_trained": len(cls_history["train_loss"]),
            "best_val_acc": best_cls_val_acc,
            "history": {
                "train_loss": cls_history["train_loss"],
                "train_acc": cls_history["train_acc"],
                "val_loss": cls_history["val_loss"],
                "val_acc": cls_history["val_acc"],
            },
        },
        "test_results": {
            "top1_accuracy": test_results["top1_acc"],
            "top5_accuracy": test_results["top5_acc"],
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_class": test_results["per_class"],
        },
        "timing": {
            "total_seconds": total_time,
            "total_formatted": str(datetime.timedelta(seconds=int(total_time))),
        },
    }

    results_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Full results JSON saved to: {results_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Total time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"  Best AE val MSE:        {best_ae_val_mse:.6f}")
    print(f"  Best classifier val acc: {best_cls_val_acc:.2f}%")
    print(f"  Test top-1 accuracy:     {test_results['top1_acc']:.2f}%")
    print(f"  Test top-5 accuracy:     {test_results['top5_acc']:.2f}%")
    print(f"  Test macro F1:           {macro_f1:.4f}")
    print(f"\n  Output files in: {OUTPUT_DIR}")
    print(f"    {OUTPUT_PREFIX}_training.log")
    print(f"    {OUTPUT_PREFIX}_ae_epoch_log.csv")
    print(f"    {OUTPUT_PREFIX}_cls_epoch_log.csv")
    print(f"    {OUTPUT_PREFIX}_best_autoencoder.pt")
    print(f"    {OUTPUT_PREFIX}_best_classifier.pt")
    print(f"    {OUTPUT_PREFIX}_ae_loss_curves.png")
    print(f"    {OUTPUT_PREFIX}_ae_reconstructions.png")
    print(f"    {OUTPUT_PREFIX}_cls_training_curves.png")
    print(f"    {OUTPUT_PREFIX}_confusion_matrix.png")
    print(f"    {OUTPUT_PREFIX}_per_class_accuracy.png")
    print(f"    {OUTPUT_PREFIX}_tsne.png")
    print(f"    {OUTPUT_PREFIX}_results.json")
    print("=" * 80)

    # Restore stdout and close logger
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
