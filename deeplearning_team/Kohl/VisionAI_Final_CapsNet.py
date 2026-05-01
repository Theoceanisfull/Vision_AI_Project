"""
VisionAI_Final_CapsNet.py
Capsule Network classifier for ASL sign images (a-z, 0-9).
"""

import os
import sys
import time
import json
import math
import random
import logging
import datetime
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\archive\asl_dataset\asl_dataset"
DATASET_SIZE = "small"  # "small" = original 36-class dataset, "large" = 87k-image 29-class dataset
LARGER_DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\LargerASL_Extract\asl_alphabet_train\asl_alphabet_train"
OUTPUT_DIR = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "VisionAI_Final_CapsNet"

# Data
IMG_SIZE = 64
DATA_SUBSET_FRACTION = 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
NUM_WORKERS = 0

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
SCHEDULER_TYPE = "step"          # "step" or "cosine"
STEP_LR_STEP_SIZE = 15
STEP_LR_GAMMA = 0.5

# CapsNet architecture
PRIMARY_CAPS = 32
PRIMARY_DIM = 8
CLASS_DIM = 16
ROUTING_ITERATIONS = 3
RECON_WEIGHT = 0.0005
MARGIN_M_PLUS = 0.9
MARGIN_M_MINUS = 0.1
MARGIN_LAMBDA = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10     # 0 to disable

# Augmentation
AUGMENT_ROTATION = 10
AUGMENT_HFLIP = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config_dict():
    return {
        "DATASET_ROOT": DATASET_ROOT, "DATASET_SIZE": DATASET_SIZE,
        "LARGER_DATASET_ROOT": LARGER_DATASET_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR, "OUTPUT_PREFIX": OUTPUT_PREFIX,
        "IMG_SIZE": IMG_SIZE, "DATA_SUBSET_FRACTION": DATA_SUBSET_FRACTION,
        "TRAIN_RATIO": TRAIN_RATIO, "VAL_RATIO": VAL_RATIO, "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS, "BATCH_SIZE": BATCH_SIZE, "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE, "WEIGHT_DECAY": WEIGHT_DECAY,
        "SCHEDULER_TYPE": SCHEDULER_TYPE, "STEP_LR_STEP_SIZE": STEP_LR_STEP_SIZE,
        "STEP_LR_GAMMA": STEP_LR_GAMMA, "PRIMARY_CAPS": PRIMARY_CAPS,
        "PRIMARY_DIM": PRIMARY_DIM, "CLASS_DIM": CLASS_DIM,
        "ROUTING_ITERATIONS": ROUTING_ITERATIONS, "RECON_WEIGHT": RECON_WEIGHT,
        "MARGIN_M_PLUS": MARGIN_M_PLUS, "MARGIN_M_MINUS": MARGIN_M_MINUS,
        "MARGIN_LAMBDA": MARGIN_LAMBDA, "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "AUGMENT_ROTATION": AUGMENT_ROTATION, "AUGMENT_HFLIP": AUGMENT_HFLIP,
        "DEVICE": DEVICE,
    }

def setup_logging(log_path):
    logger = logging.getLogger("CapsNet")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ============================================================================
# DATASET
# ============================================================================

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
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        folder = os.path.join(root, c)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                paths.append(fpath)
                labels.append(class_to_idx[c])
    return paths, labels, classes, class_to_idx


def stratified_split(paths, labels, train_ratio, val_ratio, seed, subset_fraction=1.0):
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for lbl in sorted(class_indices.keys()):
        idxs = class_indices[lbl][:]
        rng.shuffle(idxs)
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

    return train_idx, val_idx, test_idx

# ============================================================================
# CAPSULE NETWORK
# ============================================================================

def squash(tensor, dim=-1):
    sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm) / (torch.sqrt(sq_norm) + 1e-8)
    return scale * tensor


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size=9, stride=2, padding=0):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, capsule_dim, kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        outputs = [cap(x) for cap in self.capsules]             # list of (B, capsule_dim, H, W)
        outputs = torch.stack(outputs, dim=1)                    # (B, num_caps, capsule_dim, H, W)
        B, C, D, H, W = outputs.shape
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, H, W, D)
        outputs = outputs.view(B, C * H * W, D)                 # (B, N, D)
        return squash(outputs)


class ClassCapsules(nn.Module):
    def __init__(self, num_input_caps, input_dim, num_classes, output_dim, routing_iterations):
        super().__init__()
        self.num_input_caps = num_input_caps
        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(1, num_input_caps, num_classes, output_dim, input_dim) * 0.01)

    def forward(self, x):
        # x: (B, num_input, input_dim)
        B = x.size(0)
        # (B, num_input, 1, input_dim, 1)
        x_hat = x[:, :, None, :, None]
        # W: (1, num_input, num_classes, output_dim, input_dim)
        # u_hat: (B, num_input, num_classes, output_dim, 1)
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x_hat.expand(-1, -1, self.num_classes, -1, -1))
        u_hat = u_hat.squeeze(-1)  # (B, num_input, num_classes, output_dim)

        b = torch.zeros(B, self.num_input_caps, self.num_classes, device=x.device)

        for i in range(self.routing_iterations):
            c = F.softmax(b, dim=2)                             # (B, num_input, num_classes)
            # s: (B, num_classes, output_dim)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s, dim=-1)                                # (B, num_classes, output_dim)

            if i < self.routing_iterations - 1:
                # agreement: (B, num_input, num_classes)
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v  # (B, num_classes, output_dim)


class ReconstructionDecoder(nn.Module):
    def __init__(self, num_classes, capsule_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(capsule_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_size * img_size * 3),
            nn.Tanh(),
        )

    def forward(self, class_capsules, targets):
        # class_capsules: (B, num_classes, capsule_dim)
        # Mask: keep only the target class capsule
        B, C, D = class_capsules.shape
        mask = torch.zeros(B, C, device=class_capsules.device)
        mask.scatter_(1, targets.view(-1, 1), 1.0)
        masked = class_capsules * mask.unsqueeze(-1)             # (B, C, D)
        masked = masked.view(B, -1)                              # (B, C*D)
        recon = self.fc(masked)                                  # (B, img_size*img_size*3)
        return recon.view(B, 3, self.img_size, self.img_size)


class CapsuleNet(nn.Module):
    def __init__(self, num_classes, img_size, primary_caps, primary_dim, class_dim, routing_iters):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # Conv backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=9, padding=0)
        self.bn3 = nn.BatchNorm2d(256)

        # Determine spatial dims after conv backbone
        dummy = torch.zeros(1, 3, img_size, img_size)
        dummy = F.relu(self.bn1(self.conv1(dummy)))
        dummy = F.relu(self.bn2(self.conv2(dummy)))
        dummy = F.relu(self.bn3(self.conv3(dummy)))
        conv_out_size = dummy.shape  # (1, 256, H, W)

        self.primary_caps = PrimaryCapsules(256, primary_caps, primary_dim, kernel_size=9, stride=2, padding=0)

        # Determine num primary capsule outputs
        dummy_pc = self.primary_caps(dummy)
        num_primary_out = dummy_pc.shape[1]

        self.class_caps = ClassCapsules(num_primary_out, primary_dim, num_classes, class_dim, routing_iters)
        self.decoder = ReconstructionDecoder(num_classes, class_dim, img_size)

    def forward(self, x, targets=None):
        # Conv backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        primary = self.primary_caps(h)
        class_capsules = self.class_caps(primary)   # (B, num_classes, class_dim)

        lengths = torch.sqrt((class_capsules ** 2).sum(dim=-1) + 1e-8)  # (B, num_classes)

        if targets is None:
            targets = lengths.argmax(dim=1)

        recon = self.decoder(class_capsules, targets)
        return lengths, recon, class_capsules


class CapsuleLoss(nn.Module):
    def __init__(self, m_plus, m_minus, lam, recon_weight):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lam = lam
        self.recon_weight = recon_weight

    def forward(self, lengths, recon, images, targets, num_classes):
        # Margin loss
        one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        left = F.relu(self.m_plus - lengths) ** 2
        right = F.relu(lengths - self.m_minus) ** 2
        margin_loss = (one_hot * left + self.lam * (1 - one_hot) * right).sum(dim=1).mean()

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, images)

        total_loss = margin_loss + self.recon_weight * recon_loss
        return total_loss, margin_loss, recon_loss

# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("conv", "bn")))
    primary_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("primary"))
    class_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("class_caps"))
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("decoder"))
    return total, conv_params, primary_params, class_params, decoder_params

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss, total_margin, total_recon = 0.0, 0.0, 0.0
    correct, total = 0, 0
    grad_norm = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        lengths, recon, _ = model(images, labels)
        loss, m_loss, r_loss = criterion(lengths, recon, images, labels, num_classes)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_margin += m_loss.item() * images.size(0)
        total_recon += r_loss.item() * images.size(0)
        preds = lengths.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    n = total if total > 0 else 1
    return total_loss / n, total_margin / n, total_recon / n, correct / n, grad_norm


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_margin, total_recon = 0.0, 0.0, 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            lengths, recon, _ = model(images, labels)
            loss, m_loss, r_loss = criterion(lengths, recon, images, labels, num_classes)
            total_loss += loss.item() * images.size(0)
            total_margin += m_loss.item() * images.size(0)
            total_recon += r_loss.item() * images.size(0)
            preds = lengths.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    n = total if total > 0 else 1
    return total_loss / n, total_margin / n, total_recon / n, correct / n


def full_evaluation(model, loader, device, num_classes, class_names):
    model.eval()
    all_preds, all_labels = [], []
    all_images, all_recons, all_caps = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            lengths, recon, class_capsules = model(images)
            preds = lengths.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.append(images.cpu())
            all_recons.append(recon.cpu())
            all_caps.append(class_capsules.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_images = torch.cat(all_images, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    all_caps = torch.cat(all_caps, dim=0)

    # Accuracy
    top1_acc = (all_preds == all_labels).sum() / len(all_labels)

    # Top-5 accuracy
    model.eval()
    top5_correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            lengths, _, _ = model(images)
            _, top5_preds = lengths.topk(min(5, num_classes), dim=1)
            for i in range(labels.size(0)):
                if labels[i].item() in top5_preds[i].cpu().numpy():
                    top5_correct += 1
    top5_acc = top5_correct / len(all_labels)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    # Per-class precision, recall, F1
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c][c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0

    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        total_c = cm[c].sum()
        per_class_acc[c] = cm[c][c] / total_c if total_c > 0 else 0.0

    return {
        "top1_acc": float(top1_acc),
        "top5_acc": float(top5_acc),
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_acc": per_class_acc,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_images": all_images,
        "all_recons": all_recons,
        "all_caps": all_caps,
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(history, output_dir, prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_margin_loss"], label="Train")
    axes[1].plot(epochs, history["val_margin_loss"], label="Val")
    axes[1].set_title("Margin Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["train_recon_loss"], label="Train")
    axes[2].plot(epochs, history["val_recon_loss"], label="Val")
    axes[2].set_title("Reconstruction Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_loss_curves.png"), dpi=150)
    plt.close()

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], label="Val Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_accuracy_curves.png"), dpi=150)
    plt.close()

    # LR schedule
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["lr"], label="Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_lr_schedule.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir, prefix):
    num_classes = len(class_names)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.4), max(10, num_classes * 0.4)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()


def plot_per_class_accuracy(per_class_acc, class_names, output_dir, prefix):
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.4), 6))
    bars = ax.bar(range(len(class_names)), per_class_acc, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_per_class_accuracy.png"), dpi=150)
    plt.close()


def plot_reconstructions(images, recons, labels, class_names, output_dir, prefix, n=8):
    n = min(n, images.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        rec = recons[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        rec = np.clip(rec, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(class_names[labels[i]], fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec)
        axes[1, i].set_title("Recon", fontsize=8)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_reconstructions.png"), dpi=150)
    plt.close()


def plot_capsule_activation_heatmap(all_caps, all_labels, all_preds, class_names, output_dir, prefix):
    num_classes = len(class_names)
    # Compute mean capsule norms per true class
    cap_norms = torch.sqrt((all_caps ** 2).sum(dim=-1) + 1e-8).numpy()  # (N, num_classes)

    # True class vs capsule activation
    true_heatmap = np.zeros((num_classes, num_classes))
    true_counts = np.zeros(num_classes)
    for i in range(len(all_labels)):
        true_heatmap[all_labels[i]] += cap_norms[i]
        true_counts[all_labels[i]] += 1
    for c in range(num_classes):
        if true_counts[c] > 0:
            true_heatmap[c] /= true_counts[c]

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    im0 = axes[0].imshow(true_heatmap, aspect="auto", cmap="viridis")
    axes[0].set_title("Mean Capsule Norms (by True Class)")
    axes[0].set_xlabel("Capsule (Class)")
    axes[0].set_ylabel("True Class")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=90, fontsize=6)
    axes[0].set_yticks(range(num_classes))
    axes[0].set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im0, ax=axes[0])

    # Predicted class vs capsule activation
    pred_heatmap = np.zeros((num_classes, num_classes))
    pred_counts = np.zeros(num_classes)
    for i in range(len(all_preds)):
        pred_heatmap[all_preds[i]] += cap_norms[i]
        pred_counts[all_preds[i]] += 1
    for c in range(num_classes):
        if pred_counts[c] > 0:
            pred_heatmap[c] /= pred_counts[c]

    im1 = axes[1].imshow(pred_heatmap, aspect="auto", cmap="viridis")
    axes[1].set_title("Mean Capsule Norms (by Predicted Class)")
    axes[1].set_xlabel("Capsule (Class)")
    axes[1].set_ylabel("Predicted Class")
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=90, fontsize=6)
    axes[1].set_yticks(range(num_classes))
    axes[1].set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_capsule_activation_heatmap.png"), dpi=150)
    plt.close()


def plot_capsule_dimension_perturbation(model, test_loader, class_names, device, output_dir, prefix,
                                         class_dim=16, num_perturbations=8):
    model.eval()
    # Get one sample from each of a few classes
    images_batch, labels_batch = next(iter(test_loader))
    images_batch = images_batch.to(device)
    labels_batch = labels_batch.to(device)

    with torch.no_grad():
        lengths, recon, class_capsules = model(images_batch, labels_batch)

    # Pick first sample
    sample_idx = 0
    sample_img = images_batch[sample_idx:sample_idx + 1]
    sample_label = labels_batch[sample_idx:sample_idx + 1]
    sample_caps = class_capsules[sample_idx:sample_idx + 1]  # (1, num_classes, class_dim)

    target_class = sample_label.item()
    target_capsule = sample_caps[0, target_class].clone()  # (class_dim,)

    dims_to_perturb = min(num_perturbations, class_dim)
    perturbation_range = np.linspace(-0.25, 0.25, 7)

    fig, axes = plt.subplots(dims_to_perturb, len(perturbation_range), figsize=(len(perturbation_range) * 1.5, dims_to_perturb * 1.5))

    for d in range(dims_to_perturb):
        for pi, delta in enumerate(perturbation_range):
            perturbed = target_capsule.clone()
            perturbed[d] += delta

            # Reconstruct with the decoder
            caps_input = sample_caps.clone()
            caps_input[0, target_class] = perturbed
            with torch.no_grad():
                rec = model.decoder(caps_input, sample_label)
            rec_img = rec[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
            rec_img = np.clip(rec_img, 0, 1)

            ax = axes[d, pi] if dims_to_perturb > 1 else axes[pi]
            ax.imshow(rec_img)
            ax.axis("off")
            if d == 0:
                ax.set_title(f"{delta:+.2f}", fontsize=7)
            if pi == 0:
                ax.set_ylabel(f"Dim {d}", fontsize=7)

    plt.suptitle(f"Capsule Dimension Perturbation (class: {class_names[target_class]})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_capsule_perturbation.png"), dpi=150)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f"{OUTPUT_PREFIX}.log")
    logger = setup_logging(log_path)
    logger.info("=" * 70)
    logger.info("CapsNet ASL Training Script")
    logger.info("=" * 70)

    # Log config
    config = get_config_dict()
    logger.info("CONFIGURATION:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Load dataset
    active_root = LARGER_DATASET_ROOT if DATASET_SIZE == "large" else DATASET_ROOT
    logger.info("-" * 50)
    logger.info(f"Loading dataset (DATASET_SIZE={DATASET_SIZE!r}, root={active_root})...")
    paths, labels, class_names, class_to_idx = load_dataset_info(active_root)
    num_classes = len(class_names)
    logger.info(f"  Total images: {len(paths)}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Classes: {class_names}")
    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1
    for c in range(num_classes):
        logger.info(f"    {class_names[c]}: {class_counts[c]} images")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(paths, labels, TRAIN_RATIO, VAL_RATIO, SEED, DATA_SUBSET_FRACTION)
    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Transforms
    train_transforms_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if AUGMENT_ROTATION > 0:
        train_transforms_list.append(transforms.RandomRotation(AUGMENT_ROTATION))
    if AUGMENT_HFLIP:
        train_transforms_list.append(transforms.RandomHorizontalFlip())
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_transform = transforms.Compose(train_transforms_list)

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create datasets
    full_dataset_train = ASLDataset(paths, labels, transform=train_transform)
    full_dataset_eval = ASLDataset(paths, labels, transform=val_transform)

    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_eval, val_idx)
    test_dataset = Subset(full_dataset_eval, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Build model
    logger.info("-" * 50)
    logger.info("Building CapsuleNet...")
    device = torch.device(DEVICE)
    model = CapsuleNet(num_classes, IMG_SIZE, PRIMARY_CAPS, PRIMARY_DIM, CLASS_DIM, ROUTING_ITERATIONS).to(device)
    total_p, conv_p, primary_p, class_p, decoder_p = count_parameters(model)
    logger.info(f"  Total parameters:     {total_p:,}")
    logger.info(f"  Conv backbone:        {conv_p:,}")
    logger.info(f"  Primary capsules:     {primary_p:,}")
    logger.info(f"  Class capsules:       {class_p:,}")
    logger.info(f"  Reconstruction dec:   {decoder_p:,}")

    criterion = CapsuleLoss(MARGIN_M_PLUS, MARGIN_M_MINUS, MARGIN_LAMBDA, RECON_WEIGHT)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    else:
        scheduler = None

    # CSV logger
    csv_path = os.path.join(run_dir, f"{OUTPUT_PREFIX}_epoch_log.csv")
    csv_fields = ["epoch", "train_loss", "train_margin_loss", "train_recon_loss", "train_acc",
                  "val_loss", "val_margin_loss", "val_recon_loss", "val_acc", "lr", "grad_norm", "epoch_time"]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # History
    history = {k: [] for k in ["train_loss", "train_margin_loss", "train_recon_loss", "train_acc",
                                "val_loss", "val_margin_loss", "val_recon_loss", "val_acc", "lr", "grad_norm"]}

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(run_dir, f"{OUTPUT_PREFIX}_best_model.pth")

    # Training loop
    logger.info("-" * 50)
    logger.info("Starting training...")
    training_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_margin, tr_recon, tr_acc, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes)

        vl_loss, vl_margin, vl_recon, vl_acc = evaluate(
            model, val_loader, criterion, device, num_classes)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0

        # ETA
        avg_epoch_time = (time.time() - training_start) / epoch
        eta_seconds = avg_epoch_time * (NUM_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        # Track best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc,
                "config": config,
                "class_names": class_names,
            }, best_model_path)
            logger.info(f"  ** New best model saved (val_acc={vl_acc:.4f}) **")
        else:
            epochs_no_improve += 1

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"TrLoss={tr_loss:.4f} (M={tr_margin:.4f} R={tr_recon:.4f}) | "
            f"VlLoss={vl_loss:.4f} (M={vl_margin:.4f} R={vl_recon:.4f}) | "
            f"TrAcc={tr_acc:.4f} VlAcc={vl_acc:.4f} | "
            f"LR={current_lr:.6f} GradNorm={grad_norm:.4f} | "
            f"Time={epoch_time:.1f}s | ETA={eta_str} | Best={best_val_acc:.4f}@E{best_epoch} | "
            f"NoImprove={epochs_no_improve}"
        )

        # History
        history["train_loss"].append(tr_loss)
        history["train_margin_loss"].append(tr_margin)
        history["train_recon_loss"].append(tr_recon)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_margin_loss"].append(vl_margin)
        history["val_recon_loss"].append(vl_recon)
        history["val_acc"].append(vl_acc)
        history["lr"].append(current_lr)
        history["grad_norm"].append(grad_norm)

        # CSV
        csv_writer.writerow({
            "epoch": epoch, "train_loss": f"{tr_loss:.6f}", "train_margin_loss": f"{tr_margin:.6f}",
            "train_recon_loss": f"{tr_recon:.6f}", "train_acc": f"{tr_acc:.6f}",
            "val_loss": f"{vl_loss:.6f}", "val_margin_loss": f"{vl_margin:.6f}",
            "val_recon_loss": f"{vl_recon:.6f}", "val_acc": f"{vl_acc:.6f}",
            "lr": f"{current_lr:.8f}", "grad_norm": f"{grad_norm:.6f}", "epoch_time": f"{epoch_time:.2f}",
        })
        csv_file.flush()

        # Early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs (patience={EARLY_STOPPING_PATIENCE}).")
            break

    csv_file.close()
    logger.info("Training complete.")

    # ---------------------------------------------------------------
    # FINAL EVALUATION on test set
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Final evaluation on TEST set (loading best model)...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = full_evaluation(model, test_loader, device, num_classes, class_names)

    logger.info(f"  Top-1 Accuracy: {results['top1_acc']:.4f}")
    logger.info(f"  Top-5 Accuracy: {results['top5_acc']:.4f}")
    logger.info("-" * 50)
    logger.info(f"  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8}")
    logger.info("-" * 50)
    for c in range(num_classes):
        logger.info(
            f"  {class_names[c]:<12} {results['precision'][c]:>8.4f} "
            f"{results['recall'][c]:>8.4f} {results['f1'][c]:>8.4f} "
            f"{results['per_class_acc'][c]:>8.4f}"
        )
    macro_prec = results["precision"].mean()
    macro_rec = results["recall"].mean()
    macro_f1 = results["f1"].mean()
    logger.info("-" * 50)
    logger.info(f"  {'MACRO AVG':<12} {macro_prec:>8.4f} {macro_rec:>8.4f} {macro_f1:>8.4f}")

    # ---------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------
    logger.info("Generating plots...")
    plot_training_curves(history, run_dir, OUTPUT_PREFIX)
    plot_confusion_matrix(results["confusion_matrix"], class_names, run_dir, OUTPUT_PREFIX)
    plot_per_class_accuracy(results["per_class_acc"], class_names, run_dir, OUTPUT_PREFIX)
    plot_reconstructions(results["all_images"], results["all_recons"], results["all_labels"],
                         class_names, run_dir, OUTPUT_PREFIX)
    plot_capsule_activation_heatmap(results["all_caps"], results["all_labels"], results["all_preds"],
                                    class_names, run_dir, OUTPUT_PREFIX)
    plot_capsule_dimension_perturbation(model, test_loader, class_names, device, run_dir, OUTPUT_PREFIX,
                                         class_dim=CLASS_DIM, num_perturbations=min(8, CLASS_DIM))
    logger.info("Plots saved.")

    # ---------------------------------------------------------------
    # RESULTS JSON
    # ---------------------------------------------------------------
    results_json = {
        "config": config,
        "class_names": class_names,
        "num_classes": num_classes,
        "total_parameters": total_p,
        "component_parameters": {
            "conv_backbone": conv_p,
            "primary_capsules": primary_p,
            "class_capsules": class_p,
            "reconstruction_decoder": decoder_p,
        },
        "dataset": {
            "total_images": len(paths),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "total_epochs_run": len(history["train_loss"]),
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
        },
        "test_results": {
            "top1_accuracy": results["top1_acc"],
            "top5_accuracy": results["top5_acc"],
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "per_class": {
                class_names[c]: {
                    "precision": float(results["precision"][c]),
                    "recall": float(results["recall"][c]),
                    "f1": float(results["f1"][c]),
                    "accuracy": float(results["per_class_acc"][c]),
                } for c in range(num_classes)
            },
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    json_path = os.path.join(run_dir, f"{OUTPUT_PREFIX}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON saved to {json_path}")

    logger.info("=" * 70)
    logger.info("All outputs saved to: " + run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
