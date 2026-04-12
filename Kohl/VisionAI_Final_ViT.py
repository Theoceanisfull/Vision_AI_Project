"""
VisionAI Final ViT — Vision Transformer for ASL Sign Classification
=====================================================================
A from-scratch Vision Transformer (ViT) implementation for classifying
American Sign Language images (a-z, 0-9).

Why Vision Transformers for ASL?
  - Self-attention captures GLOBAL relationships across the entire image
    in a single layer, while CNNs are limited to local receptive fields.
  - For ASL, the overall hand pose (how fingers relate to each other
    across the full hand) matters as much as local finger detail.
  - ViTs learn which patches of the image to attend to — effectively
    learning where to look for discriminative features.
  - Provides a direct, modern comparison against our CNN baseline.

Architecture (Dosovitskiy et al., 2020):
  1. Split image into fixed-size patches (e.g., 16x16)
  2. Linearly embed each patch into a vector
  3. Add positional embeddings + a learnable [CLS] token
  4. Pass through N Transformer encoder layers (multi-head self-attention + FFN)
  5. Take the [CLS] token output for classification

Dataset: Kaggle ASL image dataset (2,515 RGB images, 400x400, 36 classes: a-z + 0-9)
"""

import time
import json
import csv
import random
import sys
import datetime
import math
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
#  CONFIGURATION
# ============================================================================

# --- Paths ---
DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\archive\asl_dataset\asl_dataset"
DATASET_SIZE = "small"  # "small" = original 36-class dataset, "large" = 87k-image 29-class dataset
LARGER_DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\LargerASL_Extract\asl_alphabet_train\asl_alphabet_train"
OUTPUT_DIR   = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "VisionAI_Final_ViT_v3"  # v3: revert to v1 architecture size, moderate regularization

# --- Data ---
IMG_SIZE              = 128       # Resize images to IMG_SIZE x IMG_SIZE
DATA_SUBSET_FRACTION  = 1.0      # 0.0–1.0; set < 1.0 for quick test runs
TRAIN_RATIO           = 0.70
VAL_RATIO             = 0.15     # test = 1 – TRAIN – VAL
SEED                  = 42
NUM_WORKERS           = 0        # 0 for Windows compatibility

# --- Training ---
BATCH_SIZE            = 32
NUM_EPOCHS            = 60
LEARNING_RATE         = 3e-4     # ViTs prefer lower LR than CNNs
WEIGHT_DECAY          = 0.01     # Higher WD helps ViTs regularize
WARMUP_EPOCHS         = 5        # Linear warmup before cosine decay
SCHEDULER_TYPE        = "cosine_warmup"   # "cosine_warmup", "cosine", or "step"
STEP_LR_STEP_SIZE     = 15
STEP_LR_GAMMA         = 0.5
LABEL_SMOOTHING       = 0.1      # Helps ViT generalization on small datasets
GRAD_CLIP_MAX_NORM    = 1.0      # Gradient clipping (0 to disable)

# --- ViT Architecture ---
PATCH_SIZE            = 16       # Each patch is PATCH_SIZE x PATCH_SIZE pixels
EMBED_DIM             = 256      # reverted to v1 size (v2's 192 was too small)
NUM_HEADS             = 8        # reverted to v1 (must divide EMBED_DIM evenly)
NUM_LAYERS            = 6        # reverted to v1 (v2's 4 was too few)
MLP_RATIO             = 4.0      # reverted to v1
DROPOUT_RATE          = 0.15     # between v1 (0.1) and v2 (0.2)
ATTN_DROPOUT          = 0.12     # between v1 (0.1) and v2 (0.15)

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 15     # more patience with 60 epochs

# --- Augmentation ---
AUGMENT_ROTATION      = 20       # was 15; more rotation variety
AUGMENT_HFLIP         = True
AUGMENT_COLOR_JITTER  = 0.25     # between v1 (0.2) and v2 (0.3)
AUGMENT_RANDOM_ERASING = 0.15    # between v1 (0.1) and v2 (0.2)
AUGMENT_MIXUP_ALPHA   = 0.0      # Mixup alpha (0 to disable; try 0.2 for regularization)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
#  LOGGER — dual output to console + file
# ============================================================================

class DualLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def log(self, msg: str = ""):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.stdout.write(line)
        self.stdout.flush()
        self.log_file.write(line)
        self.log_file.flush()

    def raw(self, msg: str = ""):
        line = f"{msg}\n"
        self.stdout.write(line)
        self.stdout.flush()
        self.log_file.write(line)
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ============================================================================
#  VISION TRANSFORMER MODEL
# ============================================================================

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional attention map output."""
    def __init__(self, embed_dim=256, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: MHSA + FFN with residual + LayerNorm."""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Image -> Patches -> Linear Embedding + [CLS] token + Positional Embedding
    -> N x Transformer Blocks -> [CLS] output -> Classification Head
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=3,
                 num_classes=36, embed_dim=256, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        attn_weights_all = []
        for blk in self.blocks:
            if return_attn:
                x, attn_w = blk(x, return_attn=True)
                attn_weights_all.append(attn_w)
            else:
                x = blk(x)

        x = self.norm(x)

        # [CLS] token output
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        if return_attn:
            return logits, attn_weights_all
        return logits


# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(dataset, train_ratio, val_ratio, seed):
    """Split into train/val/test with proportional class representation."""
    rng = random.Random(seed)
    targets = [dataset.targets[i] for i in range(len(dataset))]
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
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, logger):
    logger.log("Model Architecture:")
    logger.raw("-" * 70)
    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            logger.raw(f"  {name}: {len(module)} blocks")
            if len(module) > 0:
                logger.raw(f"    [0] {module[0]}")
        elif isinstance(module, nn.Sequential):
            for sub_name, sub_mod in module.named_children():
                logger.raw(f"  {name}.{sub_name}: {sub_mod}")
        else:
            params = sum(p.numel() for p in module.parameters())
            logger.raw(f"  {name}: {type(module).__name__} ({params:,} params)")
    total, trainable = count_parameters(model)
    logger.raw("-" * 70)
    logger.raw(f"  Total parameters:     {total:,}")
    logger.raw(f"  Trainable parameters: {trainable:,}")
    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    logger.raw(f"  Num patches:          {num_patches} ({IMG_SIZE}px / {PATCH_SIZE}px = {IMG_SIZE//PATCH_SIZE} grid)")
    logger.raw(f"  Sequence length:      {num_patches + 1} (patches + [CLS])")
    logger.raw(f"  Attention per layer:  {NUM_HEADS} heads x {EMBED_DIM // NUM_HEADS} dim/head")
    logger.raw("-" * 70)


# ============================================================================
#  METRICS — manual computation (no sklearn)
# ============================================================================

def compute_confusion_matrix(all_preds, all_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1
    return cm


def precision_recall_f1_from_cm(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[c] = prec
        recall[c] = rec
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return precision, recall, f1


def top_k_accuracy(outputs_list, labels_list, k=5):
    outputs = torch.cat(outputs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    _, topk_preds = outputs.topk(min(k, outputs.size(1)), dim=1)
    correct = topk_preds.eq(labels.unsqueeze(1).expand_as(topk_preds))
    return correct.any(dim=1).float().sum().item() / labels.size(0)


# ============================================================================
#  LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================================

class CosineWarmupScheduler:
    """Linear warmup for WARMUP_EPOCHS, then cosine decay to 0."""
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ============================================================================
#  TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def full_test_evaluation(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_outputs = [], [], []
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_outputs.append(outputs.cpu())
    return all_preds, all_labels, all_outputs, running_loss / total, correct / total


# ============================================================================
#  VISUALIZATION
# ============================================================================

def plot_training_curves(history, output_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", ms=3, label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "r-o", ms=3, label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("ViT — Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], "b-o", ms=3, label="Train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "r-o", ms=3, label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("ViT — Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [h["lr"] for h in history], "g-o", ms=3, label="LR")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("ViT — LR Schedule (Warmup + Cosine)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("ViT — Normalized Confusion Matrix")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()


def plot_per_class_accuracy(cm, class_names, output_path):
    accs = []
    for i in range(len(class_names)):
        total = cm[i].sum()
        accs.append(cm[i, i] / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if a >= 0.7 else "orange" if a >= 0.4 else "red" for a in accs]
    ax.bar(range(len(class_names)), accs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7)
    ax.set_ylabel("Accuracy"); ax.set_title("ViT — Per-Class Accuracy")
    ax.axhline(y=np.mean(accs), color="blue", linestyle="--", label=f"Mean: {np.mean(accs):.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_attention_maps(model, dataset, device, class_names, output_path, n_samples=4):
    """
    Visualize the attention maps from the last transformer layer.
    Shows which patches the model attends to for classification.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    num_patches_side = IMG_SIZE // PATCH_SIZE

    fig, axes = plt.subplots(n_samples, NUM_LAYERS + 1, figsize=(3 * (NUM_LAYERS + 1), 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        img, label = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            _, attn_weights = model(img_tensor, return_attn=True)

        # Show original image
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        axes[row][0].imshow(img_np)
        axes[row][0].set_title(f"'{class_names[label]}'", fontsize=9)
        axes[row][0].axis("off")

        # Show attention from [CLS] token to patches for each layer
        for layer_idx in range(min(NUM_LAYERS, len(axes[row]) - 1)):
            attn = attn_weights[layer_idx]  # (1, heads, N, N)
            # Average over heads, take [CLS] token's attention to patch tokens
            cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (num_patches,)
            cls_attn = cls_attn.reshape(num_patches_side, num_patches_side).cpu().numpy()

            axes[row][layer_idx + 1].imshow(cls_attn, cmap="hot", interpolation="bilinear")
            axes[row][layer_idx + 1].set_title(f"Layer {layer_idx+1}", fontsize=8)
            axes[row][layer_idx + 1].axis("off")

    plt.suptitle("ViT — [CLS] Attention Maps per Layer", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_grad_norm_history(history, output_path):
    epochs = [h["epoch"] for h in history]
    grad_norms = [h["grad_norm"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, grad_norms, "m-o", ms=3, label="Gradient Norm")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L2 Norm")
    ax.set_title("ViT — Gradient Norm Over Training")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_position_embedding_similarity(model, output_path):
    """
    Visualize the learned positional embeddings by computing their
    cosine similarity. Shows whether the model learned spatial structure.
    """
    pos_emb = model.pos_embed[0, 1:].detach().cpu()  # exclude [CLS], (num_patches, embed_dim)
    num_patches = pos_emb.shape[0]
    side = int(num_patches ** 0.5)

    # Cosine similarity between all patch positions
    pos_norm = pos_emb / pos_emb.norm(dim=-1, keepdim=True)
    sim = (pos_norm @ pos_norm.T).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full similarity matrix
    axes[0].imshow(sim, cmap="viridis")
    axes[0].set_title("Position Embedding Similarity (all pairs)")
    axes[0].set_xlabel("Patch Index"); axes[0].set_ylabel("Patch Index")

    # Similarity of each position to the center patch
    center = num_patches // 2
    center_sim = sim[center].reshape(side, side)
    axes[1].imshow(center_sim, cmap="hot", interpolation="bilinear")
    axes[1].set_title(f"Similarity to Center Patch ({center})")

    plt.suptitle("ViT — Learned Positional Embeddings", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
#  MAIN
# ============================================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = DualLogger(output_dir / f"{OUTPUT_PREFIX}.log")
    set_seed(SEED)

    # --- Log configuration ---
    logger.log("=" * 70)
    logger.log("VisionAI Final ViT — Vision Transformer for ASL Classification")
    logger.log("=" * 70)

    config = OrderedDict([
        ("DATASET_ROOT", DATASET_ROOT), ("DATASET_SIZE", DATASET_SIZE),
        ("LARGER_DATASET_ROOT", LARGER_DATASET_ROOT), ("OUTPUT_DIR", OUTPUT_DIR),
        ("OUTPUT_PREFIX", OUTPUT_PREFIX), ("IMG_SIZE", IMG_SIZE),
        ("DATA_SUBSET_FRACTION", DATA_SUBSET_FRACTION),
        ("TRAIN_RATIO", TRAIN_RATIO), ("VAL_RATIO", VAL_RATIO), ("SEED", SEED),
        ("BATCH_SIZE", BATCH_SIZE), ("NUM_EPOCHS", NUM_EPOCHS),
        ("LEARNING_RATE", LEARNING_RATE), ("WEIGHT_DECAY", WEIGHT_DECAY),
        ("WARMUP_EPOCHS", WARMUP_EPOCHS), ("SCHEDULER_TYPE", SCHEDULER_TYPE),
        ("LABEL_SMOOTHING", LABEL_SMOOTHING), ("GRAD_CLIP_MAX_NORM", GRAD_CLIP_MAX_NORM),
        ("PATCH_SIZE", PATCH_SIZE), ("EMBED_DIM", EMBED_DIM),
        ("NUM_HEADS", NUM_HEADS), ("NUM_LAYERS", NUM_LAYERS),
        ("MLP_RATIO", MLP_RATIO), ("DROPOUT_RATE", DROPOUT_RATE),
        ("ATTN_DROPOUT", ATTN_DROPOUT),
        ("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE),
        ("AUGMENT_ROTATION", AUGMENT_ROTATION), ("AUGMENT_HFLIP", AUGMENT_HFLIP),
        ("AUGMENT_COLOR_JITTER", AUGMENT_COLOR_JITTER),
        ("AUGMENT_RANDOM_ERASING", AUGMENT_RANDOM_ERASING),
        ("AUGMENT_MIXUP_ALPHA", AUGMENT_MIXUP_ALPHA),
        ("DEVICE", str(DEVICE)),
    ])
    logger.log("Configuration:")
    for k, v in config.items():
        logger.raw(f"  {k}: {v}")

    # --- Dataset ---
    logger.log("\nLoading dataset...")
    train_tf_list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if AUGMENT_HFLIP:
        train_tf_list.append(transforms.RandomHorizontalFlip())
    if AUGMENT_ROTATION > 0:
        train_tf_list.append(transforms.RandomRotation(AUGMENT_ROTATION))
    if AUGMENT_COLOR_JITTER > 0:
        train_tf_list.append(transforms.ColorJitter(
            brightness=AUGMENT_COLOR_JITTER, contrast=AUGMENT_COLOR_JITTER,
            saturation=AUGMENT_COLOR_JITTER))
    train_tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if AUGMENT_RANDOM_ERASING > 0:
        train_tf_list.append(transforms.RandomErasing(p=AUGMENT_RANDOM_ERASING))
    train_transform = transforms.Compose(train_tf_list)

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    active_root = LARGER_DATASET_ROOT if DATASET_SIZE == "large" else DATASET_ROOT
    full_dataset = datasets.ImageFolder(active_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(active_root, transform=val_transform)

    class_names = full_dataset.classes
    num_classes = len(class_names)
    logger.log(f"Found {num_classes} classes: {class_names}")
    logger.log(f"Total images: {len(full_dataset)}")

    # Class distribution
    class_counts = Counter(full_dataset.targets)
    for cls_idx in sorted(class_counts.keys()):
        logger.raw(f"  {class_names[cls_idx]}: {class_counts[cls_idx]} images")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(full_dataset, TRAIN_RATIO, VAL_RATIO, SEED)

    # Apply subset fraction
    if DATA_SUBSET_FRACTION < 1.0:
        rng = random.Random(SEED)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        train_idx = train_idx[:max(1, int(len(train_idx) * DATA_SUBSET_FRACTION))]
        val_idx = val_idx[:max(1, int(len(val_idx) * DATA_SUBSET_FRACTION))]
        test_idx = test_idx[:max(1, int(len(test_idx) * DATA_SUBSET_FRACTION))]
        logger.log(f"DATA_SUBSET_FRACTION={DATA_SUBSET_FRACTION}: using reduced splits")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(Subset(val_dataset, test_idx), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    logger.log(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # --- Model ---
    model = VisionTransformer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3,
        num_classes=num_classes, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
        dropout=DROPOUT_RATE, attn_dropout=ATTN_DROPOUT,
    ).to(DEVICE)

    logger.log("")
    model_summary(model, logger)

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "cosine_warmup":
        scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    elif SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)
    else:
        raise ValueError(f"Unknown SCHEDULER_TYPE: {SCHEDULER_TYPE}")

    # --- CSV epoch log ---
    csv_path = output_dir / f"{OUTPUT_PREFIX}_epoch_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "lr", "grad_norm", "epoch_time_s", "best_val_acc", "epochs_no_improve",
    ])

    # --- Training loop ---
    logger.log("\n" + "=" * 70)
    logger.log("TRAINING")
    logger.log("=" * 70)

    header = (f"{'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | "
              f"{'LR':>10} | {'GradNorm':>9} | {'Time':>6} | {'Best':>7} | {'NoImp':>5} | {'ETA':>12}")
    logger.raw(header)
    logger.raw("-" * len(header))

    history = []
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Warmup scheduler needs to be stepped before training
        if SCHEDULER_TYPE == "cosine_warmup":
            scheduler.step(epoch - 1)
            lr = scheduler.get_lr()
        else:
            lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, GRAD_CLIP_MAX_NORM)
        grad_norm = compute_gradient_norm(model)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        if SCHEDULER_TYPE != "cosine_warmup":
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ETA
        avg_epoch_time = (time.time() - start_time) / epoch
        eta_seconds = avg_epoch_time * (NUM_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        record = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
            "grad_norm": grad_norm, "epoch_time": elapsed,
        }
        history.append(record)

        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "config": dict(config),
                "class_names": class_names,
            }, output_dir / f"{OUTPUT_PREFIX}_best.pth")
        else:
            epochs_no_improve += 1

        logger.raw(f"{epoch:4d} | {train_loss:8.4f} | {train_acc:7.4f} | {val_loss:8.4f} | {val_acc:7.4f} | "
                    f"{lr:10.2e} | {grad_norm:9.4f} | {elapsed:5.1f}s | {best_val_acc:7.4f} | {epochs_no_improve:5d} | ETA: {eta_str}")

        csv_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.8f}",
            f"{grad_norm:.6f}", f"{elapsed:.2f}", f"{best_val_acc:.6f}", epochs_no_improve,
        ])
        csv_file.flush()

        # Early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.log(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    csv_file.close()
    total_train_time = time.time() - start_time
    logger.log(f"\nTraining complete in {total_train_time:.1f}s ({len(history)} epochs)")
    logger.log(f"Best validation accuracy: {best_val_acc:.4f}")

    # ========================================================================
    #  FINAL TEST EVALUATION
    # ========================================================================
    logger.log("\n" + "=" * 70)
    logger.log("FINAL TEST EVALUATION")
    logger.log("=" * 70)

    # Load best model
    ckpt = torch.load(output_dir / f"{OUTPUT_PREFIX}_best.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.log(f"Loaded best model from epoch {ckpt['epoch']}")

    all_preds, all_labels, all_outputs, test_loss, test_acc = full_test_evaluation(
        model, test_loader, DEVICE, num_classes)

    logger.log(f"\nTest Loss:     {test_loss:.4f}")
    logger.log(f"Test Accuracy: {test_acc:.4f} (Top-1)")

    top5_acc = top_k_accuracy(all_outputs, [torch.tensor(all_labels)], k=5)
    logger.log(f"Test Top-5:    {top5_acc:.4f}")

    # Confusion matrix & per-class metrics
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)

    logger.log(f"\n{'Class':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>8}")
    logger.raw("-" * 45)
    for i, name in enumerate(class_names):
        support = cm[i].sum()
        logger.raw(f"{name:>6} | {precision[i]:6.4f} | {recall[i]:6.4f} | {f1[i]:6.4f} | {support:>8}")

    macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()
    logger.raw("-" * 45)
    logger.raw(f"{'MACRO':>6} | {macro_p:6.4f} | {macro_r:6.4f} | {macro_f1:6.4f} | {sum(cm.sum(axis=1)):>8}")

    # ========================================================================
    #  PLOTS
    # ========================================================================
    logger.log("\nSaving plots...")

    plot_training_curves(history, output_dir / f"{OUTPUT_PREFIX}_training_curves.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_training_curves.png")

    plot_confusion_matrix(cm, class_names, output_dir / f"{OUTPUT_PREFIX}_confusion_matrix.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_confusion_matrix.png")

    plot_per_class_accuracy(cm, class_names, output_dir / f"{OUTPUT_PREFIX}_per_class_accuracy.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_per_class_accuracy.png")

    plot_grad_norm_history(history, output_dir / f"{OUTPUT_PREFIX}_grad_norm.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_grad_norm.png")

    plot_position_embedding_similarity(model, output_dir / f"{OUTPUT_PREFIX}_position_embeddings.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_position_embeddings.png")

    # Attention maps (using val dataset for clean images)
    plot_attention_maps(model, Subset(val_dataset, test_idx), DEVICE, class_names,
                        output_dir / f"{OUTPUT_PREFIX}_attention_maps.png")
    logger.raw(f"  Saved: {OUTPUT_PREFIX}_attention_maps.png")

    # ========================================================================
    #  RESULTS JSON
    # ========================================================================
    results = {
        "model": "VisionTransformer (ViT)",
        "config": dict(config),
        "dataset": {
            "total_images": len(full_dataset),
            "num_classes": num_classes,
            "class_names": class_names,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "model_info": {
            "total_params": count_parameters(model)[0],
            "trainable_params": count_parameters(model)[1],
            "num_patches": (IMG_SIZE // PATCH_SIZE) ** 2,
            "sequence_length": (IMG_SIZE // PATCH_SIZE) ** 2 + 1,
        },
        "training": {
            "epochs_run": len(history),
            "total_time_s": total_train_time,
            "best_val_acc": best_val_acc,
            "best_epoch": ckpt["epoch"],
        },
        "test_results": {
            "test_loss": test_loss,
            "test_acc_top1": test_acc,
            "test_acc_top5": top5_acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
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
    logger.close()


if __name__ == "__main__":
    main()
