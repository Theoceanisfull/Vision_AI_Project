"""
VisionAI Final CNN — ASL Sign Language Classifier
===================================================
A production-quality 4-block CNN training pipeline for classifying
American Sign Language images (a-z, 0-9) with comprehensive logging,
evaluation, and visualization.
"""

import time
import json
import csv
import random
import sys
import os
import datetime
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
OUTPUT_DIR = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "visionai_cnn_v3"  # v3: restore 30 epochs, patience 12, rotation 20

# --- Data ---
IMG_SIZE = 128                 # resize images to IMG_SIZE x IMG_SIZE
DATA_SUBSET_FRACTION = 1.0    # 0.0-1.0; set < 1.0 for quick test runs
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15              # test = 1 - TRAIN_RATIO - VAL_RATIO
SEED = 42
NUM_WORKERS = 2

# --- Training ---
BATCH_SIZE = 32
NUM_EPOCHS = 30                  # restored from v1; v2's 50 epochs hurt via cosine LR curve
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = "cosine"     # "cosine" or "step"
STEP_LR_STEP_SIZE = 10
STEP_LR_GAMMA = 0.1

# --- Model ---
DROPOUT_RATE = 0.5
HIDDEN_DIM = 512

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 12  # set to 0 to disable

# --- Augmentation ---
AUGMENT_ROTATION = 20         # degrees (v3: +5 from v2's 15)
AUGMENT_HFLIP = True
AUGMENT_COLOR_JITTER = 0.2    # brightness/contrast/saturation jitter

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

    def close(self):
        self.log_file.close()


# ============================================================================
#  MODEL DEFINITION — 4-Block CNN
# ============================================================================

class ASLCNN(nn.Module):
    """
    4-block CNN: each block is Conv-BN-ReLU-Conv-BN-ReLU-MaxPool.
    Channels: 32 -> 64 -> 128 -> 256
    Followed by AdaptiveAvgPool, FC(hidden_dim), Dropout, FC(num_classes).
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.5,
                 hidden_dim: int = 512):
        super().__init__()

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


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
    """
    Split a dataset into train/val/test using stratified sampling.
    Returns three lists of indices.
    """
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
        # ensure at least 1 in test
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


def model_summary(model, logger):
    """Print a concise architecture summary."""
    logger.log("Model Architecture:")
    logger.log("-" * 60)
    for name, module in model.named_children():
        logger.log(f"  {name}:")
        if hasattr(module, '__iter__'):
            for sub_name, sub_mod in module.named_children():
                logger.log(f"    [{sub_name}] {sub_mod}")
        else:
            logger.log(f"    {module}")
    total, trainable = count_parameters(model)
    logger.log("-" * 60)
    logger.log(f"  Total parameters:     {total:,}")
    logger.log(f"  Trainable parameters: {trainable:,}")
    logger.log("-" * 60)


# ============================================================================
#  METRICS — manual computation (no sklearn)
# ============================================================================

def compute_confusion_matrix(all_preds, all_labels, num_classes):
    """Build a num_classes x num_classes confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1
    return cm


def precision_recall_f1_from_cm(cm):
    """
    Compute per-class precision, recall, F1 from a confusion matrix.
    Returns three numpy arrays of shape (num_classes,).
    """
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
    """Compute top-k accuracy from collected logits and labels."""
    outputs = torch.cat(outputs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    _, topk_preds = outputs.topk(k, dim=1)
    correct = topk_preds.eq(labels.unsqueeze(1).expand_as(topk_preds))
    return correct.any(dim=1).float().sum().item() / labels.size(0)


# ============================================================================
#  TRAINING & EVALUATION LOOPS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def full_test_evaluation(model, loader, device, num_classes):
    """
    Run full evaluation on test set.
    Returns: all_preds, all_labels, all_outputs (logits), test_loss, test_acc.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
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

    test_loss = running_loss / total
    test_acc = correct / total
    return all_preds, all_labels, all_outputs, test_loss, test_acc


# ============================================================================
#  PLOTTING FUNCTIONS
# ============================================================================

def plot_training_curves(history, output_path):
    """Plot loss, accuracy (train vs val), and LR overlay."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, val_acc, "r-o", markersize=3, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, lrs, "g-o", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot normalized confusion matrix as a heatmap."""
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
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_accuracy(cm, class_names, output_path):
    """Bar chart of per-class accuracy."""
    per_class_acc = []
    for c in range(len(class_names)):
        total = cm[c].sum()
        acc = cm[c, c] / total if total > 0 else 0.0
        per_class_acc.append(acc)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(class_names))
    colors = ["#2196F3" if a >= 0.5 else "#f44336" for a in per_class_acc]
    ax.bar(x, per_class_acc, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(per_class_acc), color="orange", linestyle="--",
               label=f"Mean: {np.mean(per_class_acc):.3f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lr_schedule(history, output_path):
    """Standalone learning rate schedule plot."""
    epochs = [h["epoch"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, "g-o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
#  MAIN TRAINING PIPELINE
# ============================================================================

def main():
    start_wall = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Create output directory ---
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    def out_path(suffix):
        return output_dir / f"{OUTPUT_PREFIX}_{suffix}"

    # --- Logger ---
    logger = DualLogger(out_path("training.log"))
    logger.log("=" * 70)
    logger.log("  VisionAI Final CNN — ASL Classifier Training")
    logger.log(f"  Started: {timestamp}")
    logger.log("=" * 70)

    # --- Log configuration ---
    config = OrderedDict([
        ("DATASET_ROOT", DATASET_ROOT),
        ("DATASET_SIZE", DATASET_SIZE),
        ("LARGER_DATASET_ROOT", LARGER_DATASET_ROOT),
        ("OUTPUT_DIR", OUTPUT_DIR),
        ("OUTPUT_PREFIX", OUTPUT_PREFIX),
        ("IMG_SIZE", IMG_SIZE),
        ("DATA_SUBSET_FRACTION", DATA_SUBSET_FRACTION),
        ("TRAIN_RATIO", TRAIN_RATIO),
        ("VAL_RATIO", VAL_RATIO),
        ("TEST_RATIO", round(1.0 - TRAIN_RATIO - VAL_RATIO, 4)),
        ("SEED", SEED),
        ("NUM_WORKERS", NUM_WORKERS),
        ("BATCH_SIZE", BATCH_SIZE),
        ("NUM_EPOCHS", NUM_EPOCHS),
        ("LEARNING_RATE", LEARNING_RATE),
        ("WEIGHT_DECAY", WEIGHT_DECAY),
        ("SCHEDULER_TYPE", SCHEDULER_TYPE),
        ("STEP_LR_STEP_SIZE", STEP_LR_STEP_SIZE),
        ("STEP_LR_GAMMA", STEP_LR_GAMMA),
        ("DROPOUT_RATE", DROPOUT_RATE),
        ("HIDDEN_DIM", HIDDEN_DIM),
        ("EARLY_STOPPING_PATIENCE", EARLY_STOPPING_PATIENCE),
        ("AUGMENT_ROTATION", AUGMENT_ROTATION),
        ("AUGMENT_HFLIP", AUGMENT_HFLIP),
        ("AUGMENT_COLOR_JITTER", AUGMENT_COLOR_JITTER),
        ("DEVICE", str(DEVICE)),
    ])

    logger.log("\n--- Configuration ---")
    for k, v in config.items():
        logger.log(f"  {k}: {v}")

    # --- Seed ---
    set_seed(SEED)

    # ----------------------------------------------------------------
    #  DATA LOADING & STRATIFIED SPLIT
    # ----------------------------------------------------------------
    logger.log("\n--- Data Loading ---")

    active_root = LARGER_DATASET_ROOT if DATASET_SIZE == "large" else DATASET_ROOT

    # Build a simple transform for loading (resize only); augmentation applied later
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(active_root, transform=base_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    logger.log(f"  Dataset root: {active_root}")
    logger.log(f"  Total images found: {len(full_dataset)}")
    logger.log(f"  Number of classes: {num_classes}")
    logger.log(f"  Classes: {class_names}")

    # Per-class counts
    label_counts = Counter(full_dataset.targets)
    logger.log("\n  Per-class sample counts:")
    for cls_idx in sorted(label_counts.keys()):
        logger.log(f"    {class_names[cls_idx]:>5s}: {label_counts[cls_idx]}")

    # Subset if requested
    if DATA_SUBSET_FRACTION < 1.0:
        rng = random.Random(SEED)
        all_indices = list(range(len(full_dataset)))
        rng.shuffle(all_indices)
        n_keep = max(num_classes, int(len(full_dataset) * DATA_SUBSET_FRACTION))
        # Stratified subset
        class_indices = {}
        for idx in all_indices:
            lbl = full_dataset.targets[idx]
            class_indices.setdefault(lbl, []).append(idx)
        subset_indices = []
        for cls in sorted(class_indices.keys()):
            idxs = class_indices[cls]
            n_cls = max(3, int(len(idxs) * DATA_SUBSET_FRACTION))
            subset_indices.extend(idxs[:n_cls])
        full_dataset = Subset(full_dataset, subset_indices)
        # Rebuild targets for subset
        full_dataset.targets = [full_dataset.dataset.targets[i]
                                for i in subset_indices]
        logger.log(f"\n  [DATA_SUBSET_FRACTION={DATA_SUBSET_FRACTION}] "
                    f"Using {len(full_dataset)} of original samples.")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, TRAIN_RATIO, VAL_RATIO, SEED
    )

    logger.log(f"\n  Stratified split:")
    logger.log(f"    Train: {len(train_idx)} samples")
    logger.log(f"    Val:   {len(val_idx)} samples")
    logger.log(f"    Test:  {len(test_idx)} samples")

    # Augmented transforms for training
    train_transform_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if AUGMENT_ROTATION > 0:
        train_transform_list.append(
            transforms.RandomRotation(AUGMENT_ROTATION))
    if AUGMENT_HFLIP:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if AUGMENT_COLOR_JITTER > 0:
        train_transform_list.append(
            transforms.ColorJitter(
                brightness=AUGMENT_COLOR_JITTER,
                contrast=AUGMENT_COLOR_JITTER,
                saturation=AUGMENT_COLOR_JITTER,
            ))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose(train_transform_list)

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Re-create dataset objects with proper transforms
    # We need separate ImageFolder instances with different transforms
    train_dataset_full = datasets.ImageFolder(active_root,
                                              transform=train_transform)
    eval_dataset_full = datasets.ImageFolder(active_root,
                                             transform=eval_transform)

    # Handle subset indices mapping
    if DATA_SUBSET_FRACTION < 1.0:
        # subset_indices maps our local idx -> original dataset idx
        # train_idx/val_idx/test_idx are local indices into subset
        # We need to map back to original indices
        train_orig_idx = [subset_indices[i] for i in train_idx]
        val_orig_idx = [subset_indices[i] for i in val_idx]
        test_orig_idx = [subset_indices[i] for i in test_idx]
    else:
        train_orig_idx = train_idx
        val_orig_idx = val_idx
        test_orig_idx = test_idx

    train_set = Subset(train_dataset_full, train_orig_idx)
    val_set = Subset(eval_dataset_full, val_orig_idx)
    test_set = Subset(eval_dataset_full, test_orig_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # ----------------------------------------------------------------
    #  MODEL, OPTIMIZER, SCHEDULER, CRITERION
    # ----------------------------------------------------------------
    logger.log("\n--- Model Setup ---")

    model = ASLCNN(num_classes=num_classes, dropout_rate=DROPOUT_RATE,
                   hidden_dim=HIDDEN_DIM).to(DEVICE)
    model_summary(model, logger)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS)
        logger.log(f"  Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    elif SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_LR_STEP_SIZE, gamma=STEP_LR_GAMMA)
        logger.log(f"  Scheduler: StepLR (step={STEP_LR_STEP_SIZE}, "
                    f"gamma={STEP_LR_GAMMA})")
    else:
        raise ValueError(f"Unknown SCHEDULER_TYPE: {SCHEDULER_TYPE}")

    logger.log(f"  Optimizer: Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    logger.log(f"  Criterion: CrossEntropyLoss")

    # ----------------------------------------------------------------
    #  CSV EPOCH LOG SETUP
    # ----------------------------------------------------------------
    csv_path = out_path("epoch_log.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "lr", "grad_norm", "epoch_time_s", "best_val_acc"
    ])

    # ----------------------------------------------------------------
    #  TRAINING LOOP
    # ----------------------------------------------------------------
    logger.log("\n" + "=" * 70)
    logger.log("  TRAINING")
    logger.log("=" * 70)

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    checkpoint_path = out_path("best_model.pth")

    header = (f"{'Epoch':>6s} | {'TrainLoss':>10s} | {'TrainAcc':>9s} | "
              f"{'ValLoss':>10s} | {'ValAcc':>9s} | {'LR':>10s} | "
              f"{'GradNorm':>10s} | {'Time':>7s} | {'Best':>9s} | "
              f"{'NoImp':>5s} | {'ETA':>12s}")
    logger.log(header)
    logger.log("-" * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)

        # Gradient norm (computed after backward in last step; approximate by
        # running one extra forward/backward on a single batch)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = compute_gradient_norm(model)
            optimizer.zero_grad()  # don't double-step
            break

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # ETA computation
        avg_epoch_time = (time.time() - start_wall) / epoch
        remaining_epochs = NUM_EPOCHS - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        if eta_hr > 0:
            eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s"
        elif eta_min > 0:
            eta_str = f"{eta_min}m {eta_sec}s"
        else:
            eta_str = f"{eta_sec}s"

        # Best model tracking
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "config": dict(config),
                "class_names": class_names,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Log
        star = " *" if improved else ""
        logger.log(
            f"{epoch:>6d} | {train_loss:>10.4f} | {train_acc:>8.4f}  | "
            f"{val_loss:>10.4f} | {val_acc:>8.4f}  | {current_lr:>10.6f} | "
            f"{grad_norm:>10.4f} | {epoch_time:>6.1f}s | "
            f"{best_val_acc:>8.4f}  | {epochs_no_improve:>5d} | "
            f"ETA: {eta_str}{star}"
        )

        # CSV
        csv_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{current_lr:.8f}",
            f"{grad_norm:.6f}",
            f"{epoch_time:.2f}",
            f"{best_val_acc:.6f}",
        ])
        csv_file.flush()

        # History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "grad_norm": grad_norm,
            "epoch_time_s": epoch_time,
            "best_val_acc": best_val_acc,
        })

        # Early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.log(f"\n  ** Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs) **")
            break

    csv_file.close()
    training_time = time.time() - start_wall
    logger.log(f"\nTraining complete. Total time: {training_time:.1f}s")
    logger.log(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # ----------------------------------------------------------------
    #  LOAD BEST MODEL & FINAL TEST EVALUATION
    # ----------------------------------------------------------------
    logger.log("\n" + "=" * 70)
    logger.log("  FINAL TEST EVALUATION (best model)")
    logger.log("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    all_preds, all_labels, all_outputs, test_loss, test_acc = \
        full_test_evaluation(model, test_loader, DEVICE, num_classes)

    # Top-5 accuracy
    k = min(5, num_classes)
    top5_acc = top_k_accuracy(all_outputs, [
        torch.tensor(all_labels[i:i+1]) for i in range(len(all_labels))
    ], k=k)
    # Recompute properly using collected tensors
    all_labels_tensor = [torch.tensor([lbl]) for lbl in all_labels]
    top5_acc = top_k_accuracy(all_outputs, all_labels_tensor, k=k)

    logger.log(f"\n  Test Loss:      {test_loss:.4f}")
    logger.log(f"  Test Accuracy:  {test_acc:.4f}  (Top-1)")
    logger.log(f"  Top-{k} Accuracy: {top5_acc:.4f}")

    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)

    # Per-class report
    logger.log(f"\n  {'Class':>8s} | {'Prec':>7s} | {'Recall':>7s} | "
               f"{'F1':>7s} | {'Support':>8s}")
    logger.log("  " + "-" * 50)
    supports = []
    for c in range(num_classes):
        support = int(cm[c].sum())
        supports.append(support)
        logger.log(f"  {class_names[c]:>8s} | {precision[c]:>7.4f} | "
                    f"{recall[c]:>7.4f} | {f1[c]:>7.4f} | {support:>8d}")

    # Macro averages
    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)
    total_support = sum(supports)

    # Weighted averages
    weights = np.array(supports, dtype=np.float64)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    weighted_prec = np.sum(precision * weights)
    weighted_rec = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)

    logger.log("  " + "-" * 50)
    logger.log(f"  {'Macro':>8s} | {macro_prec:>7.4f} | {macro_rec:>7.4f} | "
               f"{macro_f1:>7.4f} | {total_support:>8d}")
    logger.log(f"  {'Weighted':>8s} | {weighted_prec:>7.4f} | "
               f"{weighted_rec:>7.4f} | {weighted_f1:>7.4f} | "
               f"{total_support:>8d}")

    # ----------------------------------------------------------------
    #  PLOTS
    # ----------------------------------------------------------------
    logger.log("\n--- Saving Plots ---")

    plot_training_curves(history, out_path("training_curves.png"))
    logger.log(f"  Saved: {out_path('training_curves.png')}")

    plot_confusion_matrix(cm, class_names, out_path("confusion_matrix.png"))
    logger.log(f"  Saved: {out_path('confusion_matrix.png')}")

    plot_per_class_accuracy(cm, class_names,
                            out_path("per_class_accuracy.png"))
    logger.log(f"  Saved: {out_path('per_class_accuracy.png')}")

    plot_lr_schedule(history, out_path("lr_schedule.png"))
    logger.log(f"  Saved: {out_path('lr_schedule.png')}")

    # ----------------------------------------------------------------
    #  CLASSIFICATION REPORT (text file)
    # ----------------------------------------------------------------
    report_path = out_path("classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VisionAI Final CNN — Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Test Loss:       {test_loss:.4f}\n")
        f.write(f"Top-1 Accuracy:  {test_acc:.4f}\n")
        f.write(f"Top-{k} Accuracy:  {top5_acc:.4f}\n\n")
        f.write(f"{'Class':>8s} | {'Prec':>7s} | {'Recall':>7s} | "
                f"{'F1':>7s} | {'Support':>8s}\n")
        f.write("-" * 50 + "\n")
        for c in range(num_classes):
            support = int(cm[c].sum())
            f.write(f"{class_names[c]:>8s} | {precision[c]:>7.4f} | "
                    f"{recall[c]:>7.4f} | {f1[c]:>7.4f} | {support:>8d}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Macro':>8s} | {macro_prec:>7.4f} | {macro_rec:>7.4f} | "
                f"{macro_f1:>7.4f} | {total_support:>8d}\n")
        f.write(f"{'Weighted':>8s} | {weighted_prec:>7.4f} | "
                f"{weighted_rec:>7.4f} | {weighted_f1:>7.4f} | "
                f"{total_support:>8d}\n")
    logger.log(f"  Saved: {report_path}")

    # ----------------------------------------------------------------
    #  FULL RESULTS JSON
    # ----------------------------------------------------------------
    per_class_metrics = {}
    for c in range(num_classes):
        per_class_metrics[class_names[c]] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(cm[c].sum()),
            "accuracy": float(cm[c, c] / cm[c].sum()) if cm[c].sum() > 0 else 0.0,
        }

    results = {
        "config": dict(config),
        "dataset": {
            "total_samples": len(full_dataset),
            "num_classes": num_classes,
            "class_names": class_names,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "model": {
            "total_parameters": count_parameters(model)[0],
            "trainable_parameters": count_parameters(model)[1],
        },
        "training": {
            "total_epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "early_stopped": (EARLY_STOPPING_PATIENCE > 0
                              and len(history) < NUM_EPOCHS),
            "total_training_time_s": training_time,
        },
        "test_metrics": {
            "test_loss": test_loss,
            "top1_accuracy": test_acc,
            f"top{k}_accuracy": top5_acc,
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_prec),
            "weighted_recall": float(weighted_rec),
            "weighted_f1": float(weighted_f1),
        },
        "per_class_metrics": per_class_metrics,
        "epoch_history": history,
        "confusion_matrix": cm.tolist(),
    }

    json_path = out_path("results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.log(f"  Saved: {json_path}")

    # ----------------------------------------------------------------
    #  FINAL SUMMARY
    # ----------------------------------------------------------------
    logger.log("\n" + "=" * 70)
    logger.log("  SUMMARY")
    logger.log("=" * 70)
    logger.log(f"  Best Val Acc:   {best_val_acc:.4f} (epoch {best_epoch})")
    logger.log(f"  Test Top-1 Acc: {test_acc:.4f}")
    logger.log(f"  Test Top-{k} Acc: {top5_acc:.4f}")
    logger.log(f"  Macro F1:       {macro_f1:.4f}")
    logger.log(f"  Total Time:     {training_time:.1f}s")
    logger.log(f"  Checkpoint:     {checkpoint_path}")
    logger.log(f"  Output dir:     {output_dir}")
    logger.log("=" * 70)
    logger.log("Done.")

    logger.close()


if __name__ == "__main__":
    main()
