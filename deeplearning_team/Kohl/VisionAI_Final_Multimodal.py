"""
VisionAI_Final_Multimodal.py
============================
Multimodal CLIP-style contrastive learning for ASL sign classification,
retrieval, QA, and conditional text-to-image generation.

Two-phase training:
  Phase 1 -- Contrastive image-text alignment (InfoNCE / NT-Xent)
  Phase 2 -- Conditional decoder for text -> image generation

Post-training capabilities:
  * Image -> Text classification (top-k)
  * Text  -> Image retrieval   (top-k)
  * QA about ASL signs
  * Image -> Image similarity
  * Text  -> Image generation (decoder)
"""

import os
import sys
import json
import math
import time
import random
import logging
import csv
import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# ---- Paths ----------------------------------------------------------------
DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\archive\asl_dataset\asl_dataset"
DATASET_SIZE = "small"  # "small" = original 36-class dataset, "large" = 87k-image 29-class dataset
LARGER_DATASET_ROOT = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\LargerASL_Extract\asl_alphabet_train\asl_alphabet_train"
OUTPUT_DIR = r"C:\Users\kspar\OneDrive - University of St. Thomas\VisionAI\outputs"
OUTPUT_PREFIX = "VisionAI_Final_Multimodal_v2"

# ---- Data -----------------------------------------------------------------
IMG_SIZE = 128
DATA_SUBSET_FRACTION = 1.0        # 1.0 = full dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15                  # test = 1 - train - val
SEED = 42
NUM_WORKERS = 0                   # 0 is safest on Windows

# ---- Phase 1: Contrastive -------------------------------------------------
CONTRASTIVE_EPOCHS = 40
CONTRASTIVE_LR = 3e-4
CONTRASTIVE_BATCH_SIZE = 32
TEMPERATURE = 0.07
WEIGHT_DECAY = 1e-4

# ---- Phase 2: Decoder -----------------------------------------------------
DECODER_EPOCHS = 30
DECODER_LR = 1e-3
DECODER_BATCH_SIZE = 32

# ---- Model ----------------------------------------------------------------
EMBED_DIM = 256
TEXT_HIDDEN_DIM = 128
TEXT_MAX_LEN = 80
TEXT_NUM_HEADS = 4

# ---- Early stopping -------------------------------------------------------
EARLY_STOPPING_PATIENCE = 10      # 0 = disabled

# ---- Phase 3: Linear Probe -------------------------------------------
LINEAR_PROBE_EPOCHS = 30
LINEAR_PROBE_LR = 1e-3
LINEAR_PROBE_BATCH_SIZE = 32

# ---- Device ---------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 2. ASL KNOWLEDGE BASE
# ============================================================================

ASL_DESCRIPTIONS = {
    "a": "The ASL sign for letter A is made by forming a fist with the thumb resting on the side of the index finger, all fingers curled tightly inward.",
    "b": "The ASL sign for letter B is made by holding all four fingers straight up and together, with the thumb folded across the palm.",
    "c": "The ASL sign for letter C is made by curving the fingers and thumb into a C shape, as if gripping a small cup.",
    "d": "The ASL sign for letter D is made by touching the tip of the thumb to the tips of the middle, ring, and pinky fingers while the index finger points straight up.",
    "e": "The ASL sign for letter E is made by curling all fingers down toward the palm, with the thumb tucked underneath the fingertips.",
    "f": "The ASL sign for letter F is made by touching the tip of the index finger to the tip of the thumb forming a circle, with the remaining three fingers extended upward.",
    "g": "The ASL sign for letter G is made by pointing the index finger and thumb horizontally, parallel to each other, with other fingers curled in.",
    "h": "The ASL sign for letter H is made by extending the index and middle fingers together horizontally, with the thumb tucked and remaining fingers curled.",
    "i": "The ASL sign for letter I is made by extending only the pinky finger straight up, with all other fingers curled into the palm and thumb over them.",
    "j": "The ASL sign for letter J is made by extending the pinky finger and tracing a J shape in the air by rotating the wrist downward.",
    "k": "The ASL sign for letter K is made by pointing the index and middle fingers upward in a V shape, with the thumb placed between them.",
    "l": "The ASL sign for letter L is made by extending the thumb and index finger to form an L shape at a right angle, other fingers curled.",
    "m": "The ASL sign for letter M is made by placing the thumb under the index, middle, and ring fingers which are draped over it, pinky tucked.",
    "n": "The ASL sign for letter N is made by placing the thumb under the index and middle fingers which drape over it, ring and pinky tucked.",
    "o": "The ASL sign for letter O is made by curving all fingers and thumb together to form a round O shape, fingertips touching thumb tip.",
    "p": "The ASL sign for letter P is made similarly to K but pointing downward, with the index and middle fingers extended and thumb between them.",
    "q": "The ASL sign for letter Q is made similarly to G but pointing downward, with the index finger and thumb extended downward.",
    "r": "The ASL sign for letter R is made by crossing the index and middle fingers, with remaining fingers curled and thumb to the side.",
    "s": "The ASL sign for letter S is made by forming a fist with the thumb wrapped over the front of the curled fingers.",
    "t": "The ASL sign for letter T is made by placing the thumb between the index and middle fingers of a closed fist.",
    "u": "The ASL sign for letter U is made by holding the index and middle fingers straight up and together, thumb holding down ring and pinky.",
    "v": "The ASL sign for letter V is made by holding the index and middle fingers up in a V shape spread apart, other fingers curled.",
    "w": "The ASL sign for letter W is made by holding the index, middle, and ring fingers up and spread apart, thumb holding down the pinky.",
    "x": "The ASL sign for letter X is made by curling the index finger into a hook shape, with all other fingers closed into a fist.",
    "y": "The ASL sign for letter Y is made by extending the thumb and pinky finger outward, with the three middle fingers curled into the palm.",
    "z": "The ASL sign for letter Z is made by using the index finger to trace a Z shape in the air, starting from the top left.",
    "0": "The ASL sign for number 0 is made by forming an O shape with all fingers and thumb, similar to the letter O sign.",
    "1": "The ASL sign for number 1 is made by pointing the index finger straight up with all other fingers curled into the palm.",
    "2": "The ASL sign for number 2 is made by holding the index and middle fingers up and spread apart in a V shape, identical to letter V.",
    "3": "The ASL sign for number 3 is made by extending the thumb, index, and middle fingers while curling the ring and pinky fingers.",
    "4": "The ASL sign for number 4 is made by extending all four fingers straight up and spread apart, with the thumb folded across the palm.",
    "5": "The ASL sign for number 5 is made by extending all five fingers and the thumb wide apart with the palm facing forward.",
    "6": "The ASL sign for number 6 is made by touching the tip of the thumb to the tip of the pinky finger, with the other three fingers extended.",
    "7": "The ASL sign for number 7 is made by touching the tip of the thumb to the tip of the ring finger, with the other three fingers extended.",
    "8": "The ASL sign for number 8 is made by touching the tip of the thumb to the tip of the middle finger, with the other three fingers extended.",
    "9": "The ASL sign for number 9 is made by touching the tip of the thumb to the tip of the index finger forming a circle, similar to letter F.",
}

ASL_QA_PAIRS = {
    "a": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter A."),
        ("How is the hand positioned?", "A closed fist with the thumb resting on the side of the index finger."),
        ("Is this a static or dynamic sign?", "This is a static sign; the hand does not move."),
    ],
    "b": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter B."),
        ("How many fingers are extended?", "All four fingers are extended straight up and held together."),
    ],
    "c": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter C."),
        ("What shape does the hand form?", "The hand curves into a C shape, like gripping a small cup."),
    ],
    "d": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter D."),
        ("Which finger points up?", "The index finger points straight up while the other fingertips touch the thumb."),
    ],
    "f": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter F."),
        ("Which fingers form a circle?", "The index finger and thumb touch to form a circle."),
    ],
    "l": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter L."),
        ("What shape does the hand make?", "The thumb and index finger form an L shape at a right angle."),
    ],
    "o": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter O."),
        ("How are the fingers arranged?", "All fingers and thumb curve together to form a round O shape."),
    ],
    "v": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter V."),
        ("What is distinctive about this sign?", "The index and middle fingers are held up and spread in a V shape."),
    ],
    "y": [
        ("What does this ASL sign represent?", "This is the ASL fingerspelling sign for the letter Y."),
        ("Which fingers are extended?", "The thumb and pinky finger are extended outward."),
    ],
    "0": [
        ("What does this ASL sign represent?", "This is the ASL sign for the number 0."),
        ("How does it differ from the letter O?", "They are essentially the same hand shape in ASL fingerspelling."),
    ],
    "1": [
        ("What does this ASL sign represent?", "This is the ASL sign for the number 1."),
        ("Which finger is used?", "Only the index finger points straight up."),
    ],
    "5": [
        ("What does this ASL sign represent?", "This is the ASL sign for the number 5."),
        ("How many fingers are extended?", "All five fingers and the thumb are spread wide apart."),
    ],
}


# ============================================================================
# 3. HELPERS: LOGGING, SEEDING, DIRECTORIES
# ============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str, prefix: str):
    """Set up dual logging: console + file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{prefix}.log")
    logger = logging.getLogger("VisionAI")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 4. DATASET
# ============================================================================

class ASLDataset(Dataset):
    """Load ASL images from class-name sub-folders."""

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None, subset_fraction: float = 1.0):
        super().__init__()
        self.root = root
        self.transform = transform
        self.samples = []       # (path, label_idx)
        self.class_names = []   # sorted

        # Discover classes
        class_dirs = sorted(
            [d for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
        )
        self.class_names = class_dirs
        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}

        for cls_name in class_dirs:
            cls_dir = os.path.join(root, cls_name)
            files = [
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in self.VALID_EXT
            ]
            for fname in sorted(files):
                self.samples.append(
                    (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                )

        # Optional subset
        if 0.0 < subset_fraction < 1.0:
            rng = random.Random(SEED)
            k = max(1, int(len(self.samples) * subset_fraction))
            self.samples = rng.sample(self.samples, k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Build text description
        cls_name = self.class_names[label]
        text = ASL_DESCRIPTIONS.get(cls_name, f"ASL sign for {cls_name}")
        return img, text, label, path


def text_to_tensor(text: str, max_len: int) -> torch.LongTensor:
    """Character-level encoding: ASCII ordinals, 0-padded."""
    encoded = [ord(c) % 128 for c in text[:max_len]]
    pad_len = max_len - len(encoded)
    encoded += [0] * pad_len
    return torch.tensor(encoded, dtype=torch.long)


def stratified_split(dataset, train_ratio, val_ratio, seed):
    """Stratified split into train/val/test index lists."""
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)
    for i, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(label_to_indices.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


# ============================================================================
# 5. MODEL ARCHITECTURE
# ============================================================================

class ImageEncoder(nn.Module):
    """Conv blocks: 3->64->128->256->512, BN+ReLU+MaxPool, AdaptiveAvgPool, Linear."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 64, stride 2
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class SimpleTextEncoder(nn.Module):
    """Char-level embedding + positional embedding, MultiheadAttention, LayerNorm,
       global avg pool, FC -> embed_dim."""

    def __init__(self, vocab_size: int = 128, max_len: int = 80,
                 hidden_dim: int = 128, embed_dim: int = 256,
                 num_heads: int = 4):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # x: (B, max_len) LongTensor
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.char_embed(x) + self.pos_embed(positions)
        # Self-attention
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        # Global average pooling (ignore padding)
        mask = (x != 0).unsqueeze(-1).float()  # (B, L, 1)
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        h = (h * mask).sum(dim=1) / lengths
        h = self.fc(h)
        return h


class ConditionalDecoder(nn.Module):
    """Linear(embed_dim -> 512*4*4), 5x ConvTranspose2d -> 3x128x128, Sigmoid."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            # 512x4x4 -> 256x8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256x8x8 -> 128x16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128x16x16 -> 64x32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64x32x32 -> 32x64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32x64x64 -> 3x128x128
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x


class LinearProbe(nn.Module):
    """Lightweight linear classifier on top of frozen image embeddings."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ASLMultimodal(nn.Module):
    """Wraps image + text encoders with learnable temperature for contrastive loss."""

    def __init__(self, embed_dim=256, text_hidden_dim=128, text_max_len=80,
                 text_num_heads=4, init_temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = SimpleTextEncoder(
            vocab_size=128, max_len=text_max_len,
            hidden_dim=text_hidden_dim, embed_dim=embed_dim,
            num_heads=text_num_heads,
        )
        # Learnable log-temperature (clamped)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature))
        )

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, text_tokens):
        return F.normalize(self.text_encoder(text_tokens), dim=-1)

    def contrastive_loss(self, image_embeds, text_embeds):
        """Symmetric InfoNCE / NT-Xent loss."""
        temp = self.temperature
        logits = image_embeds @ text_embeds.t() / temp     # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2.0


# ============================================================================
# 6. METRICS (no sklearn)
# ============================================================================

def compute_i2t_accuracy(model, dataloader, class_names, device, text_max_len):
    """Image-to-text retrieval accuracy: for each image, find closest text embedding
       among all class descriptions and check if it matches the ground truth."""
    model.eval()

    # Pre-encode all class text embeddings
    class_texts = []
    for cn in class_names:
        desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
        class_texts.append(text_to_tensor(desc, text_max_len))
    class_text_batch = torch.stack(class_texts).to(device)

    with torch.no_grad():
        class_text_embeds = model.encode_text(class_text_batch)  # (C, D)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, _, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            img_embeds = model.encode_image(images)       # (B, D)
            sims = img_embeds @ class_text_embeds.t()     # (B, C)
            preds = sims.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


# ============================================================================
# 7. TRAINING LOOPS
# ============================================================================

def train_contrastive_epoch(model, dataloader, optimizer, device, text_max_len):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    for images, texts, labels, _ in dataloader:
        images = images.to(device)
        text_tokens = torch.stack([text_to_tensor(t, text_max_len) for t in texts]).to(device)

        img_embeds = model.encode_image(images)
        txt_embeds = model.encode_text(text_tokens)
        loss = model.contrastive_loss(img_embeds, txt_embeds)

        optimizer.zero_grad()
        loss.backward()

        # Gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_grad_norm / max(n_batches, 1)


def eval_contrastive(model, dataloader, device, text_max_len):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, texts, labels, _ in dataloader:
            images = images.to(device)
            text_tokens = torch.stack([text_to_tensor(t, text_max_len) for t in texts]).to(device)
            img_embeds = model.encode_image(images)
            txt_embeds = model.encode_text(text_tokens)
            loss = model.contrastive_loss(img_embeds, txt_embeds)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_decoder_epoch(decoder, model, dataloader, optimizer, device, text_max_len):
    decoder.train()
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        pass  # model stays frozen
    for images, texts, labels, _ in dataloader:
        images = images.to(device)
        text_tokens = torch.stack([text_to_tensor(t, text_max_len) for t in texts]).to(device)
        with torch.no_grad():
            txt_embeds = model.encode_text(text_tokens)
        recon = decoder(txt_embeds)
        loss = F.mse_loss(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def eval_decoder(decoder, model, dataloader, device, text_max_len):
    decoder.eval()
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, texts, labels, _ in dataloader:
            images = images.to(device)
            text_tokens = torch.stack([text_to_tensor(t, text_max_len) for t in texts]).to(device)
            txt_embeds = model.encode_text(text_tokens)
            recon = decoder(txt_embeds)
            loss = F.mse_loss(recon, images)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ============================================================================
# 7b. LINEAR PROBE TRAINING LOOPS
# ============================================================================

def train_linear_probe_epoch(probe, model, dataloader, optimizer, device):
    """Train the linear probe for one epoch. The main model is frozen."""
    probe.train()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    with torch.no_grad():
        pass  # model stays frozen throughout
    for images, texts, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            img_embeds = model.encode_image(images)
        logits = probe(img_embeds)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def eval_linear_probe(probe, model, dataloader, device):
    """Evaluate the linear probe. Returns loss and accuracy."""
    probe.eval()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    with torch.no_grad():
        for images, texts, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            img_embeds = model.encode_image(images)
            logits = probe(img_embeds)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ============================================================================
# 7c. NEW DEMO / ANALYSIS FUNCTIONS
# ============================================================================

def cross_modal_arithmetic_demo(model, dataset, class_names, device, text_max_len,
                                save_path, logger):
    """Cross-modal arithmetic: embed(img_X) - embed(text_X) + embed(text_Y) ~ embed(img_Y)?

    Tests whether the shared embedding space supports vector arithmetic analogies.
    For each (source, target) pair we check if the transformed vector is closest
    to the target class text embedding.
    """
    try:
        model.eval()
        # Pre-encode all class text embeddings
        class_texts = []
        for cn in class_names:
            desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
            class_texts.append(text_to_tensor(desc, text_max_len))
        class_text_batch = torch.stack(class_texts).to(device)
        with torch.no_grad():
            class_text_embeds = model.encode_text(class_text_batch)  # (C, D)

        # Define analogy pairs: (source_class, target_class)
        pairs = [("a", "b"), ("c", "d"), ("l", "v"), ("1", "5"), ("o", "0")]
        # Filter to classes actually present
        pairs = [(s, t) for s, t in pairs
                 if s in class_names and t in class_names]

        if not pairs:
            logger.warning("Cross-modal arithmetic: no valid pairs found, skipping.")
            return

        # Collect one image per class from dataset
        class_to_img = {}
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for images, _, labels, _ in loader:
            cls = class_names[labels[0].item()]
            if cls not in class_to_img:
                class_to_img[cls] = images[0]
            if len(class_to_img) >= len(class_names):
                break

        results = []
        logger.info("Cross-Modal Arithmetic Demo:")
        logger.info("  Formula: embed(img_X) - embed(text_X) + embed(text_Y) => nearest class?")

        for src, tgt in pairs:
            if src not in class_to_img:
                continue
            src_idx = class_names.index(src)
            tgt_idx = class_names.index(tgt)

            with torch.no_grad():
                img_embed = model.encode_image(class_to_img[src].unsqueeze(0).to(device))
                # arithmetic: img(src) - text(src) + text(tgt)
                transformed = img_embed - class_text_embeds[src_idx].unsqueeze(0) + class_text_embeds[tgt_idx].unsqueeze(0)
                transformed = F.normalize(transformed, dim=-1)
                sims = (transformed @ class_text_embeds.t()).squeeze(0)
                pred_idx = sims.argmax().item()
                pred_class = class_names[pred_idx]
                correct = pred_class == tgt

            results.append({
                "source": src, "target": tgt,
                "predicted": pred_class, "correct": correct,
                "similarity": sims[tgt_idx].item(),
            })
            status = "CORRECT" if correct else f"WRONG (got '{pred_class}')"
            logger.info(f"  img({src}) - text({src}) + text({tgt}) => {status}  "
                        f"[sim to target: {sims[tgt_idx].item():.4f}]")

        n_correct = sum(1 for r in results if r["correct"])
        logger.info(f"  Arithmetic accuracy: {n_correct}/{len(results)}")

        # Visualization
        fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
        if len(results) == 1:
            axes = [axes]
        for i, r in enumerate(results):
            axes[i].bar(["Target", "Predicted"],
                        [r["similarity"], 1.0 if r["correct"] else 0.5],
                        color=["green" if r["correct"] else "red",
                               "green" if r["correct"] else "red"])
            axes[i].set_title(f"img({r['source']})-txt({r['source']})+txt({r['target']})\n"
                              f"=> {r['predicted']} ({'OK' if r['correct'] else 'WRONG'})",
                              fontsize=9)
            axes[i].set_ylim(0, 1.1)
        plt.suptitle("Cross-Modal Arithmetic Demo", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"  Saved cross-modal arithmetic figure to {save_path}")

    except Exception as e:
        logger.error(f"Cross-modal arithmetic demo failed: {e}")


def image_to_image_retrieval_demo(model, dataset, device, save_path, logger,
                                  n_queries=4, top_k=3):
    """Pick query images from dataset, find top-K most similar images by cosine
    similarity in the embedding space, and create a visualization grid."""
    try:
        model.eval()
        # Collect all embeddings
        all_embeds = []
        all_paths = []
        all_labels = []
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        with torch.no_grad():
            for images, _, labels, paths in loader:
                images = images.to(device)
                embeds = model.encode_image(images)
                all_embeds.append(embeds.cpu())
                all_paths.extend(paths)
                all_labels.extend(labels.tolist())

        all_embeds = torch.cat(all_embeds, dim=0)  # (N, D)
        all_embeds = F.normalize(all_embeds, dim=-1)

        # Pick n_queries random indices (spread across classes)
        rng = random.Random(SEED)
        query_indices = rng.sample(range(len(all_paths)), min(n_queries, len(all_paths)))

        fig, axes = plt.subplots(n_queries, top_k + 1,
                                 figsize=(3 * (top_k + 1), 3 * n_queries))
        if n_queries == 1:
            axes = [axes]

        for qi, q_idx in enumerate(query_indices):
            q_embed = all_embeds[q_idx].unsqueeze(0)
            sims = (q_embed @ all_embeds.t()).squeeze(0)
            # Exclude the query itself
            sims[q_idx] = -2.0
            topk_vals, topk_idxs = sims.topk(top_k)

            # Show query
            q_img = Image.open(all_paths[q_idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            axes[qi][0].imshow(np.array(q_img))
            axes[qi][0].set_title(f"Query (label={all_labels[q_idx]})", fontsize=9)
            axes[qi][0].axis("off")

            # Show retrieved
            for ri in range(top_k):
                r_idx = topk_idxs[ri].item()
                r_img = Image.open(all_paths[r_idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                match = all_labels[r_idx] == all_labels[q_idx]
                color = "green" if match else "red"
                axes[qi][ri + 1].imshow(np.array(r_img))
                axes[qi][ri + 1].set_title(
                    f"#{ri+1} sim={topk_vals[ri].item():.3f}\nlabel={all_labels[r_idx]}",
                    fontsize=8, color=color)
                axes[qi][ri + 1].axis("off")

        plt.suptitle("Image-to-Image Retrieval (Cosine Similarity)", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Image-to-image retrieval demo saved to {save_path}")

    except Exception as e:
        logger.error(f"Image-to-image retrieval demo failed: {e}")


def compute_per_class_metrics(model, dataloader, class_names, device, text_max_len,
                              save_path, logger):
    """Compute Precision@1, @3, @5 per class. Create a bar chart of P@1 and a
    confusion matrix heatmap (predicted vs actual from I2T retrieval)."""
    try:
        model.eval()
        num_classes = len(class_names)

        # Pre-encode class text embeddings
        class_texts = []
        for cn in class_names:
            desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
            class_texts.append(text_to_tensor(desc, text_max_len))
        class_text_batch = torch.stack(class_texts).to(device)
        with torch.no_grad():
            class_text_embeds = model.encode_text(class_text_batch)  # (C, D)

        # Collect predictions
        all_labels = []
        all_top5 = []  # list of lists of top-5 predicted indices
        with torch.no_grad():
            for images, _, labels, _ in dataloader:
                images = images.to(device)
                img_embeds = model.encode_image(images)
                sims = img_embeds @ class_text_embeds.t()  # (B, C)
                top5 = sims.topk(min(5, num_classes), dim=1).indices.cpu().tolist()
                all_labels.extend(labels.tolist())
                all_top5.extend(top5)

        # Per-class precision@k
        precisions = {1: defaultdict(list), 3: defaultdict(list), 5: defaultdict(list)}
        confusion = np.zeros((num_classes, num_classes), dtype=int)

        for label, top5 in zip(all_labels, all_top5):
            pred = top5[0]
            confusion[label][pred] += 1
            for k in [1, 3, 5]:
                topk = top5[:k]
                hit = 1.0 if label in topk else 0.0
                precisions[k][label].append(hit)

        logger.info("Per-Class Retrieval Metrics:")
        logger.info(f"  {'Class':>6s}  P@1     P@3     P@5")
        logger.info(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

        p1_values = []
        for ci, cn in enumerate(class_names):
            p1 = np.mean(precisions[1][ci]) if precisions[1][ci] else 0.0
            p3 = np.mean(precisions[3][ci]) if precisions[3][ci] else 0.0
            p5 = np.mean(precisions[5][ci]) if precisions[5][ci] else 0.0
            p1_values.append(p1)
            logger.info(f"  {cn:>6s}  {p1:.4f}  {p3:.4f}  {p5:.4f}")

        avg_p1 = np.mean(p1_values)
        logger.info(f"  {'AVG':>6s}  {avg_p1:.4f}")

        # Bar chart of P@1
        fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.4), 5))
        x_pos = np.arange(num_classes)
        bars = ax.bar(x_pos, p1_values, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.axhline(y=avg_p1, color="red", linestyle="--", label=f"Mean P@1 = {avg_p1:.3f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, fontsize=7, rotation=90)
        ax.set_ylabel("Precision@1")
        ax.set_title("Per-Class Precision@1 (I2T Retrieval)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        bar_path = save_path.replace(".png", "_p1_bar.png")
        plt.savefig(bar_path, dpi=150)
        plt.close()
        logger.info(f"  P@1 bar chart saved to {bar_path}")

        # Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.35),
                                        max(8, num_classes * 0.3)))
        im = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, fontsize=6, rotation=90)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("I2T Retrieval Confusion Matrix")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        cm_path = save_path.replace(".png", "_confusion.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        logger.info(f"  Confusion matrix saved to {cm_path}")

        return {"per_class_p1": {cn: p1_values[i] for i, cn in enumerate(class_names)},
                "mean_p1": avg_p1}

    except Exception as e:
        logger.error(f"Per-class metrics computation failed: {e}")
        return {}


def plot_tsne_embeddings(model, dataloader, class_names, device, text_max_len,
                         save_path, max_samples=500):
    """Collect image + text embeddings and visualize with t-SNE.
    Uses sklearn if available, otherwise skips gracefully."""
    try:
        # Try to import TSNE
        tsne_available = False
        try:
            from sklearn.manifold import TSNE
            tsne_available = True
        except ImportError:
            pass

        if not tsne_available:
            # Try scipy-based simple fallback -- but t-SNE is complex, so just skip
            logging.getLogger("VisionAI").warning(
                "t-SNE visualization skipped: sklearn not available. "
                "Install scikit-learn for t-SNE plots.")
            return

        model.eval()
        all_img_embeds = []
        all_labels = []
        count = 0

        with torch.no_grad():
            for images, _, labels, _ in dataloader:
                images = images.to(device)
                img_embeds = model.encode_image(images)
                all_img_embeds.append(img_embeds.cpu())
                all_labels.extend(labels.tolist())
                count += images.size(0)
                if count >= max_samples:
                    break

        all_img_embeds = torch.cat(all_img_embeds, dim=0)[:max_samples]
        all_labels = all_labels[:max_samples]

        # Encode class text embeddings
        class_texts = []
        for cn in class_names:
            desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
            class_texts.append(text_to_tensor(desc, text_max_len))
        class_text_batch = torch.stack(class_texts).to(device)
        with torch.no_grad():
            class_text_embeds = model.encode_text(class_text_batch).cpu()

        # Combine image and text embeddings
        combined = torch.cat([all_img_embeds, class_text_embeds], dim=0).numpy()
        n_img = len(all_img_embeds)
        n_txt = len(class_text_embeds)

        perplexity = min(30, max(5, combined.shape[0] // 4))
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
        coords = tsne.fit_transform(combined)

        img_coords = coords[:n_img]
        txt_coords = coords[n_img:]

        # Assign colors per class
        unique_labels = sorted(set(all_labels))
        cmap = plt.cm.get_cmap("tab20", len(class_names))

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot image embeddings as circles
        for li, lbl in enumerate(unique_labels):
            mask = [i for i, l in enumerate(all_labels) if l == lbl]
            ax.scatter(img_coords[mask, 0], img_coords[mask, 1],
                       c=[cmap(lbl)], marker="o", s=15, alpha=0.5,
                       label=f"img:{class_names[lbl]}")

        # Plot text embeddings as triangles
        for ci in range(n_txt):
            ax.scatter(txt_coords[ci, 0], txt_coords[ci, 1],
                       c=[cmap(ci)], marker="^", s=80, edgecolors="black",
                       linewidths=0.8, zorder=5)
            ax.annotate(class_names[ci], (txt_coords[ci, 0], txt_coords[ci, 1]),
                        fontsize=7, ha="center", va="bottom")

        ax.set_title("t-SNE: Image (circles) and Text (triangles) Embeddings", fontsize=13)
        ax.grid(True, alpha=0.2)
        # Legend would be too crowded with many classes, skip if > 20
        if len(class_names) <= 20:
            ax.legend(fontsize=6, ncol=3, loc="upper right",
                      markerscale=1.5, framealpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.getLogger("VisionAI").info(f"t-SNE visualization saved to {save_path}")

    except Exception as e:
        logging.getLogger("VisionAI").error(f"t-SNE visualization failed: {e}")


def plot_embedding_interpolation(decoder, model, class_names, device, text_max_len,
                                 save_path, pairs=None, n_steps=8):
    """Interpolate between text embeddings of class pairs, generate images through
    the decoder, and show the smooth morphing in a grid."""
    try:
        decoder.eval()
        model.eval()

        if pairs is None:
            # Default pairs -- filter to those present in class_names
            candidate_pairs = [("a", "b"), ("l", "v"), ("1", "5"), ("c", "o")]
            pairs = [(a, b) for a, b in candidate_pairs
                     if a in class_names and b in class_names]

        if not pairs:
            logging.getLogger("VisionAI").warning(
                "Embedding interpolation: no valid pairs, skipping.")
            return

        # Pre-encode text embeddings for all classes we need
        needed = set()
        for a, b in pairs:
            needed.add(a)
            needed.add(b)

        class_embeds = {}
        for cn in needed:
            desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
            tok = text_to_tensor(desc, text_max_len).unsqueeze(0).to(device)
            with torch.no_grad():
                class_embeds[cn] = model.encode_text(tok).squeeze(0)

        alphas = np.linspace(0, 1, n_steps)
        n_pairs = len(pairs)

        fig, axes = plt.subplots(n_pairs, n_steps, figsize=(2.2 * n_steps, 2.5 * n_pairs))
        if n_pairs == 1:
            axes = [axes]

        for pi, (cls_a, cls_b) in enumerate(pairs):
            embed_a = class_embeds[cls_a]
            embed_b = class_embeds[cls_b]

            for si, alpha in enumerate(alphas):
                interp = (1 - alpha) * embed_a + alpha * embed_b
                interp = F.normalize(interp.unsqueeze(0), dim=-1)
                with torch.no_grad():
                    generated = decoder(interp)
                img = generated[0].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[pi][si].imshow(img)
                axes[pi][si].axis("off")
                if si == 0:
                    axes[pi][si].set_title(f"{cls_a}", fontsize=9, fontweight="bold")
                elif si == n_steps - 1:
                    axes[pi][si].set_title(f"{cls_b}", fontsize=9, fontweight="bold")
                else:
                    axes[pi][si].set_title(f"a={alpha:.2f}", fontsize=7)

        plt.suptitle("Embedding Interpolation through Decoder", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.getLogger("VisionAI").info(
            f"Embedding interpolation saved to {save_path}")

    except Exception as e:
        logging.getLogger("VisionAI").error(f"Embedding interpolation failed: {e}")


def plot_linear_probe_curves(epoch_data, save_path):
    """Plot train/val loss and accuracy curves for the linear probe."""
    epochs = [d["epoch"] for d in epoch_data]
    train_loss = [d["train_loss"] for d in epoch_data]
    val_loss = [d["val_loss"] for d in epoch_data]
    train_acc = [d["train_acc"] for d in epoch_data]
    val_acc = [d["val_acc"] for d in epoch_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Phase 3: Linear Probe Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, "b-o", markersize=3, label="Train Accuracy")
    ax2.plot(epochs, val_acc, "g-o", markersize=3, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Phase 3: Linear Probe Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# 8. DEMO FUNCTIONS
# ============================================================================

def image_to_text(model, image_tensor, class_names, device, text_max_len, top_k=5):
    """Given a single image tensor, return top-k matching class labels with scores."""
    model.eval()
    class_texts = []
    for cn in class_names:
        desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
        class_texts.append(text_to_tensor(desc, text_max_len))
    class_text_batch = torch.stack(class_texts).to(device)

    with torch.no_grad():
        img_embed = model.encode_image(image_tensor.unsqueeze(0).to(device))
        txt_embeds = model.encode_text(class_text_batch)
        sims = (img_embed @ txt_embeds.t()).squeeze(0)
    scores, indices = sims.topk(top_k)
    results = [(class_names[idx.item()], scores[i].item()) for i, idx in enumerate(indices)]
    return results


def text_to_image_retrieval(model, query_text, dataset, device, text_max_len, top_k=5):
    """Given query text, find top-k closest images from the dataset."""
    model.eval()
    text_tok = text_to_tensor(query_text, text_max_len).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embed = model.encode_text(text_tok)

    all_sims = []
    all_paths = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images, _, labels, paths in loader:
            images = images.to(device)
            img_embeds = model.encode_image(images)
            sims = (query_embed @ img_embeds.t()).squeeze(0)
            for i in range(sims.size(0)):
                all_sims.append(sims[i].item())
                all_paths.append(paths[i])

    paired = sorted(zip(all_sims, all_paths), reverse=True)[:top_k]
    return paired


def answer_question(model, image_tensor, question, class_names, device, text_max_len):
    """Match image to class, then search QA pairs for best answer."""
    top_results = image_to_text(model, image_tensor, class_names, device, text_max_len, top_k=1)
    predicted_class = top_results[0][0]
    confidence = top_results[0][1]

    qa_pairs = ASL_QA_PAIRS.get(predicted_class, [])
    if not qa_pairs:
        return predicted_class, confidence, "No QA data available for this class.", question

    # Simple keyword overlap matching
    q_words = set(question.lower().split())
    best_score = -1
    best_answer = qa_pairs[0][1]
    best_q = qa_pairs[0][0]
    for qa_q, qa_a in qa_pairs:
        qa_words = set(qa_q.lower().split())
        overlap = len(q_words & qa_words)
        if overlap > best_score:
            best_score = overlap
            best_answer = qa_a
            best_q = qa_q

    return predicted_class, confidence, best_answer, best_q


# ============================================================================
# 9. PLOTTING
# ============================================================================

def plot_contrastive_curves(epoch_data, save_path):
    """Train/val loss + I2T accuracy curves."""
    epochs = [d["epoch"] for d in epoch_data]
    train_loss = [d["train_loss"] for d in epoch_data]
    val_loss = [d["val_loss"] for d in epoch_data]
    i2t_acc = [d["val_i2t_acc"] for d in epoch_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Contrastive Loss")
    ax1.set_title("Phase 1: Contrastive Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, i2t_acc, "g-o", markersize=3, label="Val I2T Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Phase 1: Image-to-Text Retrieval Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_lr_schedule(epoch_data, save_path):
    """Learning rate over epochs."""
    epochs = [d["epoch"] for d in epoch_data]
    lrs = [d["lr"] for d in epoch_data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, "m-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Phase 1)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_text_similarity_matrix(model, class_names, device, text_max_len, save_path):
    """Heatmap of cosine similarities among all class text embeddings."""
    model.eval()
    texts = []
    for cn in class_names:
        desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
        texts.append(text_to_tensor(desc, text_max_len))
    batch = torch.stack(texts).to(device)
    with torch.no_grad():
        embeds = model.encode_text(batch)
    sim_matrix = (embeds @ embeds.t()).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7, rotation=90)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_title("Text Embedding Cosine Similarity Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_generated_images(decoder, model, class_names, device, text_max_len, save_path, n=8):
    """Generate images from text descriptions for n classes."""
    decoder.eval()
    model.eval()
    selected = class_names[:n]
    texts = []
    for cn in selected:
        desc = ASL_DESCRIPTIONS.get(cn, f"ASL sign for {cn}")
        texts.append(text_to_tensor(desc, text_max_len))
    batch = torch.stack(texts).to(device)
    with torch.no_grad():
        txt_embeds = model.encode_text(batch)
        generated = decoder(txt_embeds)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i in range(n):
        img = generated[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {selected[i]}", fontsize=10)
        axes[i].axis("off")
    plt.suptitle("Phase 2: Text-to-Image Generation", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_retrieval_demo(model, dataset, class_names, device, text_max_len, save_path):
    """Show top-3 retrieved images for a few query texts."""
    queries = ["ASL sign for letter A", "ASL sign for letter V",
               "ASL sign for number 5", "ASL sign for letter L"]
    n_queries = len(queries)

    fig, axes = plt.subplots(n_queries, 4, figsize=(14, 3.5 * n_queries))
    for qi, query in enumerate(queries):
        # Query label
        axes[qi][0].text(0.5, 0.5, query, ha="center", va="center",
                         fontsize=10, wrap=True,
                         transform=axes[qi][0].transAxes)
        axes[qi][0].set_title("Query", fontsize=10)
        axes[qi][0].axis("off")

        results = text_to_image_retrieval(model, query, dataset, device,
                                          text_max_len, top_k=3)
        for ri, (score, path) in enumerate(results):
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            axes[qi][ri + 1].imshow(np.array(img))
            axes[qi][ri + 1].set_title(f"Score: {score:.3f}", fontsize=9)
            axes[qi][ri + 1].axis("off")

    plt.suptitle("Text-to-Image Retrieval Demo", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR, OUTPUT_PREFIX)

    logger.info("=" * 70)
    logger.info("VisionAI Final Multimodal -- Training Script")
    logger.info("=" * 70)

    # ---- Log all config ---------------------------------------------------
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
        "CONTRASTIVE_EPOCHS": CONTRASTIVE_EPOCHS,
        "CONTRASTIVE_LR": CONTRASTIVE_LR,
        "CONTRASTIVE_BATCH_SIZE": CONTRASTIVE_BATCH_SIZE,
        "TEMPERATURE": TEMPERATURE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DECODER_EPOCHS": DECODER_EPOCHS,
        "DECODER_LR": DECODER_LR,
        "DECODER_BATCH_SIZE": DECODER_BATCH_SIZE,
        "EMBED_DIM": EMBED_DIM,
        "TEXT_HIDDEN_DIM": TEXT_HIDDEN_DIM,
        "TEXT_MAX_LEN": TEXT_MAX_LEN,
        "TEXT_NUM_HEADS": TEXT_NUM_HEADS,
        "LINEAR_PROBE_EPOCHS": LINEAR_PROBE_EPOCHS,
        "LINEAR_PROBE_LR": LINEAR_PROBE_LR,
        "LINEAR_PROBE_BATCH_SIZE": LINEAR_PROBE_BATCH_SIZE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "DEVICE": DEVICE,
    }
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # ---- Transforms -------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # ---- Dataset ----------------------------------------------------------
    active_root = LARGER_DATASET_ROOT if DATASET_SIZE == "large" else DATASET_ROOT
    full_dataset = ASLDataset(active_root, transform=None,
                              subset_fraction=DATA_SUBSET_FRACTION)
    logger.info(f"Dataset loaded: {len(full_dataset)} samples, "
                f"{len(full_dataset.class_names)} classes")
    logger.info(f"Classes: {full_dataset.class_names}")

    # Per-class counts
    class_counts = defaultdict(int)
    for _, label in full_dataset.samples:
        class_counts[full_dataset.class_names[label]] += 1
    for cn in full_dataset.class_names:
        logger.info(f"  Class '{cn}': {class_counts[cn]} images")

    # ---- Stratified split -------------------------------------------------
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, TRAIN_RATIO, VAL_RATIO, SEED
    )
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Wrap subsets with appropriate transforms
    class TransformSubset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            path, label = self.base.samples[real_idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            cls_name = self.base.class_names[label]
            text = ASL_DESCRIPTIONS.get(cls_name, f"ASL sign for {cls_name}")
            return img, text, label, path

    train_set = TransformSubset(full_dataset, train_idx, train_transform)
    val_set = TransformSubset(full_dataset, val_idx, eval_transform)
    test_set = TransformSubset(full_dataset, test_idx, eval_transform)

    train_loader = DataLoader(train_set, batch_size=CONTRASTIVE_BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=True, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_set, batch_size=CONTRASTIVE_BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            drop_last=False, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_set, batch_size=CONTRASTIVE_BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             drop_last=False, pin_memory=(DEVICE == "cuda"))

    # ---- Model ------------------------------------------------------------
    model = ASLMultimodal(
        embed_dim=EMBED_DIM,
        text_hidden_dim=TEXT_HIDDEN_DIM,
        text_max_len=TEXT_MAX_LEN,
        text_num_heads=TEXT_NUM_HEADS,
        init_temperature=TEMPERATURE,
    ).to(DEVICE)

    logger.info(f"ImageEncoder params: {count_parameters(model.image_encoder):,}")
    logger.info(f"TextEncoder params:  {count_parameters(model.text_encoder):,}")
    logger.info(f"Total model params:  {count_parameters(model):,}")

    # ====================================================================
    # PHASE 1: CONTRASTIVE TRAINING
    # ====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: Contrastive Image-Text Alignment")
    logger.info("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONTRASTIVE_LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONTRASTIVE_EPOCHS, eta_min=1e-6
    )

    contrastive_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_contrastive_epochs.csv")
    csv_file = open(contrastive_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_i2t_acc",
                         "lr", "grad_norm", "temperature", "epoch_time_s"])

    best_i2t_acc = 0.0
    best_contrastive_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_contrastive.pt")
    epochs_no_improve = 0
    contrastive_epoch_data = []
    phase1_start = time.time()

    for epoch in range(1, CONTRASTIVE_EPOCHS + 1):
        t0 = time.time()
        train_loss, grad_norm = train_contrastive_epoch(
            model, train_loader, optimizer, DEVICE, TEXT_MAX_LEN
        )
        val_loss = eval_contrastive(model, val_loader, DEVICE, TEXT_MAX_LEN)
        val_i2t = compute_i2t_accuracy(
            model, val_loader, full_dataset.class_names, DEVICE, TEXT_MAX_LEN
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        current_temp = model.temperature.item()
        elapsed = time.time() - t0

        improved = ""
        if val_i2t > best_i2t_acc:
            best_i2t_acc = val_i2t
            torch.save(model.state_dict(), best_contrastive_path)
            epochs_no_improve = 0
            improved = " [BEST]"
        else:
            epochs_no_improve += 1

        # ETA
        avg_epoch_time = (time.time() - phase1_start) / epoch
        eta_seconds = avg_epoch_time * (CONTRASTIVE_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        logger.info(
            f"Epoch {epoch:3d}/{CONTRASTIVE_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val I2T Acc: {val_i2t:.4f} | Best I2T: {best_i2t_acc:.4f} | "
            f"LR: {current_lr:.2e} | Temp: {current_temp:.4f} | "
            f"GradNorm: {grad_norm:.2f} | Time: {elapsed:.1f}s | ETA: {eta_str}{improved}"
        )

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_i2t_acc": val_i2t, "lr": current_lr, "grad_norm": grad_norm,
            "temperature": current_temp, "epoch_time": elapsed,
        }
        contrastive_epoch_data.append(row)
        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_i2t:.6f}", f"{current_lr:.8f}",
                             f"{grad_norm:.4f}", f"{current_temp:.6f}",
                             f"{elapsed:.2f}"])
        csv_file.flush()

        # Early stopping
        if EARLY_STOPPING_PATIENCE > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs "
                        f"({EARLY_STOPPING_PATIENCE} epochs without improvement).")
            break

    csv_file.close()
    logger.info(f"Phase 1 complete. Best val I2T accuracy: {best_i2t_acc:.4f}")

    # Reload best contrastive model
    model.load_state_dict(torch.load(best_contrastive_path, map_location=DEVICE,
                                     weights_only=True))
    logger.info("Loaded best contrastive model checkpoint.")

    # Test set evaluation
    test_i2t = compute_i2t_accuracy(
        model, test_loader, full_dataset.class_names, DEVICE, TEXT_MAX_LEN
    )
    logger.info(f"Test I2T Accuracy: {test_i2t:.4f}")

    # ====================================================================
    # PHASE 2: CONDITIONAL DECODER
    # ====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 2: Conditional Text-to-Image Decoder")
    logger.info("=" * 70)

    decoder = ConditionalDecoder(embed_dim=EMBED_DIM).to(DEVICE)
    logger.info(f"Decoder params: {count_parameters(decoder):,}")

    dec_train_loader = DataLoader(train_set, batch_size=DECODER_BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS,
                                  drop_last=True, pin_memory=(DEVICE == "cuda"))
    dec_val_loader = DataLoader(val_set, batch_size=DECODER_BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS,
                                drop_last=False, pin_memory=(DEVICE == "cuda"))

    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=10, gamma=0.5)

    decoder_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_decoder_epochs.csv")
    dec_csv = open(decoder_csv_path, "w", newline="", encoding="utf-8")
    dec_csv_writer = csv.writer(dec_csv)
    dec_csv_writer.writerow(["epoch", "train_mse", "val_mse", "epoch_time_s"])

    best_val_mse = float("inf")
    best_decoder_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_decoder.pt")
    decoder_epoch_data = []
    phase2_start = time.time()

    for epoch in range(1, DECODER_EPOCHS + 1):
        t0 = time.time()
        train_mse = train_decoder_epoch(
            decoder, model, dec_train_loader, dec_optimizer, DEVICE, TEXT_MAX_LEN
        )
        val_mse = eval_decoder(decoder, model, dec_val_loader, DEVICE, TEXT_MAX_LEN)
        dec_scheduler.step()
        elapsed = time.time() - t0

        improved = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(decoder.state_dict(), best_decoder_path)
            improved = " [BEST]"

        # ETA
        avg_epoch_time = (time.time() - phase2_start) / epoch
        eta_seconds = avg_epoch_time * (DECODER_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0 else f"{eta_min}m {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

        logger.info(
            f"Epoch {epoch:3d}/{DECODER_EPOCHS} | "
            f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | "
            f"Time: {elapsed:.1f}s | ETA: {eta_str}{improved}"
        )

        decoder_epoch_data.append({
            "epoch": epoch, "train_mse": train_mse,
            "val_mse": val_mse, "epoch_time": elapsed,
        })
        dec_csv_writer.writerow([epoch, f"{train_mse:.8f}", f"{val_mse:.8f}",
                                 f"{elapsed:.2f}"])
        dec_csv.flush()

    dec_csv.close()
    logger.info(f"Phase 2 complete. Best val MSE: {best_val_mse:.6f}")

    # Reload best decoder
    decoder.load_state_dict(torch.load(best_decoder_path, map_location=DEVICE,
                                       weights_only=True))
    logger.info("Loaded best decoder checkpoint.")

    # ====================================================================
    # DEMO
    # ====================================================================
    logger.info("=" * 70)
    logger.info("DEMO RESULTS")
    logger.info("=" * 70)

    # Demo: image_to_text
    demo_classes = ["a", "b", "v", "5", "l"]
    for dc in demo_classes:
        # Find first image of this class in test set
        demo_img = None
        for idx in test_idx:
            path, label = full_dataset.samples[idx]
            if full_dataset.class_names[label] == dc:
                img = Image.open(path).convert("RGB")
                demo_img = eval_transform(img)
                break
        if demo_img is None:
            continue
        results = image_to_text(model, demo_img, full_dataset.class_names,
                                DEVICE, TEXT_MAX_LEN, top_k=5)
        logger.info(f"Image-to-Text for class '{dc}':")
        for rank, (cls, score) in enumerate(results, 1):
            logger.info(f"  #{rank}: {cls} (score={score:.4f})")

    # Demo: answer_question
    for dc in ["a", "v", "1"]:
        demo_img = None
        for idx in test_idx:
            path, label = full_dataset.samples[idx]
            if full_dataset.class_names[label] == dc:
                img = Image.open(path).convert("RGB")
                demo_img = eval_transform(img)
                break
        if demo_img is None:
            continue
        pred_cls, conf, answer, matched_q = answer_question(
            model, demo_img, "What does this ASL sign represent?",
            full_dataset.class_names, DEVICE, TEXT_MAX_LEN
        )
        logger.info(f"QA for class '{dc}': predicted='{pred_cls}' (conf={conf:.4f})")
        logger.info(f"  Q: What does this ASL sign represent?")
        logger.info(f"  Matched Q: {matched_q}")
        logger.info(f"  A: {answer}")

    # Demo: text_to_image retrieval
    for query_label in ["a", "5"]:
        query = ASL_DESCRIPTIONS[query_label]
        results = text_to_image_retrieval(
            model, query, test_set, DEVICE, TEXT_MAX_LEN, top_k=3
        )
        logger.info(f"Text-to-Image Retrieval for '{query_label}':")
        for rank, (score, path) in enumerate(results, 1):
            logger.info(f"  #{rank}: score={score:.4f}, path={os.path.basename(path)}")

    # ====================================================================
    # PLOTS
    # ====================================================================
    logger.info("Generating plots...")

    plot_contrastive_curves(
        contrastive_epoch_data,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_contrastive_curves.png")
    )
    plot_lr_schedule(
        contrastive_epoch_data,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_lr_schedule.png")
    )
    plot_text_similarity_matrix(
        model, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_text_sim_matrix.png")
    )
    plot_generated_images(
        decoder, model, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_generated_images.png"), n=8
    )
    plot_retrieval_demo(
        model, test_set, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_retrieval_demo.png")
    )
    logger.info("All plots saved.")

    # ====================================================================
    # PHASE 3: LINEAR PROBE
    # ====================================================================
    logger.info("=" * 70)
    logger.info("PHASE 3: Linear Probe on Frozen Embeddings")
    logger.info("=" * 70)
    logger.info("This phase trains a lightweight linear classifier on top of")
    logger.info("frozen image embeddings to prove the contrastive representations")
    logger.info("are discriminative enough for direct classification.")

    num_classes = len(full_dataset.class_names)
    probe = LinearProbe(embed_dim=EMBED_DIM, num_classes=num_classes).to(DEVICE)
    logger.info(f"LinearProbe params: {count_parameters(probe):,} "
                f"(embed_dim={EMBED_DIM}, num_classes={num_classes})")

    # Freeze the main model entirely
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    probe_train_loader = DataLoader(train_set, batch_size=LINEAR_PROBE_BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_WORKERS,
                                    drop_last=True, pin_memory=(DEVICE == "cuda"))
    probe_val_loader = DataLoader(val_set, batch_size=LINEAR_PROBE_BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  drop_last=False, pin_memory=(DEVICE == "cuda"))
    probe_test_loader = DataLoader(test_set, batch_size=LINEAR_PROBE_BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS,
                                   drop_last=False, pin_memory=(DEVICE == "cuda"))

    probe_optimizer = torch.optim.Adam(probe.parameters(), lr=LINEAR_PROBE_LR)

    probe_csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_probe_epochs.csv")
    probe_csv = open(probe_csv_path, "w", newline="", encoding="utf-8")
    probe_csv_writer = csv.writer(probe_csv)
    probe_csv_writer.writerow(["epoch", "train_loss", "val_loss",
                                "train_acc", "val_acc", "epoch_time_s"])

    best_probe_val_acc = 0.0
    best_probe_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_best_probe.pt")
    probe_epoch_data = []
    phase3_start = time.time()

    for epoch in range(1, LINEAR_PROBE_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_linear_probe_epoch(
            probe, model, probe_train_loader, probe_optimizer, DEVICE
        )
        val_loss, val_acc = eval_linear_probe(
            probe, model, probe_val_loader, DEVICE
        )
        elapsed = time.time() - t0

        improved = ""
        if val_acc > best_probe_val_acc:
            best_probe_val_acc = val_acc
            torch.save(probe.state_dict(), best_probe_path)
            improved = " [BEST]"

        # ETA
        avg_epoch_time = (time.time() - phase3_start) / epoch
        eta_seconds = avg_epoch_time * (LINEAR_PROBE_EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_hr, eta_min = divmod(eta_min, 60)
        eta_str = (f"{eta_hr}h {eta_min}m {eta_sec}s" if eta_hr > 0
                   else f"{eta_min}m {eta_sec}s" if eta_min > 0
                   else f"{eta_sec}s")

        logger.info(
            f"Epoch {epoch:3d}/{LINEAR_PROBE_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Best Val Acc: {best_probe_val_acc:.4f} | "
            f"Time: {elapsed:.1f}s | ETA: {eta_str}{improved}"
        )

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc, "epoch_time": elapsed,
        }
        probe_epoch_data.append(row)
        probe_csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                   f"{train_acc:.6f}", f"{val_acc:.6f}",
                                   f"{elapsed:.2f}"])
        probe_csv.flush()

    probe_csv.close()
    logger.info(f"Phase 3 complete. Best val accuracy: {best_probe_val_acc:.4f}")

    # Reload best probe
    probe.load_state_dict(torch.load(best_probe_path, map_location=DEVICE,
                                     weights_only=True))
    logger.info("Loaded best linear probe checkpoint.")

    # Test set evaluation
    test_probe_loss, test_probe_acc = eval_linear_probe(
        probe, model, probe_test_loader, DEVICE
    )
    logger.info(f"Linear Probe Test Loss: {test_probe_loss:.4f}")
    logger.info(f"Linear Probe Test Accuracy: {test_probe_acc:.4f}")

    # Plot probe training curves
    plot_linear_probe_curves(
        probe_epoch_data,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_probe_curves.png")
    )
    logger.info("Linear probe training curves saved.")

    # ====================================================================
    # EXPANDED DEMOS (after all training phases)
    # ====================================================================
    logger.info("=" * 70)
    logger.info("EXPANDED DEMO RESULTS")
    logger.info("=" * 70)

    # Cross-modal arithmetic
    logger.info("-" * 40)
    logger.info("Demo: Cross-Modal Arithmetic")
    logger.info("Tests whether embed(img_X) - embed(text_X) + embed(text_Y)")
    logger.info("lands near the text embedding for class Y.")
    logger.info("-" * 40)
    cross_modal_arithmetic_demo(
        model, test_set, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_cross_modal_arithmetic.png"),
        logger
    )

    # Image-to-image retrieval
    logger.info("-" * 40)
    logger.info("Demo: Image-to-Image Retrieval")
    logger.info("Finds the most similar images in embedding space using cosine")
    logger.info("similarity, demonstrating that the embeddings cluster by class.")
    logger.info("-" * 40)
    image_to_image_retrieval_demo(
        model, test_set, DEVICE,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_i2i_retrieval.png"),
        logger, n_queries=4, top_k=3
    )

    # Per-class metrics + confusion matrix
    logger.info("-" * 40)
    logger.info("Demo: Per-Class Retrieval Metrics")
    logger.info("Computes Precision@1, @3, @5 for each class and produces")
    logger.info("a confusion matrix of I2T retrieval predictions.")
    logger.info("-" * 40)
    per_class_results = compute_per_class_metrics(
        model, test_loader, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_per_class_metrics.png"),
        logger
    )

    # t-SNE visualization
    logger.info("-" * 40)
    logger.info("Demo: t-SNE Embedding Visualization")
    logger.info("Visualizes the shared image-text embedding space in 2D,")
    logger.info("showing how image and text embeddings cluster together.")
    logger.info("-" * 40)
    plot_tsne_embeddings(
        model, test_loader, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_tsne.png"),
        max_samples=500
    )

    # Embedding interpolation through decoder
    logger.info("-" * 40)
    logger.info("Demo: Embedding Interpolation")
    logger.info("Smoothly interpolates between text embeddings of class pairs")
    logger.info("and generates images through the decoder, showing morphing.")
    logger.info("-" * 40)
    plot_embedding_interpolation(
        decoder, model, full_dataset.class_names, DEVICE, TEXT_MAX_LEN,
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_interpolation.png"),
        pairs=None, n_steps=8
    )

    logger.info("All expanded demos complete.")

    # ====================================================================
    # RESULTS JSON
    # ====================================================================
    results_json = {
        "config": config,
        "dataset": {
            "total_samples": len(full_dataset),
            "num_classes": len(full_dataset.class_names),
            "class_names": full_dataset.class_names,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
        },
        "model": {
            "image_encoder_params": count_parameters(model.image_encoder),
            "text_encoder_params": count_parameters(model.text_encoder),
            "decoder_params": count_parameters(decoder),
            "probe_params": count_parameters(probe),
            "total_params": (count_parameters(model) + count_parameters(decoder)
                             + count_parameters(probe)),
        },
        "phase1_contrastive": {
            "total_epochs_run": len(contrastive_epoch_data),
            "best_val_i2t_accuracy": best_i2t_acc,
            "test_i2t_accuracy": test_i2t,
            "final_train_loss": contrastive_epoch_data[-1]["train_loss"],
            "final_val_loss": contrastive_epoch_data[-1]["val_loss"],
            "epoch_history": contrastive_epoch_data,
        },
        "phase2_decoder": {
            "total_epochs_run": len(decoder_epoch_data),
            "best_val_mse": best_val_mse,
            "final_train_mse": decoder_epoch_data[-1]["train_mse"],
            "final_val_mse": decoder_epoch_data[-1]["val_mse"],
            "epoch_history": decoder_epoch_data,
        },
        "phase3_linear_probe": {
            "total_epochs_run": len(probe_epoch_data),
            "best_val_accuracy": best_probe_val_acc,
            "test_loss": test_probe_loss,
            "test_accuracy": test_probe_acc,
            "final_train_loss": probe_epoch_data[-1]["train_loss"],
            "final_val_loss": probe_epoch_data[-1]["val_loss"],
            "final_train_acc": probe_epoch_data[-1]["train_acc"],
            "final_val_acc": probe_epoch_data[-1]["val_acc"],
            "epoch_history": probe_epoch_data,
        },
        "expanded_demos": {
            "per_class_metrics": per_class_results if per_class_results else {},
        },
        "saved_files": {
            "best_contrastive_model": best_contrastive_path,
            "best_decoder_model": best_decoder_path,
            "best_probe_model": best_probe_path,
            "contrastive_csv": contrastive_csv_path,
            "decoder_csv": decoder_csv_path,
            "probe_csv": probe_csv_path,
            "log_file": os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}.log"),
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }

    results_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info(f"Results JSON saved to {results_path}")

    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"  Phase 1 best val I2T accuracy: {best_i2t_acc:.4f}")
    logger.info(f"  Phase 1 test I2T accuracy:     {test_i2t:.4f}")
    logger.info(f"  Phase 2 best val MSE:          {best_val_mse:.6f}")
    logger.info(f"  Phase 3 best val probe acc:    {best_probe_val_acc:.4f}")
    logger.info(f"  Phase 3 test probe accuracy:   {test_probe_acc:.4f}")
    logger.info(f"  All outputs saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
