"""
How2Sign sentence-level sign language translation dataset.

Expects pre-extracted .npy keypoint files (run preprocess_keypoints.py first).

Each .npy file: (T, D) float32
  D = 201  (pose 25 + left_hand 21 + right_hand 21) * 3
  T = variable number of frames (resampled to num_frames at load time)
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

KEYPOINT_DIM = 201   # default; 411 if include_face=True

# BODY_25 pose joint indices (x at joint*3, y at joint*3+1, conf at joint*3+2)
_RSHOULDER_X, _RSHOULDER_Y = 6,  7    # joint 2
_LSHOULDER_X, _LSHOULDER_Y = 15, 16   # joint 5


def _resample(kp, num_frames):
    """Uniformly resample keypoint array to num_frames. Returns (kp, valid_len)."""
    T = len(kp)
    if T == 0:
        dim = kp.shape[-1] if kp.ndim == 2 else KEYPOINT_DIM
        return np.zeros((num_frames, dim), dtype=np.float32), 0
    if T >= num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
        return kp[idx], num_frames
    pad = np.zeros((num_frames - T, kp.shape[1]), dtype=np.float32)
    return np.vstack([kp, pad]), T


def _normalize_pose(kp: np.ndarray) -> np.ndarray:
    """Normalize all (x,y) coords relative to shoulder midpoint and shoulder width.

    Makes features invariant to signer position in frame and distance from camera.
    Falls back to no-op on frames where both shoulders are undetected (conf~0).
    kp : (T, D) float32 — returns a new array of the same shape.
    """
    out = kp.copy()
    rx, ry = kp[:, _RSHOULDER_X], kp[:, _RSHOULDER_Y]
    lx, ly = kp[:, _LSHOULDER_X], kp[:, _LSHOULDER_Y]
    cx = (rx + lx) / 2
    cy = (ry + ly) / 2
    sw = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

    # Only normalize frames where both shoulders are visible
    valid    = sw > 0.01                          # shoulder width > 1% of image width
    sw_safe  = np.where(valid, sw, 1.0)

    out[:, 0::3] = np.where(valid[:, None],
                             (kp[:, 0::3] - cx[:, None]) / sw_safe[:, None],
                             kp[:, 0::3])
    out[:, 1::3] = np.where(valid[:, None],
                             (kp[:, 1::3] - cy[:, None]) / sw_safe[:, None],
                             kp[:, 1::3])
    return out


def _add_velocity(kp: np.ndarray) -> np.ndarray:
    """Append frame-to-frame delta as additional features: (T, D) -> (T, 2*D)."""
    vel = np.zeros_like(kp)
    vel[1:] = kp[1:] - kp[:-1]
    return np.concatenate([kp, vel], axis=-1)


class How2SignDataset(Dataset):
    """
    Parameters
    ----------
    data_root         : root of How2Sign data (e.g. 'data/how2sign')
    split             : 'train' | 'val' | 'test'
    num_frames        : fixed sequence length fed to the encoder
    max_train_samples : optional cap on training set size for quick experiments
    augment           : light temporal jitter + keypoint dropout (train only)
    normalize_pose    : shoulder-relative coordinate normalization
    use_velocity      : append frame-to-frame delta features (doubles feature dim)
    """

    def __init__(self, data_root, split, num_frames=128,
                 max_train_samples=None, augment=False,
                 normalize_pose=False, use_velocity=False):
        self.num_frames     = num_frames
        self.augment        = augment
        self.normalize_pose = normalize_pose
        self.use_velocity   = use_velocity

        csv_path = os.path.join(
            data_root, 'How2Sign', 'sentence_level', split,
            'text', 'en', 'raw_text', 're_aligned',
            f'how2sign_realigned_{split}.csv',
        )
        npy_dir = os.path.join(data_root, f'{split}_keypoints_npy')

        df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        self.samples = []
        for _, row in df.iterrows():
            name = str(row['SENTENCE_NAME']).strip()
            text = str(row['SENTENCE']).strip()
            npy  = os.path.join(npy_dir, f'{name}.npy')
            if os.path.exists(npy) and text:
                self.samples.append({'npy': npy, 'text': text})

        if max_train_samples is not None and split == 'train':
            self.samples = self.samples[:max_train_samples]

        print(f'[How2Sign/{split}] {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s  = self.samples[idx]
        kp = np.load(s['npy']).astype(np.float32)
        kp, valid = _resample(kp, self.num_frames)

        if self.normalize_pose:
            kp = _normalize_pose(kp)

        if self.augment and valid > 4:
            # Temporal jitter
            shift = np.random.randint(0, max(1, valid // 8))
            kp    = np.roll(kp, shift, axis=0)
            # Keypoint dropout: randomly zero out 10% of joints per frame
            if np.random.rand() < 0.5:
                mask = np.random.rand(*kp.shape) > 0.10
                kp   = kp * mask

        if self.use_velocity:
            kp = _add_velocity(kp)

        src          = torch.from_numpy(kp)
        padding_mask = torch.zeros(self.num_frames, dtype=torch.bool)
        padding_mask[valid:] = True   # True = ignore padding in attention

        return {'keypoints': src, 'padding_mask': padding_mask, 'text': s['text']}


class How2SignS3DDataset(Dataset):
    """
    How2Sign dataset backed by pre-extracted S3D features (run extract_s3d_features.py first).

    Each .npy file is (T', 1024) — S3D temporal tokens, already pooled spatially.
    num_frames here is the number of S3D tokens (default 32, matching MAX_FRAMES//8=256//8).

    Parameters
    ----------
    data_root         : root of How2Sign data (e.g. 'data/how2sign')
    split             : 'train' | 'val' | 'test'
    num_frames        : fixed number of S3D temporal tokens (pad/resample to this)
    max_train_samples : optional cap for quick experiments
    augment           : light temporal jitter (train only)
    """

    def __init__(self, data_root, split, num_frames=32,
                 max_train_samples=None, augment=False):
        self.num_frames = num_frames
        self.augment    = augment

        csv_path = os.path.join(
            data_root, 'How2Sign', 'sentence_level', split,
            'text', 'en', 'raw_text', 're_aligned',
            f'how2sign_realigned_{split}.csv',
        )
        feat_dir = os.path.join(data_root, f'{split}_s3d_features')

        df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        self.samples = []
        for _, row in df.iterrows():
            name = str(row['SENTENCE_NAME']).strip()
            text = str(row['SENTENCE']).strip()
            npy  = os.path.join(feat_dir, f'{name}.npy')
            if os.path.exists(npy) and text:
                self.samples.append({'npy': npy, 'text': text})

        if max_train_samples is not None and split == 'train':
            self.samples = self.samples[:max_train_samples]

        print(f'[How2Sign-S3D/{split}] {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        feat = np.load(s['npy']).astype(np.float32)   # (T', 1024)

        feat, valid = _resample(feat, self.num_frames)

        if self.augment and valid > 2:
            shift = np.random.randint(0, max(1, valid // 4))
            feat  = np.roll(feat, shift, axis=0)

        src          = torch.from_numpy(feat)
        padding_mask = torch.zeros(self.num_frames, dtype=torch.bool)
        padding_mask[valid:] = True

        return {'keypoints': src, 'padding_mask': padding_mask, 'text': s['text']}


class H2SCollator:
    """Collates dataset items into batched tensors and tokenized labels."""

    def __init__(self, tokenizer, max_tgt_len=128):
        self.tokenizer   = tokenizer
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch):
        keypoints    = torch.stack([b['keypoints']    for b in batch])
        padding_mask = torch.stack([b['padding_mask'] for b in batch])
        texts        = [b['text'] for b in batch]

        tok = self.tokenizer(
            text_target=texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_tgt_len,
        )
        label_ids = tok['input_ids'].clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            'keypoints':    keypoints,
            'padding_mask': padding_mask,
            'labels':       label_ids,
            'texts':        texts,
        }
