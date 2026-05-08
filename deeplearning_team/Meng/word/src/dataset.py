import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# Kinetics-400 statistics (matches R2Plus1D-18 pretraining)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145,  0.216989]


def read_video_clip(video_path, frame_start, frame_end, num_frames):
    """Read num_frames uniformly sampled from [frame_start, frame_end] of a video.

    Returns numpy array (T, H, W, 3) uint8, or None on failure.
    frame_start/frame_end are 1-indexed as in WLASL JSON; frame_end=-1 means last frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(0, frame_start - 1)
    e = total if frame_end == -1 else min(frame_end, total)
    if e <= s:
        s, e = 0, total
    if e <= s:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    frames = []
    for _ in range(e - s):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return np.stack([frames[i] for i in indices])  # (T, H, W, 3)


class WLASLDataset(Dataset):
    def __init__(self, json_path, video_dir, split, num_frames=16,
                 resize=128, crop=112, augment=False,
                 nslt_path=None, class_list_path=None):
        self.num_frames = num_frames
        self.resize = resize
        self.crop = crop
        self.augment = augment

        self._video_index = self._build_index(video_dir)
        if nslt_path is not None and class_list_path is not None:
            self.samples, self.label_to_idx, self.idx_to_label = \
                self._parse_nslt(nslt_path, class_list_path, split)
        else:
            self.samples, self.label_to_idx, self.idx_to_label = self._parse(json_path, split)

        before = len(self.samples)
        self.samples = [s for s in self.samples if s['video_id'] in self._video_index]
        missing = before - len(self.samples)
        if missing:
            print(f"[{split}] Skipped {missing} samples with missing videos "
                  f"({len(self.samples)} remaining)")

    # ------------------------------------------------------------------
    def _build_index(self, video_dir):
        """Walk video_dir once and map video_id (stem) -> full path."""
        index = {}
        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    index[os.path.splitext(f)[0]] = os.path.join(root, f)
        return index

    def _parse_nslt(self, nslt_path, class_list_path, split):
        """Parse nslt_*.json + wlasl_class_list.txt for subset-specific training."""
        with open(class_list_path, encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f if l.strip()]
        full_idx_to_label = {}
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                full_idx_to_label[int(parts[0])] = parts[1]

        with open(nslt_path, encoding='utf-8') as f:
            nslt = json.load(f)

        # Collect only the class indices actually present in this nslt file
        used_indices = sorted({info['action'][0] for info in nslt.values()})
        idx_to_label = {i: full_idx_to_label[i] for i in used_indices if i in full_idx_to_label}
        label_to_idx = {g: i for i, g in idx_to_label.items()}

        samples = []
        for video_id, info in nslt.items():
            if info['subset'] != split:
                continue
            class_idx, frame_start, frame_end = info['action']
            samples.append({
                'video_id':    video_id,
                'label':       class_idx,
                'frame_start': frame_start,
                'frame_end':   frame_end,
            })
        return samples, label_to_idx, idx_to_label

    def _parse(self, json_path, split):
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        label_to_idx = {}
        idx_to_label = {}
        samples = []

        for entry in data:
            gloss = entry['gloss']
            if gloss not in label_to_idx:
                idx = len(label_to_idx)
                label_to_idx[gloss] = idx
                idx_to_label[idx] = gloss

            lbl = label_to_idx[gloss]
            for inst in entry['instances']:
                if inst.get('split') != split:
                    continue
                samples.append({
                    'video_id':    inst['video_id'],
                    'label':       lbl,
                    'frame_start': inst.get('frame_start', 1),
                    'frame_end':   inst.get('frame_end', -1),
                })

        return samples, label_to_idx, idx_to_label

    # ------------------------------------------------------------------
    def _transform(self, frames_np):
        """frames_np: (T, H, W, 3) uint8  ->  (3, T, H, W) float32 normalized."""
        T = frames_np.shape[0]
        tensors = [
            TF.to_tensor(frames_np[t])   # (3, H, W), float32 [0,1]
            for t in range(T)
        ]

        # Resize all frames to (resize x resize) — same target for spatial crop
        tensors = [TF.resize(t, [self.resize, self.resize],
                             interpolation=TF.InterpolationMode.BILINEAR)
                   for t in tensors]

        if self.augment:
            # Shared random crop across all frames
            i = random.randint(0, self.resize - self.crop)
            j = random.randint(0, self.resize - self.crop)
            tensors = [TF.crop(t, i, j, self.crop, self.crop) for t in tensors]

            # Horizontal flip for handedness robustness (standard in WLASL literature)
            if random.random() < 0.5:
                tensors = [TF.hflip(t) for t in tensors]

            # Color jitter — same params applied to all frames (temporal consistency)
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            tensors = [TF.adjust_brightness(t, brightness) for t in tensors]
            tensors = [TF.adjust_contrast(t, contrast)     for t in tensors]
            tensors = [TF.adjust_saturation(t, saturation) for t in tensors]
        else:
            tensors = [TF.center_crop(t, self.crop) for t in tensors]

        mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1)
        std  = torch.tensor(KINETICS_STD).view(3, 1, 1)
        tensors = [(t - mean) / std for t in tensors]

        return torch.stack(tensors, dim=1)  # (3, T, H, W)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = self._video_index[s['video_id']]
        frames = read_video_clip(path, s['frame_start'], s['frame_end'], self.num_frames)

        if frames is None:
            clip = torch.zeros(3, self.num_frames, self.crop, self.crop)
        else:
            clip = self._transform(frames)

        return clip, torch.tensor(s['label'], dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.label_to_idx)


class ASLCitizenDataset(Dataset):
    """ASL-Citizen dataset (CSV-based, full videos, no frame annotations)."""

    def __init__(self, csv_dir, video_dir, split, num_frames=16,
                 resize=128, crop=112, augment=False, top_n=None):
        self.num_frames = num_frames
        self.resize = resize
        self.crop = crop
        self.augment = augment

        self._video_index = self._build_index(video_dir)
        self.samples, self.label_to_idx, self.idx_to_label = \
            self._parse(csv_dir, split, top_n)

        before = len(self.samples)
        self.samples = [s for s in self.samples if s['video_file'] in self._video_index]
        missing = before - len(self.samples)
        if missing:
            print(f"[{split}] Skipped {missing} samples with missing videos "
                  f"({len(self.samples)} remaining)")

    def _build_index(self, video_dir):
        index = {}
        for f in os.listdir(video_dir):
            if f.lower().endswith('.mp4'):
                index[f] = os.path.join(video_dir, f)
        return index

    def _parse(self, csv_dir, split, top_n):
        import csv as _csv

        def read_csv(path):
            with open(path, encoding='utf-8') as f:
                return list(_csv.DictReader(f))

        train_rows = read_csv(os.path.join(csv_dir, 'train.csv'))

        # Class list always derived from train to keep val/test consistent
        counts = {}
        for r in train_rows:
            counts[r['Gloss']] = counts.get(r['Gloss'], 0) + 1
        if top_n is not None:
            top_classes = sorted(counts, key=lambda g: -counts[g])[:top_n]
        else:
            top_classes = list(counts.keys())
        top_classes_sorted = sorted(top_classes)
        label_to_idx = {g: i for i, g in enumerate(top_classes_sorted)}
        idx_to_label = {i: g for g, i in label_to_idx.items()}

        split_file = {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}[split]
        rows = read_csv(os.path.join(csv_dir, split_file))
        samples = []
        for r in rows:
            if r['Gloss'] not in label_to_idx:
                continue
            samples.append({
                'video_file': r['Video file'],
                'label':      label_to_idx[r['Gloss']],
            })
        return samples, label_to_idx, idx_to_label

    def _transform(self, frames_np):
        T = frames_np.shape[0]
        tensors = [TF.to_tensor(frames_np[t]) for t in range(T)]
        tensors = [TF.resize(t, [self.resize, self.resize],
                             interpolation=TF.InterpolationMode.BILINEAR)
                   for t in tensors]
        if self.augment:
            i = random.randint(0, self.resize - self.crop)
            j = random.randint(0, self.resize - self.crop)
            tensors = [TF.crop(t, i, j, self.crop, self.crop) for t in tensors]
            if random.random() < 0.5:
                tensors = [TF.hflip(t) for t in tensors]
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            tensors = [TF.adjust_brightness(t, brightness) for t in tensors]
            tensors = [TF.adjust_contrast(t, contrast)     for t in tensors]
            tensors = [TF.adjust_saturation(t, saturation) for t in tensors]
        else:
            tensors = [TF.center_crop(t, self.crop) for t in tensors]
        mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1)
        std  = torch.tensor(KINETICS_STD).view(3, 1, 1)
        tensors = [(t - mean) / std for t in tensors]
        return torch.stack(tensors, dim=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = self._video_index[s['video_file']]
        frames = read_video_clip(path, 1, -1, self.num_frames)
        if frames is None:
            clip = torch.zeros(3, self.num_frames, self.crop, self.crop)
        else:
            clip = self._transform(frames)
        return clip, torch.tensor(s['label'], dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.label_to_idx)
