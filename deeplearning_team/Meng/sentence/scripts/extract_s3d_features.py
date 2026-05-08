"""
Extract S3D (Kinetics-400 pretrained) features from How2Sign video clips.
Run once before training — produces one .npy file per clip.

Output: data/how2sign/{split}_s3d_features/{SENTENCE_NAME}.npy
        Shape: (T', 1024) float32  where T' = ceil(T_frames / 8), max 32

Usage:
  env/python.exe sentence/scripts/extract_s3d_features.py
  env/python.exe sentence/scripts/extract_s3d_features.py --splits test val
  env/python.exe sentence/scripts/extract_s3d_features.py --data_root data/how2sign
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import s3d, S3D_Weights
from tqdm import tqdm

MAX_FRAMES = 256   # sample longer clips uniformly to this length
MIN_FRAMES = 16    # S3D needs at least ~8 frames; pad shorter clips by repeating
IMG_SIZE   = 224
# Kinetics-400 normalisation (same as S3D_Weights.KINETICS400_V1 transform)
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3, 1, 1, 1)


def load_video_frames(video_path, max_frames=MAX_FRAMES):
    """Load and uniformly sample frames from an MP4. Returns (C, T, H, W) float32 in [0,1]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    frames = np.stack(frames, axis=0)   # (T, H, W, C)

    # Uniform sampling if too long
    T = len(frames)
    if T > max_frames:
        idx    = np.linspace(0, T - 1, max_frames, dtype=int)
        frames = frames[idx]

    # Repeat-pad if too short for S3D's temporal pooling
    if len(frames) < MIN_FRAMES:
        reps   = (MIN_FRAMES + len(frames) - 1) // len(frames)
        frames = np.tile(frames, (reps, 1, 1, 1))[:MIN_FRAMES]

    # (T, H, W, C) -> (C, T, H, W), float32 [0, 1]
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return frames   # (3, T, H, W)


@torch.no_grad()
def extract_features(frames_chw, model, device):
    """
    frames_chw : (3, T, H, W) float32 tensor in [0,1]
    returns    : (T', 1024) float32 numpy array
    """
    x = frames_chw.to(device)

    # Normalise
    mean = MEAN.to(device)
    std  = STD.to(device)
    x    = (x - mean) / std          # (3, T, H, W)
    x    = x.unsqueeze(0)            # (1, 3, T, H, W)

    feats = model.features(x)        # (1, 1024, T', H', W')
    feats = feats.mean(dim=[3, 4])   # spatial avg pool -> (1, 1024, T')
    feats = feats.squeeze(0).T       # (T', 1024)
    return feats.cpu().float().numpy()


def process_split(split, data_root, model, device):
    vid_dir = os.path.join(data_root, f'{split}_rgb_front_clips', 'raw_videos')
    out_dir = os.path.join(data_root, f'{split}_s3d_features')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(vid_dir):
        print(f'  Video dir not found: {vid_dir} — skipping {split}')
        return

    videos  = sorted(f for f in os.listdir(vid_dir) if f.endswith('.mp4'))
    skipped = 0

    for fname in tqdm(videos, desc=split):
        name     = fname[:-4]   # strip .mp4
        out_path = os.path.join(out_dir, f'{name}.npy')
        if os.path.exists(out_path):
            continue

        try:
            frames = load_video_frames(os.path.join(vid_dir, fname))
            if frames is None:
                skipped += 1
                continue
            feats = extract_features(frames, model, device)
            np.save(out_path, feats)
        except Exception as e:
            skipped += 1
            tqdm.write(f'  SKIP {fname}: {e}')

    done = len(videos) - skipped
    print(f'  {split}: {done} saved, {skipped} skipped')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/how2sign')
    parser.add_argument('--splits',    nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Data root  : {args.data_root}')
    print(f'Max frames : {MAX_FRAMES}  (S3D temporal output: ~{MAX_FRAMES // 8} tokens)')

    weights = S3D_Weights.KINETICS400_V1
    model   = s3d(weights=weights).to(device).eval()
    # We only need the feature layers, not the classifier
    for p in model.parameters():
        p.requires_grad = False

    for split in args.splits:
        process_split(split, args.data_root, model, device)

    print('Done.')


if __name__ == '__main__':
    main()
