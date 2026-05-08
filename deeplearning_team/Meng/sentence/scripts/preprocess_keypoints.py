"""
Convert per-frame OpenPose JSON files into per-sentence .npy arrays.
Run once before training — takes ~1-2 hours for the full dataset.

Output: data/how2sign/{split}_keypoints_npy/{SENTENCE_NAME}.npy
        Each file is shape (T, D) float32
        D = 201  (pose 25 + left_hand 21 + right_hand 21) * 3 (x, y, confidence)
        D = 411  if --include_face is set (adds face 70 keypoints)

Usage:
  env/python.exe sentence/scripts/preprocess_keypoints.py
  env/python.exe sentence/scripts/preprocess_keypoints.py --splits train
  env/python.exe sentence/scripts/preprocess_keypoints.py --include_face
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

# OpenPose JSON field names
POSE_KEY   = 'pose_keypoints_2d'    # 25 keypoints * 3 = 75 values
FACE_KEY   = 'face_keypoints_2d'    # 70 keypoints * 3 = 210 values
HAND_L_KEY = 'hand_left_keypoints_2d'   # 21 * 3 = 63
HAND_R_KEY = 'hand_right_keypoints_2d'  # 21 * 3 = 63

# How2Sign video resolution (used to normalize pixel coords to [0, 1])
IMG_W = 1280.0
IMG_H = 720.0


def extract_frame(json_path, include_face):
    """Return flat float32 vector for one frame. All-zeros if no person detected."""
    with open(json_path) as f:
        data = json.load(f)

    n_kp  = 25 + 21 + 21 + (70 if include_face else 0)
    n_val = n_kp * 3
    vec   = np.zeros(n_val, dtype=np.float32)

    if not data.get('people'):
        return vec

    p     = data['people'][0]
    parts = []

    def add_keypoints(raw, w, h):
        for i in range(0, len(raw), 3):
            parts.append(raw[i]     / w)   # x normalized
            parts.append(raw[i + 1] / h)   # y normalized
            parts.append(raw[i + 2])        # confidence (already 0-1)

    add_keypoints(p.get(POSE_KEY,   []), IMG_W, IMG_H)
    if include_face:
        add_keypoints(p.get(FACE_KEY,   []), IMG_W, IMG_H)
    add_keypoints(p.get(HAND_L_KEY, []), IMG_W, IMG_H)
    add_keypoints(p.get(HAND_R_KEY, []), IMG_W, IMG_H)

    arr = np.array(parts, dtype=np.float32)
    vec[:len(arr)] = arr
    return vec


def process_split(split, data_root, include_face):
    json_root = os.path.join(data_root, f'{split}_2D_keypoints',
                             'openpose_output', 'json')
    out_dir   = os.path.join(data_root, f'{split}_keypoints_npy')
    os.makedirs(out_dir, exist_ok=True)

    sentences = sorted(os.listdir(json_root))
    skipped   = 0

    for sent_name in tqdm(sentences, desc=split):
        out_path = os.path.join(out_dir, f'{sent_name}.npy')
        if os.path.exists(out_path):
            continue  # resume-friendly

        sent_dir    = os.path.join(json_root, sent_name)
        frame_files = sorted(
            f for f in os.listdir(sent_dir) if f.endswith('_keypoints.json')
        )
        if not frame_files:
            skipped += 1
            continue

        frames = [
            extract_frame(os.path.join(sent_dir, ff), include_face)
            for ff in frame_files
        ]
        np.save(out_path, np.stack(frames))   # (T, D)

    done = len(sentences) - skipped
    print(f'  {split}: {done} saved, {skipped} skipped (no frames)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',    default='data/how2sign')
    parser.add_argument('--splits',       nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--include_face', action='store_true',
                        help='Include 70 face keypoints (D=411 instead of 201)')
    args = parser.parse_args()

    dim = (25 + 21 + 21 + (70 if args.include_face else 0)) * 3
    print(f'Keypoint dim per frame : {dim}')
    print(f'Output root            : {args.data_root}')

    for split in args.splits:
        process_split(split, args.data_root, args.include_face)

    print('Done.')


if __name__ == '__main__':
    main()
