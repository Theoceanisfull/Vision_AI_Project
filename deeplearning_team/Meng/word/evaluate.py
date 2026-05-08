"""
Evaluate a trained checkpoint on a dataset split.

Usage:
  env/python.exe evaluate.py --config configs/config_aslcitizen_full.yaml
  env/python.exe evaluate.py --config configs/config_wlasl100.yaml --split val
  env/python.exe evaluate.py --config configs/config.yaml --checkpoint checkpoints/wlasl2000/best.pth
"""

import argparse

import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ASLCitizenDataset, WLASLDataset
from src.model import build_model
from src.utils import accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/config.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path (default: reads from config)')
    parser.add_argument('--split',      default='test',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.checkpoint or \
        f"{cfg['training']['checkpoint_dir']}/best.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device    : {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split     : {args.split}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    num_classes = len(ckpt['label_to_idx'])

    # Build dataset matching the config type
    if cfg['data'].get('csv_dir'):
        ds = ASLCitizenDataset(
            csv_dir=cfg['data']['csv_dir'],
            video_dir=cfg['data']['video_dir'],
            split=args.split,
            num_frames=cfg['data']['num_frames'],
            resize=cfg['data']['resize'],
            crop=cfg['data']['crop'],
            augment=False,
            top_n=cfg['data'].get('top_n_classes'),
        )
    else:
        nslt_kwargs = {}
        if cfg['data'].get('nslt_path') and cfg['data'].get('class_list_path'):
            nslt_kwargs = {
                'nslt_path':       cfg['data']['nslt_path'],
                'class_list_path': cfg['data']['class_list_path'],
            }
        ds = WLASLDataset(
            json_path=cfg['data']['json_path'],
            video_dir=cfg['data']['video_dir'],
            split=args.split,
            num_frames=cfg['data']['num_frames'],
            resize=cfg['data']['resize'],
            crop=cfg['data']['crop'],
            augment=False,
            **nslt_kwargs,
        )

    print(f"Samples   : {len(ds)}  |  Classes: {ds.num_classes}\n")

    loader = DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
    )

    model = build_model(cfg['model']['backbone'], num_classes, dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    top1_sum = top5_sum = n = 0
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating"):
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(clips)
            t1, t5 = accuracy(logits, labels, topk=(1, 5))
            bs       = labels.size(0)
            top1_sum += t1 * bs
            top5_sum += t5 * bs
            n        += bs

    top1 = top1_sum / n
    top5 = top5_sum / n
    print(f"\nResults on [{args.split}] split  ({n} samples):")
    print(f"  Top-1 : {top1*100:.2f}%")
    print(f"  Top-5 : {top5*100:.2f}%")


if __name__ == '__main__':
    main()
