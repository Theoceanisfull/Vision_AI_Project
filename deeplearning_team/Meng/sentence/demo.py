"""
Interactive demo for the trained How2Sign translation model.

Usage:
  # 5 random test samples
  env/python.exe sentence/demo.py --config sentence/configs/config_how2sign.yaml

  # More samples
  env/python.exe sentence/demo.py --config sentence/configs/config_how2sign.yaml --n 20

  # Specific sentence by name
  env/python.exe sentence/demo.py --config sentence/configs/config_how2sign.yaml --sentence "-fZc293MpJk_0-1-rgb_front"

  # Use val split instead of test
  env/python.exe sentence/demo.py --config sentence/configs/config_how2sign.yaml --split val

  # Try a raw .npy file directly (no reference available)
  env/python.exe sentence/demo.py --config sentence/configs/config_how2sign.yaml --npy path/to/file.npy
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.dataset import How2SignDataset, How2SignS3DDataset, _resample
from src.model import build_model, build_tokenizer


def load_model(cfg, ckpt_path, device, tokenizer):
    model = build_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        cfg['model'].get('tgt_lang', 'en_XX')
    )
    return model.to(device).eval()


@torch.no_grad()
def translate_npy(npy_path, model, tokenizer, cfg, device):
    from src.dataset import _normalize_pose, _add_velocity
    num_frames     = cfg['data']['num_frames']
    num_beams      = cfg['training'].get('num_beams', 4)
    max_new        = cfg['training'].get('max_tgt_len', 128)
    normalize_pose = cfg['data'].get('normalize_pose', False)
    use_velocity   = cfg['data'].get('use_velocity', False)

    kp = np.load(npy_path).astype(np.float32)
    original_frames = len(kp)
    kp, valid = _resample(kp, num_frames)
    if normalize_pose:
        kp = _normalize_pose(kp)
    if use_velocity:
        kp = _add_velocity(kp)

    src  = torch.from_numpy(kp).unsqueeze(0).to(device)
    mask = torch.zeros(1, num_frames, dtype=torch.bool, device=device)
    mask[0, valid:] = True

    gen_ids = model.generate(
        keypoints=src, padding_mask=mask,
        num_beams=num_beams, max_new_tokens=max_new,
    )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text, original_frames


def print_result(i, sentence_name, ref, hyp, frames, width=70):
    print(f'\n{"─"*width}')
    print(f'[{i}] {sentence_name}')
    print(f'    Frames : {frames}')
    if ref:
        print(f'    REF    : {ref}')
    print(f'    HYP    : {hyp}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='sentence/configs/config_how2sign.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n',          type=int, default=5, help='Number of random samples')
    parser.add_argument('--sentence',   default=None, help='Specific SENTENCE_NAME to translate')
    parser.add_argument('--npy',        default=None, help='Path to a raw .npy keypoint file')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.checkpoint or os.path.join(cfg['training']['checkpoint_dir'], 'best.pth')
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device     : {device}')
    print(f'Checkpoint : {ckpt_path}')

    tokenizer = build_tokenizer(cfg)
    model     = load_model(cfg, ckpt_path, device, tokenizer)

    # ── raw .npy mode ────────────────────────────────────────────────────
    if args.npy:
        hyp, frames = translate_npy(args.npy, model, tokenizer, cfg, device)
        print_result(1, os.path.basename(args.npy), ref=None, hyp=hyp, frames=frames)
        print()
        return

    # ── dataset mode ─────────────────────────────────────────────────────
    use_s3d    = cfg['model'].get('feature_type') == 's3d'
    DatasetCls = How2SignS3DDataset if use_s3d else How2SignDataset
    ds_kwargs  = dict(data_root=cfg['data']['data_root'],
                      split=args.split, num_frames=cfg['data']['num_frames'], augment=False)
    if not use_s3d:
        ds_kwargs['normalize_pose'] = cfg['data'].get('normalize_pose', False)
        ds_kwargs['use_velocity']   = cfg['data'].get('use_velocity', False)
    ds = DatasetCls(**ds_kwargs)

    # Build name→index lookup
    name_to_idx = {
        os.path.splitext(os.path.basename(s['npy']))[0]: i
        for i, s in enumerate(ds.samples)
    }

    if args.sentence:
        if args.sentence not in name_to_idx:
            print(f'Sentence "{args.sentence}" not found in {args.split} split.')
            print('Available examples:', list(name_to_idx.keys())[:5], '...')
            return
        indices = [name_to_idx[args.sentence]]
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    print(f'\nSplit: {args.split}  |  Samples: {len(indices)}')

    num_beams = cfg['training'].get('num_beams', 4)
    max_new   = cfg['training'].get('max_tgt_len', 128)

    for rank, idx in enumerate(indices, 1):
        sample = ds.samples[idx]
        name   = os.path.splitext(os.path.basename(sample['npy']))[0]
        ref    = sample['text']

        hyp, frames = translate_npy(sample['npy'], model, tokenizer, cfg, device)
        print_result(rank, name, ref, hyp, frames)

    print(f'\n{"─"*70}')
    print(f'Done.  Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
