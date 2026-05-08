"""
Evaluate a trained checkpoint on How2Sign using BLEU-4.

Usage:
  env/python.exe sentence/evaluate.py --config sentence/configs/config_how2sign.yaml
  env/python.exe sentence/evaluate.py --config sentence/configs/config_how2sign.yaml --split test
  env/python.exe sentence/evaluate.py --config sentence/configs/config_how2sign.yaml --checkpoint checkpoints/how2sign/best.pth
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import How2SignDataset, How2SignS3DDataset, H2SCollator
from src.model import build_model, build_tokenizer
from src.utils import compute_bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='sentence/configs/config_how2sign.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num_beams',  type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ckpt_path  = args.checkpoint or f"{cfg['training']['checkpoint_dir']}/best.pth"
    num_beams  = args.num_beams  or cfg['training'].get('num_beams', 4)
    max_new    = cfg['training'].get('max_tgt_len', 128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Checkpoint : {ckpt_path}')
    print(f'Split      : {args.split}')
    print(f'Num beams  : {num_beams}')

    tokenizer = build_tokenizer(cfg)

    use_s3d    = cfg['model'].get('feature_type') == 's3d'
    DatasetCls = How2SignS3DDataset if use_s3d else How2SignDataset
    ds_kwargs  = dict(data_root=cfg['data']['data_root'],
                      split=args.split, num_frames=cfg['data']['num_frames'], augment=False)
    if not use_s3d:
        ds_kwargs['normalize_pose'] = cfg['data'].get('normalize_pose', False)
        ds_kwargs['use_velocity']   = cfg['data'].get('use_velocity', False)
    ds = DatasetCls(**ds_kwargs)
    collator = H2SCollator(tokenizer, max_tgt_len=max_new)
    loader   = DataLoader(ds, batch_size=cfg['training']['batch_size'],
                          shuffle=False, num_workers=cfg['data']['num_workers'],
                          collate_fn=collator, pin_memory=True)

    model = build_model(cfg)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        cfg['model'].get('tgt_lang', 'en_XX')
    )
    model = model.to(device).eval()

    hyps, refs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            kp   = batch['keypoints'].to(device,    non_blocking=True)
            mask = batch['padding_mask'].to(device, non_blocking=True)
            gen_ids = model.generate(
                keypoints=kp, padding_mask=mask,
                num_beams=num_beams, max_new_tokens=max_new,
            )
            hyps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(batch['texts'])

    bleu = compute_bleu(hyps, refs)
    print(f'\nResults on [{args.split}] split ({len(refs)} samples):')
    print(f'  BLEU-4 : {bleu:.2f}')
    print('\nSample predictions:')
    for i in range(min(5, len(hyps))):
        print(f'  Ref : {refs[i]}')
        print(f'  Hyp : {hyps[i]}')
        print()


if __name__ == '__main__':
    main()
