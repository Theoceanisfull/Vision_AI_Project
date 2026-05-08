"""
Train R(2+1)D-18 on WLASL for word-level ASL recognition.

Usage:
  conda run python train.py
  conda run python train.py --config configs/config.yaml
  conda run python train.py --resume checkpoints/last.pth
"""

import argparse
import csv
import json
import os
import time

# Keep torch hub cache on D: drive
os.environ.setdefault("TORCH_HOME", "D:/torch_cache")

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import ASLCitizenDataset, WLASLDataset
from src.model import build_model, get_param_groups
from src.utils import (accuracy, get_cosine_schedule_with_warmup,
                       load_checkpoint, save_checkpoint)


# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, cfg):
    model.train()
    total_loss = top1_sum = top5_sum = n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for clips, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg['training']['use_amp']):
            logits = model(clips)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        t1, t5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item() * bs
        top1_sum   += t1 * bs
        top5_sum   += t5 * bs
        n          += bs
        pbar.set_postfix(loss=f"{total_loss/n:.4f}",
                         top1=f"{top1_sum/n:.3f}",
                         top5=f"{top5_sum/n:.3f}")

    return total_loss / n, top1_sum / n, top5_sum / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch, cfg):
    model.eval()
    total_loss = top1_sum = top5_sum = n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [val]  ", leave=False)
    for clips, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=cfg['training']['use_amp']):
            logits = model(clips)
            loss   = criterion(logits, labels)

        bs = labels.size(0)
        t1, t5 = accuracy(logits, labels, topk=(1, 5))
        total_loss += loss.item() * bs
        top1_sum   += t1 * bs
        top5_sum   += t5 * bs
        n          += bs
        pbar.set_postfix(loss=f"{total_loss/n:.4f}",
                         top1=f"{top1_sum/n:.3f}",
                         top5=f"{top5_sum/n:.3f}")

    return total_loss / n, top1_sum / n, top5_sum / n


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets ---------------------------------------------------------------
    if cfg['data'].get('csv_dir'):
        ds_kwargs = dict(
            csv_dir=cfg['data']['csv_dir'],
            video_dir=cfg['data']['video_dir'],
            num_frames=cfg['data']['num_frames'],
            resize=cfg['data']['resize'],
            crop=cfg['data']['crop'],
            top_n=cfg['data'].get('top_n_classes'),
        )
        train_ds = ASLCitizenDataset(split='train', augment=True,  **ds_kwargs)
        val_ds   = ASLCitizenDataset(split='val',   augment=False, **ds_kwargs)
    else:
        nslt_kwargs = {}
        if cfg['data'].get('nslt_path') and cfg['data'].get('class_list_path'):
            nslt_kwargs = {
                'nslt_path':       cfg['data']['nslt_path'],
                'class_list_path': cfg['data']['class_list_path'],
            }
        ds_kwargs = dict(
            json_path=cfg['data']['json_path'],
            video_dir=cfg['data']['video_dir'],
            num_frames=cfg['data']['num_frames'],
            resize=cfg['data']['resize'],
            crop=cfg['data']['crop'],
        )
        train_ds = WLASLDataset(split='train', augment=True,  **ds_kwargs, **nslt_kwargs)
        val_ds   = WLASLDataset(split='val',   augment=False, **ds_kwargs, **nslt_kwargs)

    num_classes = train_ds.num_classes
    print(f"\nClasses : {num_classes}")
    print(f"Train   : {len(train_ds)} samples")
    print(f"Val     : {len(val_ds)} samples\n")

    # Save label map for inference later
    os.makedirs(cfg['training']['checkpoint_dir'], exist_ok=True)
    with open(os.path.join(cfg['training']['checkpoint_dir'], 'label_map.json'), 'w') as f:
        json.dump(train_ds.idx_to_label, f, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    # Model ------------------------------------------------------------------
    model = build_model(
        backbone=cfg['model']['backbone'],
        num_classes=num_classes,
        dropout=cfg['model']['dropout'],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg['model']['backbone']}  ({param_count:.1f}M params)")

    # Loss -------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg['training']['label_smoothing']
    )

    # Optimizer (differential LR: head gets 10x backbone LR) ----------------
    param_groups = get_param_groups(
        model,
        base_lr=cfg['training']['learning_rate'],
        head_lr_multiplier=cfg['training']['head_lr_multiplier'],
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg['training']['weight_decay'],
    )

    # Scheduler: linear warmup + cosine decay --------------------------------
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg['training']['warmup_epochs'],
        total_epochs=cfg['training']['epochs'],
    )

    scaler  = GradScaler(enabled=cfg['training']['use_amp'])
    writer  = SummaryWriter(cfg['training']['log_dir'])

    start_epoch    = 0
    best_val_top1  = 0.0

    if args.resume:
        start_epoch, best_val_top1 = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )

    # CSV history log (append so resume preserves previous epochs)
    csv_path = os.path.join(cfg['training']['log_dir'], 'history.csv')
    os.makedirs(cfg['training']['log_dir'], exist_ok=True)
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'train_top1', 'train_top5',
                             'val_loss', 'val_top1', 'val_top5', 'lr', 'epoch_secs'])

    # Training loop ----------------------------------------------------------
    print(f"{'Epoch':>5}  {'TrLoss':>8} {'TrTop1':>7} {'TrTop5':>7}  "
          f"{'VaLoss':>8} {'VaTop1':>7} {'VaTop5':>7}  {'LR':>10}  {'Time':>8}")
    print("-" * 85)

    train_start = time.time()

    for epoch in range(start_epoch, cfg['training']['epochs']):
        epoch_start = time.time()
        tr_loss, tr_t1, tr_t5 = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, cfg)
        va_loss, va_t1, va_t5 = val_epoch(
            model, val_loader, criterion, device, epoch, cfg)
        epoch_secs = time.time() - epoch_start

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalars('Loss',  {'train': tr_loss, 'val': va_loss}, epoch)
        writer.add_scalars('Top-1', {'train': tr_t1,  'val': va_t1},   epoch)
        writer.add_scalars('Top-5', {'train': tr_t5,  'val': va_t5},   epoch)
        writer.add_scalar('LR', current_lr, epoch)

        is_best       = va_t1 > best_val_top1
        best_val_top1 = max(va_t1, best_val_top1)

        mins, secs = divmod(int(epoch_secs), 60)
        print(f"{epoch:5d}  {tr_loss:8.4f} {tr_t1:7.4f} {tr_t5:7.4f}  "
              f"{va_loss:8.4f} {va_t1:7.4f} {va_t5:7.4f}  {current_lr:10.2e}  {mins:3d}m{secs:02d}s"
              + ("  *" if is_best else ""))

        csv_writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_t1:.6f}", f"{tr_t5:.6f}",
                             f"{va_loss:.6f}", f"{va_t1:.6f}", f"{va_t5:.6f}",
                             f"{current_lr:.2e}", f"{epoch_secs:.1f}"])
        csv_file.flush()

        save_checkpoint(
            {
                'epoch':          epoch + 1,
                'model':          model.state_dict(),
                'optimizer':      optimizer.state_dict(),
                'scheduler':      scheduler.state_dict(),
                'scaler':         scaler.state_dict(),
                'best_val_top1':  best_val_top1,
                'label_to_idx':   train_ds.label_to_idx,
                'cfg':            cfg,
            },
            is_best=is_best,
            checkpoint_dir=cfg['training']['checkpoint_dir'],
        )

    csv_file.close()
    writer.close()

    total_secs = time.time() - train_start
    h, rem = divmod(int(total_secs), 3600)
    m, s   = divmod(rem, 60)

    print(f"\nTraining complete.")
    print(f"  Total time      : {h}h {m}m {s}s")
    print(f"  Best val top-1  : {best_val_top1*100:.2f}%")
    print(f"  Best checkpoint : {cfg['training']['checkpoint_dir']}/best.pth")
    print(f"  Epoch history   : {csv_path}")
    print(f"\nTensorBoard: tensorboard --logdir {cfg['training']['log_dir']}")


if __name__ == '__main__':
    main()
