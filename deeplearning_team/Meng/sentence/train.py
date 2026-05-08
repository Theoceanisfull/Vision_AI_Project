"""
Train Sign Language Translation model on How2Sign.

Usage:
  env/python.exe sentence/train.py --config sentence/configs/config_how2sign.yaml
  env/python.exe sentence/train.py --config sentence/configs/config_how2sign.yaml --resume checkpoints/how2sign/last.pth
"""

import argparse
import csv
import os
import time

os.environ.setdefault('TORCH_HOME', 'D:/torch_cache')

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import How2SignDataset, How2SignS3DDataset, H2SCollator
from src.model import build_model, build_tokenizer
from src.utils import compute_bleu, save_checkpoint, load_checkpoint


def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, accum_steps):
    model.train()
    total_loss = n = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f'Epoch {epoch:3d} [train]', leave=False)
    for step, batch in enumerate(pbar):
        kp   = batch['keypoints'].to(device,    non_blocking=True)
        mask = batch['padding_mask'].to(device, non_blocking=True)
        lbl  = batch['labels'].to(device,       non_blocking=True)

        with autocast():
            loss, _ = model(keypoints=kp, padding_mask=mask, labels=lbl)
            loss     = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()   # step per optimizer update, not per epoch

        bs          = kp.size(0)
        total_loss += loss.item() * accum_steps * bs
        n          += bs
        pbar.set_postfix(loss=f'{total_loss / n:.4f}')

    return total_loss / n


@torch.no_grad()
def val_epoch(model, loader, device):
    """Fast val loss via teacher forcing (runs every epoch)."""
    model.eval()
    total_loss = n = 0
    for batch in tqdm(loader, desc='         [val]  ', leave=False):
        kp   = batch['keypoints'].to(device,    non_blocking=True)
        mask = batch['padding_mask'].to(device, non_blocking=True)
        lbl  = batch['labels'].to(device,       non_blocking=True)
        with autocast():
            loss, _ = model(keypoints=kp, padding_mask=mask, labels=lbl)
        bs          = kp.size(0)
        total_loss += loss.item() * bs
        n          += bs
    return total_loss / n


@torch.no_grad()
def evaluate_bleu(model, loader, tokenizer, device, num_beams, max_new_tokens,
                  log_dir, epoch, max_bleu_samples=200):
    """Beam-search generation + BLEU on up to max_bleu_samples for speed."""
    model.eval()
    hyps, refs = [], []

    for batch in tqdm(loader, desc='         [BLEU] ', leave=False):
        if len(refs) >= max_bleu_samples:
            break
        kp   = batch['keypoints'].to(device,    non_blocking=True)
        mask = batch['padding_mask'].to(device, non_blocking=True)
        gen_ids = model.generate(
            keypoints=kp, padding_mask=mask,
            num_beams=num_beams, max_new_tokens=max_new_tokens,
        )
        hyps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
        refs.extend(batch['texts'])

    bleu = compute_bleu(hyps, refs)

    # Save sample predictions for qualitative review
    sample_path = os.path.join(log_dir, f'samples_epoch{epoch:03d}.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(f'Epoch {epoch}  |  Val BLEU-4: {bleu:.2f}\n')
        f.write('=' * 60 + '\n\n')
        for ref, hyp in zip(refs[:20], hyps[:20]):
            f.write(f'REF: {ref}\n')
            f.write(f'HYP: {hyp}\n\n')

    return bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sentence/configs/config_how2sign.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # Tokenizer ------------------------------------------------------------
    tokenizer = build_tokenizer(cfg)

    # Datasets -------------------------------------------------------------
    data_root = cfg['data']['data_root']
    use_s3d = cfg['model'].get('feature_type') == 's3d'
    DatasetCls = How2SignS3DDataset if use_s3d else How2SignDataset

    ds_kwargs = dict(data_root=data_root, num_frames=cfg['data']['num_frames'])
    if not use_s3d:
        ds_kwargs['normalize_pose'] = cfg['data'].get('normalize_pose', False)
        ds_kwargs['use_velocity']   = cfg['data'].get('use_velocity', False)

    train_ds = DatasetCls(
        split='train', augment=True,
        max_train_samples=cfg['data'].get('max_train_samples'),
        **ds_kwargs,
    )
    val_ds = DatasetCls(split='val', augment=False, **ds_kwargs)

    collator = H2SCollator(tokenizer, max_tgt_len=cfg['training']['max_tgt_len'])

    loader_kwargs = dict(
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        collate_fn=collator, pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)

    print(f'\nTrain: {len(train_ds)}  Val: {len(val_ds)}\n')

    # Model ----------------------------------------------------------------
    model = build_model(cfg).to(device)
    model.forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        cfg['model'].get('tgt_lang', 'en_XX')
    )

    # Collect visual encoder params (everything outside mBART)
    mbart_ids  = {id(p) for p in model.mbart.parameters()}
    enc_params = [p for p in model.parameters() if id(p) not in mbart_ids]
    dec_params = [p for p in model.mbart.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW([
        {'params': enc_params, 'lr': cfg['training']['learning_rate']},
        {'params': dec_params, 'lr': cfg['training']['decoder_lr']},
    ], weight_decay=cfg['training']['weight_decay'])

    total_steps  = (len(train_loader) // cfg['training']['accumulation_steps']
                    * cfg['training']['epochs'])
    warmup_steps = cfg['training'].get('warmup_steps', 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()

    ckpt_dir = cfg['training']['checkpoint_dir']
    log_dir  = cfg['training']['log_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    writer = SummaryWriter(log_dir)

    start_epoch = 0
    best_bleu   = 0.0
    if args.resume:
        start_epoch, best_bleu = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        print(f'Resumed from epoch {start_epoch}, best BLEU {best_bleu:.2f}')

    # CSV history ----------------------------------------------------------
    csv_path   = os.path.join(log_dir, 'history.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_bleu',
                             'lr', 'epoch_secs'])

    bleu_every = cfg['training'].get('bleu_eval_every', 2)
    num_beams  = cfg['training'].get('num_beams', 4)
    max_new    = cfg['training'].get('max_tgt_len', 128)
    accum      = cfg['training']['accumulation_steps']
    patience   = cfg['training'].get('early_stopping_patience', None)
    no_improve = 0   # epochs since last BLEU improvement

    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'BLEU-4':>7}  {'LR':>10}  {'Time':>8}")
    print('-' * 62)

    for epoch in range(start_epoch, cfg['training']['epochs']):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler,
                                 device, epoch, accum)
        v_loss     = val_epoch(model, val_loader, device)

        val_bleu = 0.0
        if (epoch + 1) % bleu_every == 0:
            val_bleu = evaluate_bleu(model, val_loader, tokenizer, device,
                                     num_beams, max_new, log_dir, epoch,
                                     max_bleu_samples=cfg['training'].get('bleu_max_samples', 200))

        epoch_secs = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        is_best    = val_bleu > best_bleu and val_bleu > 0
        if val_bleu > 0:
            no_improve = 0 if is_best else no_improve + 1
        best_bleu  = max(best_bleu, val_bleu)

        # TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': v_loss}, epoch)
        if val_bleu > 0:
            writer.add_scalar('BLEU-4/val', val_bleu, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        m, s     = divmod(int(epoch_secs), 60)
        bleu_str = f'{val_bleu:.2f}' if val_bleu > 0 else '  --  '
        print(f'{epoch:5d}  {train_loss:8.4f}  {v_loss:8.4f}  {bleu_str:>7}  '
              f'{current_lr:10.2e}  {m:3d}m{s:02d}s' + ('  *' if is_best else ''))

        csv_writer.writerow([epoch, f'{train_loss:.6f}', f'{v_loss:.6f}',
                             f'{val_bleu:.4f}', f'{current_lr:.2e}',
                             f'{epoch_secs:.1f}'])
        csv_file.flush()

        save_checkpoint({
            'epoch':      epoch + 1,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  scheduler.state_dict(),
            'scaler':     scaler.state_dict(),
            'best_bleu':  best_bleu,
            'cfg':        cfg,
        }, is_best=is_best, checkpoint_dir=ckpt_dir)

        if patience and no_improve >= patience:
            print(f'\nEarly stopping: BLEU did not improve for {patience} evaluations.')
            break

    csv_file.close()
    writer.close()
    print(f'\nDone. Best val BLEU-4: {best_bleu:.2f}')
    print(f'Best checkpoint : {ckpt_dir}/best.pth')
    print(f'TensorBoard     : tensorboard --logdir {log_dir}')


if __name__ == '__main__':
    main()
