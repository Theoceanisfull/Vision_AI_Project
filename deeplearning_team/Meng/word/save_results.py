"""
Save training results: plots + summary text.
Output goes to results/

Run:
  env\python.exe save_results.py
  env\python.exe save_results.py --history logs/history.csv --checkpoint checkpoints/best.pth
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ASLCitizenDataset, WLASLDataset
from src.model import build_model
from src.utils import accuracy


def load_history(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    epochs  = [int(r['epoch'])               for r in rows]
    tr_loss = [float(r['train_loss'])        for r in rows]
    tr_top1 = [float(r['train_top1']) * 100  for r in rows]
    tr_top5 = [float(r['train_top5']) * 100  for r in rows]
    va_top1 = [float(r['val_top1'])   * 100  for r in rows]
    va_top5 = [float(r['val_top5'])   * 100  for r in rows]
    lr      = [float(r['lr'])                for r in rows]
    secs    = [float(r['epoch_secs'])        for r in rows]
    return epochs, tr_loss, tr_top1, tr_top5, va_top1, va_top5, lr, secs


def run_test_eval(cfg, checkpoint_path, device):
    print("Running test evaluation...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    num_classes = len(ckpt['label_to_idx'])

    if cfg['data'].get('csv_dir'):
        ds = ASLCitizenDataset(
            csv_dir=cfg['data']['csv_dir'],
            video_dir=cfg['data']['video_dir'],
            split='test',
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
            split='test',
            num_frames=cfg['data']['num_frames'],
            resize=cfg['data']['resize'],
            crop=cfg['data']['crop'],
            augment=False,
            **nslt_kwargs,
        )
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'],
                        shuffle=False, num_workers=cfg['data']['num_workers'],
                        pin_memory=True)

    model = build_model(cfg['model']['backbone'], num_classes, dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    top1_sum = top5_sum = n = 0
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc='  Test'):
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(clips)
            t1, t5 = accuracy(logits, labels, topk=(1, 5))
            bs = labels.size(0)
            top1_sum += t1 * bs
            top5_sum += t5 * bs
            n        += bs

    test_top1 = top1_sum / n * 100
    test_top5 = top5_sum / n * 100
    print(f"  Test Top-1: {test_top1:.2f}%  |  Top-5: {test_top5:.2f}%  ({n} samples)")
    return test_top1, test_top5, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history',    default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--config',     default='configs/config.yaml')
    parser.add_argument('--out',        default=None)
    parser.add_argument('--no-test',    action='store_true',
                        help='Skip test evaluation (plot train/val only)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir  = cfg['training']['log_dir']
    ckpt_dir = cfg['training']['checkpoint_dir']
    out_dir  = args.out or cfg['training'].get('results_dir', 'results')

    # Derive dataset label for plot titles
    if cfg['data'].get('csv_dir'):
        top_n = cfg['data'].get('top_n_classes')
        dataset_label = f"ASL-Citizen-{top_n}" if top_n else "ASL-Citizen"
    else:
        nslt = cfg['data'].get('nslt_path', '')
        for n in ['100', '300', '1000', '2000']:
            if f'nslt_{n}' in nslt:
                dataset_label = f"WLASL-{n}"
                break
        else:
            dataset_label = None  # resolved below after num_classes is known
    history_path    = args.history    or os.path.join(log_dir,  'history.csv')
    checkpoint_path = args.checkpoint or os.path.join(ckpt_dir, 'best.pth')

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs, tr_loss, tr_top1, tr_top5, va_top1, va_top5, lr, secs = load_history(history_path)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    num_classes = len(ckpt['label_to_idx'])
    if dataset_label is None:
        dataset_label = f"WLASL-{num_classes}"
    num_params = sum(p.numel() for p in
                     build_model(cfg['model']['backbone'], num_classes,
                                 cfg['model']['dropout']).parameters()) / 1e6

    total_secs = sum(secs)
    h, rem = divmod(int(total_secs), 3600)
    m, s   = divmod(rem, 60)

    best_idx   = va_top1.index(max(va_top1))
    best_epoch = epochs[best_idx]
    best_top1  = va_top1[best_idx]
    best_top5  = va_top5[best_idx]

    if args.no_test:
        test_top1 = test_top5 = test_n = None
        print("Skipping test evaluation (--no-test).")
    else:
        test_top1, test_top5, test_n = run_test_eval(cfg, checkpoint_path, device)

    # -------------------------------------------------------------------------
    # Plots  (3 rows x 2 cols)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 14))
    fig.suptitle(f'{dataset_label} ASL Recognition — {cfg["model"]["backbone"]}',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tr_loss, color='steelblue', linewidth=1.5)
    ax1.axvline(best_epoch, color='green', linestyle='--', linewidth=1, label=f'Best epoch {best_epoch}')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Top-1 accuracy — train + val + test reference line
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, tr_top1, color='steelblue',  linewidth=1.5, label='Train Top-1', alpha=0.7)
    ax2.plot(epochs, va_top1, color='darkorange', linewidth=1.5, label='Val Top-1')
    if test_top1 is not None:
        ax2.axhline(test_top1, color='crimson', linestyle=':', linewidth=1.5,
                    label=f'Test Top-1 {test_top1:.2f}%')
    ax2.axvline(best_epoch, color='green', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch}')
    ax2.set_title('Top-1 Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Val top-5 + test reference line
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, va_top5, color='mediumseagreen', linewidth=1.5, label='Val Top-5')
    if test_top5 is not None:
        ax3.axhline(test_top5, color='crimson', linestyle=':', linewidth=1.5,
                    label=f'Test Top-5 {test_top5:.2f}%')
    ax3.axvline(best_epoch, color='green', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch}')
    ax3.set_title('Top-5 Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Learning rate schedule
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, lr, color='mediumpurple', linewidth=1.5)
    ax4.set_title('Learning Rate Schedule')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('LR')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # 5. Train / Val / Test comparison bar chart
    ax5 = fig.add_subplot(gs[2, :])
    last_epoch = epochs[-1]
    if test_top1 is not None:
        splits    = [f'Train\n(epoch {last_epoch})', f'Val\n(best epoch {best_epoch})', f'Test\n(best epoch {best_epoch})']
        top1_vals = [tr_top1[best_idx], best_top1, test_top1]
        top5_vals = [tr_top5[best_idx], best_top5, test_top5]
    else:
        splits    = [f'Train\n(epoch {last_epoch})', f'Val\n(best epoch {best_epoch})']
        top1_vals = [tr_top1[best_idx], best_top1]
        top5_vals = [tr_top5[best_idx], best_top5]
    x = list(range(len(splits)))
    width = 0.35
    bars1 = ax5.bar([i - width/2 for i in x], top1_vals, width, label='Top-1', color='steelblue')
    bars2 = ax5.bar([i + width/2 for i in x], top5_vals, width, label='Top-5', color='darkorange', alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(splits)
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Train / Val / Test Accuracy Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plot_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot -> {plot_path}')

    # -------------------------------------------------------------------------
    # Summary text
    # -------------------------------------------------------------------------
    summary = f"""{dataset_label} ASL Recognition — Training Summary
{'='*50}

Model
  Backbone       : {cfg['model']['backbone']}
  Parameters     : {num_params:.1f}M
  Dropout        : {cfg['model']['dropout']}
  Pretrained     : Kinetics-400

Dataset
  Classes        : {num_classes}
  Input frames   : {cfg['data']['num_frames']}
  Frame size     : {cfg['data']['crop']}x{cfg['data']['crop']}

Training
  Epochs         : {len(epochs)}
  Batch size     : {cfg['training']['batch_size']}
  Base LR        : {cfg['training']['learning_rate']}
  Head LR mult   : {cfg['training']['head_lr_multiplier']}x
  Weight decay   : {cfg['training']['weight_decay']}
  Label smooth   : {cfg['training']['label_smoothing']}
  Warmup epochs  : {cfg['training']['warmup_epochs']}
  Mixed precision: {cfg['training']['use_amp']}
  Total time     : {h}h {m}m {s}s
  Avg epoch time : {total_secs/len(epochs)/60:.1f} min

Results (best checkpoint — epoch {best_epoch})
  Val Top-1      : {best_top1:.2f}%
  Val Top-5      : {best_top5:.2f}%
  Train Top-1    : {tr_top1[best_idx]:.2f}%
  Train Loss     : {tr_loss[best_idx]:.4f}

Test Evaluation (best checkpoint, unseen data)
  Test Top-1     : {f"{test_top1:.2f}%  ({test_n} samples)" if test_top1 is not None else "not evaluated"}
  Test Top-5     : {f"{test_top5:.2f}%" if test_top5 is not None else "not evaluated"}

Checkpoints
  Best model     : {ckpt_dir}/best.pth
  Last model     : {ckpt_dir}/last.pth
"""

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f'Saved summary -> {summary_path}')
    print()
    print(summary)


if __name__ == '__main__':
    main()
