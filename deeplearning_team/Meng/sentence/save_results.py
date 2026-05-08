"""
Save training results for the sentence-level translation model.
Generates plots from history.csv and runs final BLEU evaluation.

Usage:
  env/python.exe sentence/save_results.py --config sentence/configs/config_how2sign.yaml
  env/python.exe sentence/save_results.py --config sentence/configs/config_how2sign.yaml --no-test
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import How2SignDataset, How2SignS3DDataset, H2SCollator
from src.model import build_model, build_tokenizer
from src.utils import compute_bleu


def load_history(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    epochs     = [int(r['epoch'])              for r in rows]
    tr_loss    = [float(r['train_loss'])       for r in rows]
    va_loss    = [float(r['val_loss'])         for r in rows]
    va_bleu    = [float(r['val_bleu'])         for r in rows]
    lr         = [float(r['lr'])               for r in rows]
    secs       = [float(r['epoch_secs'])       for r in rows]
    return epochs, tr_loss, va_loss, va_bleu, lr, secs


def run_test_bleu(cfg, ckpt_path, device, tokenizer, split='test'):
    print(f'Running {split} BLEU evaluation...')
    use_s3d    = cfg['model'].get('feature_type') == 's3d'
    DatasetCls = How2SignS3DDataset if use_s3d else How2SignDataset
    ds_kwargs  = dict(data_root=cfg['data']['data_root'],
                      split=split, num_frames=cfg['data']['num_frames'], augment=False)
    if not use_s3d:
        ds_kwargs['normalize_pose'] = cfg['data'].get('normalize_pose', False)
        ds_kwargs['use_velocity']   = cfg['data'].get('use_velocity', False)
    ds = DatasetCls(**ds_kwargs)
    collator = H2SCollator(tokenizer, max_tgt_len=cfg['training']['max_tgt_len'])
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

    num_beams = cfg['training'].get('num_beams', 4)
    max_new   = cfg['training'].get('max_tgt_len', 128)
    hyps, refs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  {split}'):
            kp   = batch['keypoints'].to(device,    non_blocking=True)
            mask = batch['padding_mask'].to(device, non_blocking=True)
            gen_ids = model.generate(
                keypoints=kp, padding_mask=mask,
                num_beams=num_beams, max_new_tokens=max_new,
            )
            hyps.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
            refs.extend(batch['texts'])

    bleu = compute_bleu(hyps, refs)
    print(f'  {split} BLEU-4: {bleu:.2f}  ({len(refs)} samples)')
    return bleu, hyps, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='sentence/configs/config_how2sign.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out',        default=None)
    parser.add_argument('--no-test',    action='store_true',
                        help='Skip test BLEU evaluation (plot from history only)')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    log_dir  = cfg['training']['log_dir']
    ckpt_dir = cfg['training']['checkpoint_dir']
    out_dir  = args.out or os.path.join('results', os.path.basename(ckpt_dir))
    ckpt_path = args.checkpoint or os.path.join(ckpt_dir, 'best.pth')

    os.makedirs(out_dir, exist_ok=True)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = build_tokenizer(cfg)

    history_path = os.path.join(log_dir, 'history.csv')
    epochs, tr_loss, va_loss, va_bleu, lr, secs = load_history(history_path)

    # Filter epochs where BLEU was evaluated (non-zero)
    bleu_epochs = [e for e, b in zip(epochs, va_bleu) if b > 0]
    bleu_vals   = [b for b in va_bleu if b > 0]
    best_bleu_val = max(bleu_vals) if bleu_vals else 0.0
    best_bleu_ep  = bleu_epochs[bleu_vals.index(best_bleu_val)] if bleu_vals else 0

    total_secs = sum(secs)
    h, rem = divmod(int(total_secs), 3600)
    m, s   = divmod(rem, 60)

    test_bleu = test_hyps = test_refs = None
    if not args.no_test:
        test_bleu, test_hyps, test_refs = run_test_bleu(cfg, ckpt_path, device, tokenizer)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'How2Sign SLT — {cfg["model"]["decoder_model"].split("/")[-1]}',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # 1. Train + val loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tr_loss, color='steelblue',  linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, va_loss, color='darkorange', linewidth=1.5, label='Val Loss')
    if best_bleu_ep:
        ax1.axvline(best_bleu_ep, color='green', linestyle='--', linewidth=1,
                    label=f'Best BLEU epoch {best_bleu_ep}')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Val BLEU-4
    ax2 = fig.add_subplot(gs[0, 1])
    if bleu_epochs:
        ax2.plot(bleu_epochs, bleu_vals, color='mediumseagreen', linewidth=1.5,
                 marker='o', markersize=4, label='Val BLEU-4')
    if test_bleu is not None:
        ax2.axhline(test_bleu, color='crimson', linestyle=':', linewidth=1.5,
                    label=f'Test BLEU-4 {test_bleu:.2f}')
    ax2.set_title('BLEU-4')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU-4')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Learning rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, lr, color='mediumpurple', linewidth=1.5)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('LR')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Summary bar: best val BLEU vs test BLEU
    ax4 = fig.add_subplot(gs[1, 1])
    labels_bar = [f'Val BLEU\n(epoch {best_bleu_ep})']
    vals_bar   = [best_bleu_val]
    colors_bar = ['steelblue']
    if test_bleu is not None:
        labels_bar.append('Test BLEU')
        vals_bar.append(test_bleu)
        colors_bar.append('darkorange')
    bars = ax4.bar(labels_bar, vals_bar, color=colors_bar, width=0.4)
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    ax4.set_title('BLEU-4 Summary')
    ax4.set_ylabel('BLEU-4')
    ax4.grid(True, alpha=0.3, axis='y')

    plot_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot -> {plot_path}')

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    mc  = cfg['model']
    summary = f"""How2Sign Sign Language Translation — Training Summary
{'='*55}

Model
  Decoder        : {mc['decoder_model']}
  Encoder layers : {mc['encoder_layers']}  d_model={mc['encoder_d_model']}  heads={mc['encoder_nhead']}
  Keypoint dim   : {mc['keypoint_dim']}
  Dropout        : {mc['dropout']}

Dataset
  Data root      : {cfg['data']['data_root']}
  Num frames     : {cfg['data']['num_frames']}
  Max train samp : {cfg['data'].get('max_train_samples', 'all')}

Training
  Epochs         : {len(epochs)}
  Batch size     : {cfg['training']['batch_size']}  (accum x{cfg['training']['accumulation_steps']} = effective {cfg['training']['batch_size']*cfg['training']['accumulation_steps']})
  Encoder LR     : {cfg['training']['learning_rate']}
  Decoder LR     : {cfg['training']['decoder_lr']}
  Total time     : {h}h {m}m {s}s
  Avg epoch time : {total_secs/len(epochs)/60:.1f} min

Results
  Best Val BLEU-4 : {best_bleu_val:.2f}  (epoch {best_bleu_ep})
  Test BLEU-4     : {f"{test_bleu:.2f}" if test_bleu is not None else "not evaluated"}

Checkpoint     : {ckpt_path}
"""
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f'Saved summary -> {summary_path}')
    print()
    print(summary)

    # Sample predictions
    if test_hyps:
        samples_path = os.path.join(out_dir, 'test_samples.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            f.write(f'Test BLEU-4: {test_bleu:.2f}\n{"="*60}\n\n')
            for ref, hyp in zip(test_refs[:50], test_hyps[:50]):
                f.write(f'REF: {ref}\nHYP: {hyp}\n\n')
        print(f'Saved test samples -> {samples_path}')


if __name__ == '__main__':
    main()
