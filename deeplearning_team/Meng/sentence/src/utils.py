import os
import shutil

import sacrebleu
import torch


def compute_bleu(hypotheses, references):
    """BLEU-4 via sacrebleu. Both args are lists of strings. Returns score 0-100."""
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    last = os.path.join(checkpoint_dir, 'last.pth')
    torch.save(state, last)
    if is_best:
        shutil.copyfile(last, os.path.join(checkpoint_dir, 'best.pth'))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer  and 'optimizer'  in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler  and 'scheduler'  in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    if scaler     and 'scaler'     in ckpt: scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0), ckpt.get('best_bleu', 0.0)
