import math
import os

import torch
import torch.optim.lr_scheduler as lr_sched


def accuracy(output, target, topk=(1, 5)):
    """Return top-k accuracy as fractions (not percentages)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / batch_size).item())
        return results


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_sched.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "last.pth")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)
        print(f"  ** New best saved -> {best_path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    epoch = ckpt.get('epoch', 0)
    best  = ckpt.get('best_val_top1', 0.0)
    print(f"  Resumed from epoch {epoch}, best val top-1: {best:.4f}")
    return epoch, best
