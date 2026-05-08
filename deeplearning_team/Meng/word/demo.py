"""
Run inference on a video file and save an annotated GIF or MP4.

The ground truth label is extracted from the filename if it follows
ASL-Citizen naming convention: <id>-GLOSS.mp4

Usage:
  env/python.exe demo.py --checkpoint checkpoints/aslcitizen_full/best.pth --video path/to/video.mp4
  env/python.exe demo.py --checkpoint checkpoints/aslcitizen_full/best.pth --video path/to/video.mp4 --out demo.gif --top 5
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.dataset import read_video_clip
from src.model import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    label_to_idx = ckpt['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    cfg = ckpt.get('cfg', {})
    backbone = cfg.get('model', {}).get('backbone', 'r2plus1d_18')

    model = build_model(backbone, num_classes, dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    return model, idx_to_label, cfg


def preprocess(frames_np, resize=128, crop=112):
    """frames_np: (T, H, W, 3) uint8  ->  (1, 3, T, H, W) float32 tensor."""
    import torchvision.transforms.functional as TF
    MEAN = [0.43216, 0.394666, 0.37645]
    STD  = [0.22803, 0.22145,  0.216989]

    T = frames_np.shape[0]
    tensors = [TF.to_tensor(frames_np[t]) for t in range(T)]
    tensors = [TF.resize(t, [resize, resize],
                         interpolation=TF.InterpolationMode.BILINEAR) for t in tensors]
    tensors = [TF.center_crop(t, crop) for t in tensors]
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    tensors = [(t - mean) / std for t in tensors]
    clip = torch.stack(tensors, dim=1).unsqueeze(0)   # (1, 3, T, H, W)
    return clip


def extract_ground_truth(video_path):
    """Extract gloss from ASL-Citizen filename: <id>-GLOSS.mp4 -> GLOSS."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if '-' in stem:
        return stem.split('-', 1)[1].upper()
    return None


def read_all_frames(video_path, max_display=60):
    """Read all frames from video for display (capped to avoid huge GIFs)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    # Subsample if too many frames
    if len(frames) > max_display:
        indices = np.linspace(0, len(frames) - 1, max_display, dtype=int)
        frames = [frames[i] for i in indices]
    return frames


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

PANEL_H   = 110   # height of the bottom annotation panel
DISP_W    = 400   # display width for video frames
BAR_MAX_W = 180   # max width of confidence bars


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate_frame(frame_rgb, predictions, ground_truth, frame_idx, total_frames):
    """
    predictions: list of (label, confidence) sorted by confidence desc
    Returns annotated PIL image.
    """
    h, w = frame_rgb.shape[:2]
    scale = DISP_W / w
    new_h = int(h * scale)
    img = Image.fromarray(frame_rgb).resize((DISP_W, new_h), Image.BILINEAR)

    # Create canvas: video on top, annotation panel below
    canvas = Image.new('RGB', (DISP_W, new_h + PANEL_H), (20, 20, 20))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font_lg = get_font(17)
    font_sm = get_font(13)

    top_label, top_conf = predictions[0]
    is_correct = ground_truth and top_label.upper() == ground_truth.upper()

    # Header bar color: green if correct, red if wrong, gray if unknown
    if ground_truth is None:
        header_color = (50, 50, 180)
    elif is_correct:
        header_color = (30, 140, 60)
    else:
        header_color = (180, 40, 40)

    draw.rectangle([0, new_h, DISP_W, new_h + 28], fill=header_color)

    pred_text = f"Pred: {top_label}  ({top_conf*100:.1f}%)"
    draw.text((8, new_h + 5), pred_text, font=font_lg, fill=(255, 255, 255))

    if ground_truth:
        gt_text = f"GT: {ground_truth}"
        gt_color = (100, 255, 100) if is_correct else (255, 120, 120)
        draw.text((DISP_W - 130, new_h + 5), gt_text, font=font_lg, fill=gt_color)

    # Top-N confidence bars
    y = new_h + 34
    for label, conf in predictions:
        bar_w = int(conf * BAR_MAX_W)
        draw.rectangle([8, y + 2, 8 + bar_w, y + 13], fill=(80, 160, 240))
        draw.text((8 + BAR_MAX_W + 6, y), f"{label}  {conf*100:.1f}%", font=font_sm,
                  fill=(220, 220, 220))
        y += 17

    # Frame counter
    draw.text((DISP_W - 60, new_h + PANEL_H - 16),
              f"{frame_idx+1}/{total_frames}", font=font_sm, fill=(150, 150, 150))

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/aslcitizen_full/best.pth')
    parser.add_argument('--video',      required=True)
    parser.add_argument('--out',        default=None,
                        help='Output path (.gif or .mp4). Default: <video_stem>_demo.gif')
    parser.add_argument('--top',        type=int, default=3,
                        help='Number of top predictions to show (default: 3)')
    parser.add_argument('--fps',        type=int, default=10,
                        help='Output GIF/MP4 frame rate (default: 10)')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--resize',     type=int, default=128)
    parser.add_argument('--crop',       type=int, default=112)
    args = parser.parse_args()

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.out = f"{stem}_demo.gif"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Video      : {args.video}")

    # Load model
    model, idx_to_label, cfg = load_model(args.checkpoint, device)
    print(f"Classes    : {len(idx_to_label)}")

    # Ground truth from filename
    ground_truth = extract_ground_truth(args.video)
    if ground_truth:
        print(f"Ground truth (from filename): {ground_truth}")

    # Inference — sample 16 frames from the full video
    frames_np = read_video_clip(args.video, 1, -1, args.num_frames)
    if frames_np is None:
        print("ERROR: Could not read video.")
        return

    clip = preprocess(frames_np, resize=args.resize, crop=args.crop).to(device)
    with torch.no_grad():
        logits = model(clip)[0]                       # (num_classes,)
        probs  = F.softmax(logits, dim=0)

    top_indices = probs.topk(args.top).indices.tolist()
    predictions = [(idx_to_label[i], probs[i].item()) for i in top_indices]

    print(f"\nTop-{args.top} predictions:")
    for rank, (label, conf) in enumerate(predictions, 1):
        marker = " *" if ground_truth and label.upper() == ground_truth.upper() else ""
        print(f"  {rank}. {label:30s} {conf*100:5.1f}%{marker}")

    # Read all display frames and annotate
    display_frames = read_all_frames(args.video)
    print(f"\nAnnotating {len(display_frames)} frames...")

    annotated = [
        annotate_frame(f, predictions, ground_truth, i, len(display_frames))
        for i, f in enumerate(display_frames)
    ]

    # Save output
    ext = os.path.splitext(args.out)[1].lower()
    if ext == '.mp4':
        first = np.array(annotated[0])
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (w, h))
        for frame in annotated:
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        # GIF
        duration_ms = int(1000 / args.fps)
        annotated[0].save(
            args.out,
            save_all=True,
            append_images=annotated[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )

    print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
