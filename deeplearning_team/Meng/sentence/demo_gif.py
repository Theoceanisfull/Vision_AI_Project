"""
Generate a demo GIF: video plays with the model's predicted translation
revealed word-by-word, and the reference sentence shown below.

Usage:
  # Random sample from test split
  env/python.exe sentence/demo_gif.py --config sentence/configs/config_how2sign.yaml

  # Specific sentence
  env/python.exe sentence/demo_gif.py --config sentence/configs/config_how2sign.yaml --sentence "g3X3XE6M2_A_20-3-rgb_front"

  # Multiple GIFs at once
  env/python.exe sentence/demo_gif.py --config sentence/configs/config_how2sign.yaml --n 3

  # Custom output directory
  env/python.exe sentence/demo_gif.py --config sentence/configs/config_how2sign.yaml --out results/demos
"""

import argparse
import os
import random
import textwrap

import cv2
import imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.dataset import How2SignDataset, _resample
from src.model import build_model, build_tokenizer


# ── Layout constants ──────────────────────────────────────────────────────────
VID_W, VID_H   = 480, 270      # video panel (3/8 of 1280x720)
PANEL_H        = 140           # text panel height below video
TOTAL_H        = VID_H + PANEL_H
GIF_FPS        = 10            # output GIF fps (subsample from 24)
HOLD_SECS      = 2.0           # extra frames to hold at the end
GIF_COLORS     = 128           # palette size for quantization (smaller = smaller file)

FONT_PATH      = 'C:/Windows/Fonts/calibri.ttf'
FONT_SIZE_HYP  = 18
FONT_SIZE_REF  = 15
FONT_SIZE_LABEL= 13

BG_COLOR       = (18, 18, 18)
HYP_COLOR      = (255, 255, 255)
HYP_PENDING    = (80, 80, 80)
REF_COLOR      = (100, 220, 120)
LABEL_COLOR    = (150, 150, 150)
BAR_BG         = (45, 45, 45)
BAR_FG         = (80, 160, 240)


def load_model_and_tokenizer(cfg, ckpt_path, device):
    tokenizer = build_tokenizer(cfg)
    model     = build_model(cfg)
    ckpt      = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        cfg['model'].get('tgt_lang', 'en_XX')
    )
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def run_inference(npy_path, model, tokenizer, cfg, device):
    from src.dataset import _normalize_pose, _add_velocity
    num_frames     = cfg['data']['num_frames']
    num_beams      = cfg['training'].get('num_beams', 4)
    max_new        = cfg['training'].get('max_tgt_len', 128)
    normalize_pose = cfg['data'].get('normalize_pose', False)
    use_velocity   = cfg['data'].get('use_velocity', False)

    kp = np.load(npy_path).astype(np.float32)
    original_T = len(kp)
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
    hyp = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return hyp, original_T


def load_video_frames(video_path, target_fps=GIF_FPS):
    cap    = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step   = max(1, round(src_fps / target_fps))
    frames = []
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (VID_W, VID_H))
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def make_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """Wrap text so it fits within max_width pixels."""
    words  = text.split()
    lines  = []
    line   = ''
    for word in words:
        test = (line + ' ' + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_text_panel(hyp_words_shown, hyp_words_all, ref_text,
                    progress_frac, font_hyp, font_ref, font_label):
    """Return a PIL Image of the text panel."""
    img  = Image.new('RGB', (VID_W, PANEL_H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    PAD = 10

    # Progress bar
    bar_y, bar_h = 8, 5
    draw.rectangle([PAD, bar_y, VID_W - PAD, bar_y + bar_h], fill=BAR_BG)
    bar_end = PAD + int((VID_W - 2*PAD) * progress_frac)
    if bar_end > PAD:
        draw.rectangle([PAD, bar_y, bar_end, bar_y + bar_h], fill=BAR_FG)

    # HYP label
    hyp_label_y = bar_y + bar_h + 8
    draw.text((PAD, hyp_label_y), 'MODEL:', font=font_label, fill=LABEL_COLOR)
    lbl_bbox = draw.textbbox((0, 0), 'MODEL:', font=font_label)
    lbl_w = lbl_bbox[2] + PAD + 4

    # HYP text — shown words in white, remaining in dark grey
    hyp_y  = hyp_label_y
    full   = ' '.join(hyp_words_all)
    shown  = ' '.join(hyp_words_shown)
    rest   = ' '.join(hyp_words_all[len(hyp_words_shown):])

    text_x = PAD + lbl_w
    max_w  = VID_W - text_x - PAD

    # Render shown words (white)
    shown_lines = wrap_text(shown, font_hyp, max_w, draw) if shown else []
    for line in shown_lines[:2]:
        draw.text((text_x, hyp_y), line, font=font_hyp, fill=HYP_COLOR)
        hyp_y += font_hyp.size + 3

    # If still room and there are pending words, render them dimmed on same/next line
    if rest and len(shown_lines) < 2:
        draw.text((text_x, hyp_y), rest[:80], font=font_hyp, fill=HYP_PENDING)

    # REF label + text
    ref_label_y = PANEL_H - 55
    draw.text((PAD, ref_label_y), 'REFERENCE:', font=font_label, fill=LABEL_COLOR)
    lbl_bbox2 = draw.textbbox((0, 0), 'REFERENCE:', font=font_label)
    ref_text_x = PAD + lbl_bbox2[2] + 4

    ref_lines = wrap_text(ref_text, font_ref, VID_W - ref_text_x - PAD, draw)
    ry = ref_label_y
    for line in ref_lines[:2]:
        draw.text((ref_text_x, ry), line, font=font_ref, fill=REF_COLOR)
        ry += font_ref.size + 3

    return img


def make_gif(video_path, npy_path, ref_text, hyp_text, out_path):
    print(f'  Loading video  : {os.path.basename(video_path)}')
    vid_frames = load_video_frames(video_path)
    if not vid_frames:
        print('  ERROR: no video frames read')
        return False

    n_vid   = len(vid_frames)
    hyp_words = hyp_text.split()
    n_words   = len(hyp_words)

    font_hyp   = make_font(FONT_SIZE_HYP)
    font_ref   = make_font(FONT_SIZE_REF)
    font_label = make_font(FONT_SIZE_LABEL)

    hold_frames = max(1, int(HOLD_SECS * GIF_FPS))
    gif_frames  = []

    print(f'  Video frames   : {n_vid}  |  HYP words: {n_words}')
    print(f'  HYP : {hyp_text}')
    print(f'  REF : {ref_text}')

    for i, vf in enumerate(vid_frames):
        progress   = (i + 1) / n_vid
        n_revealed = round(progress * n_words)
        shown      = hyp_words[:n_revealed]

        vid_img    = Image.fromarray(vf)
        panel_img  = draw_text_panel(shown, hyp_words, ref_text,
                                     progress, font_hyp, font_ref, font_label)

        combined   = Image.new('RGB', (VID_W, TOTAL_H))
        combined.paste(vid_img,   (0, 0))
        combined.paste(panel_img, (0, VID_H))
        gif_frames.append(np.array(combined))

    # Hold last frame with full sentence
    last_panel = draw_text_panel(hyp_words, hyp_words, ref_text,
                                 1.0, font_hyp, font_ref, font_label)
    last_combined = Image.new('RGB', (VID_W, TOTAL_H))
    last_combined.paste(Image.fromarray(vid_frames[-1]), (0, 0))
    last_combined.paste(last_panel, (0, VID_H))
    for _ in range(hold_frames):
        gif_frames.append(np.array(last_combined))

    # Quantize each frame to reduce file size
    pil_frames = [Image.fromarray(f).quantize(colors=GIF_COLORS,
                  method=Image.Quantize.MEDIANCUT) for f in gif_frames]
    duration_ms = int(1000 / GIF_FPS)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f'  Saved GIF      : {out_path}  ({size_kb:.0f} KB)')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='sentence/configs/config_how2sign.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n',          type=int, default=1, help='Number of GIFs to generate')
    parser.add_argument('--sentence',   default=None, help='Specific SENTENCE_NAME')
    parser.add_argument('--out',        default=None, help='Output directory')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.checkpoint or os.path.join(cfg['training']['checkpoint_dir'], 'best.pth')
    out_dir   = args.out or os.path.join('results', 'demos')
    os.makedirs(out_dir, exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Checkpoint : {ckpt_path}')

    model, tokenizer = load_model_and_tokenizer(cfg, ckpt_path, device)

    ds = How2SignDataset(
        data_root      = cfg['data']['data_root'],
        split          = args.split,
        num_frames     = cfg['data']['num_frames'],
        augment        = False,
        normalize_pose = cfg['data'].get('normalize_pose', False),
        use_velocity   = cfg['data'].get('use_velocity', False),
    )

    name_to_idx = {
        os.path.splitext(os.path.basename(s['npy']))[0]: i
        for i, s in enumerate(ds.samples)
    }
    vid_dir = os.path.join(cfg['data']['data_root'],
                           f'{args.split}_rgb_front_clips', 'raw_videos')

    if args.sentence:
        if args.sentence not in name_to_idx:
            print(f'Sentence "{args.sentence}" not found.')
            return
        chosen = [name_to_idx[args.sentence]]
    else:
        random.seed(args.seed)
        # Prefer clips that have a matching video file
        candidates = [
            i for i, s in enumerate(ds.samples)
            if os.path.exists(os.path.join(
                vid_dir,
                os.path.splitext(os.path.basename(s['npy']))[0] + '.mp4'
            ))
        ]
        if not candidates:
            candidates = list(range(len(ds)))
        chosen = random.sample(candidates, min(args.n, len(candidates)))

    print(f'\nGenerating {len(chosen)} GIF(s)...\n')

    for rank, idx in enumerate(chosen, 1):
        sample = ds.samples[idx]
        name   = os.path.splitext(os.path.basename(sample['npy']))[0]
        ref    = sample['text']

        vid_path = os.path.join(vid_dir, name + '.mp4')
        if not os.path.exists(vid_path):
            print(f'[{rank}] Skipping {name}: video not found at {vid_path}')
            continue

        print(f'[{rank}/{len(chosen)}] {name}')
        hyp, _ = run_inference(sample['npy'], model, tokenizer, cfg, device)

        out_path = os.path.join(out_dir, f'{name}.gif')
        make_gif(vid_path, sample['npy'], ref, hyp, out_path)
        print()

    print(f'All GIFs saved to: {out_dir}')


if __name__ == '__main__':
    main()
