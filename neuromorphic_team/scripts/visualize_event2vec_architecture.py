#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event2vec.config import Event2VecConfig
from event2vec.e2v import Event2VecClassifier

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:  # pragma: no cover - fallback for environments without torchinfo
    torchinfo_summary = None


PAPER_URL = "https://arxiv.org/abs/2504.15371"
PAPER_TITLE = "Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space"
PAPER_FIGURE_6_CAPTION = (
    "Figure 6: The network architecture for event classification using the event2vec representation."
)

PAPER_TABLE5 = {
    "dvs_gesture": {
        "label": "DVS Gesture",
        "D": 64,
        "Df": 128,
        "nhead": 2,
        "l": 4,
        "repeats": 24,
        "ngpus": 4,
        "lrmin": "0",
        "notes": "Average pooling after each FFN is enabled for this dataset.",
    },
    "asl_dvs": {
        "label": "ASL-DVS",
        "D": 64,
        "Df": 128,
        "nhead": 2,
        "l": 2,
        "repeats": 1,
        "ngpus": 7,
        "lrmin": "1e-6",
        "notes": "No pooling after FFN in the appendix configuration.",
    },
    "dvs_lip": {
        "label": "DVS-Lip",
        "D": 192,
        "Df": 384,
        "nhead": 6,
        "l": 16,
        "repeats": 3,
        "ngpus": 4,
        "lrmin": "1e-6",
        "notes": "The paper describes a self-supervised pretraining stage for this model.",
    },
}


@dataclass(frozen=True)
class DiagramBlock:
    text: str
    color: str


@dataclass(frozen=True)
class ComparisonRow:
    stage: str
    paper: str
    local: str
    status: str
    implication: str


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def load_config(path: Path) -> Event2VecConfig:
    return Event2VecConfig.from_json(path)


def pooled_sensor_size(cfg: Event2VecConfig) -> tuple[int, int]:
    sensor_h, sensor_w = cfg.data.sensor_size
    pool_h, pool_w = cfg.data.pool_kernel
    return sensor_h // pool_h, sensor_w // pool_w


def build_model(cfg: Event2VecConfig) -> Event2VecClassifier:
    height, width = pooled_sensor_size(cfg)
    return Event2VecClassifier(
        num_classes=cfg.model.num_classes,
        height=height,
        width=width,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        pool_after_each_block=cfg.model.pool_after_each_block,
    )


def make_summary_text(model: Event2VecClassifier, cfg: Event2VecConfig) -> str:
    max_tokens = cfg.data.max_tokens
    dummy_events = torch.zeros(1, max_tokens, 5, dtype=torch.float32)
    dummy_padding_mask = torch.zeros(1, max_tokens, dtype=torch.bool)

    if torchinfo_summary is not None:
        try:
            info = torchinfo_summary(
                model,
                input_data=[dummy_events, dummy_padding_mask],
                depth=4,
                verbose=0,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                row_settings=("var_names", "depth"),
            )
            return str(info)
        except Exception as exc:  # pragma: no cover - best effort reporting
            return (
                f"torchinfo failed with: {exc}\n\n"
                f"{model}\n\n"
                f"Total parameters: {count_parameters(model):,}\n"
            )

    return f"{model}\n\nTotal parameters: {count_parameters(model):,}\n"


def make_paper_blocks(preset: dict[str, object]) -> list[DiagramBlock]:
    pooling_note = "Optional pooling" if preset["label"] == "DVS Gesture" else "No pooling in this preset"
    return [
        DiagramBlock(
            "Raw events\n(x, y, t, p)\noptional clustering gives intensity rho",
            "#d8ecff",
        ),
        DiagramBlock(
            "Spatial embedding\nLinear 3 -> D/4 -> ReLU\nLinear D/4 -> D/2 -> ReLU\nLinear D/2 -> D",
            "#dff5e1",
        ),
        DiagramBlock(
            "Temporal embedding\nConv1d 1 -> D/4 -> ReLU\nConv1d D/4 -> D/2 -> ReLU\nConv1d D/2 -> D",
            "#fff1d6",
        ),
        DiagramBlock(
            "Event2Vec fusion\nV = (Vs + Vt) * (log(rho) + 1)",
            "#f4def6",
        ),
        DiagramBlock(
            f"Backbone x l\nSelf-Attention + FFN\nD={preset['D']}, Df={preset['Df']}, heads={preset['nhead']}, l={preset['l']}\n{pooling_note}",
            "#ffe3d5",
        ),
        DiagramBlock(
            "Linear classification head\nclass logits",
            "#ececec",
        ),
    ]


def make_local_blocks(cfg: Event2VecConfig, model: Event2VecClassifier) -> list[DiagramBlock]:
    pooled_h, pooled_w = pooled_sensor_size(cfg)
    depth = cfg.model.depth
    pool_flag = any(cfg.model.pool_after_each_block) if isinstance(cfg.model.pool_after_each_block, list) else bool(cfg.model.pool_after_each_block)
    spatial_fc1 = model.event2vec.spatial.fc1.out_features
    spatial_fc2 = model.event2vec.spatial.fc2.out_features
    temporal_c1 = model.event2vec.temporal.conv1.out_channels
    temporal_c2 = model.event2vec.temporal.conv2.out_channels
    return [
        DiagramBlock(
            "Raw DVS events",
            "#d8ecff",
        ),
        DiagramBlock(
            f"event2vec/data.py\nbin to {cfg.data.num_steps} time steps\n2 channels (ON/OFF)",
            "#d6f1ff",
        ),
        DiagramBlock(
            f"AvgPool sensor\n{cfg.data.sensor_size[0]}x{cfg.data.sensor_size[1]} -> {pooled_h}x{pooled_w}\npool_kernel={tuple(cfg.data.pool_kernel)}",
            "#dff5e1",
        ),
        DiagramBlock(
            f"Encoding preset\n{cfg.data.encoding}\nrate | latency | delta",
            "#fff1d6",
        ),
        DiagramBlock(
            f"Nonzero spikes -> tokens\n[x, y, t_norm, p, rho]\nmax_tokens={cfg.data.max_tokens}",
            "#f7ead0",
        ),
        DiagramBlock(
            f"SpatialEmbedding\nLinear 3 -> {spatial_fc1} -> {spatial_fc2} -> {cfg.model.d_model}\nLayerNorm after each linear",
            "#dff5e1",
        ),
        DiagramBlock(
            f"TemporalEmbedding\nConv1d 1 -> {temporal_c1} -> {temporal_c2} -> {cfg.model.d_model}\nLayerNorm after each conv over delta-t",
            "#fff1d6",
        ),
        DiagramBlock(
            "Fusion\nv = (log(rho) + 1) * (Vs + Vt)",
            "#f4def6",
        ),
        DiagramBlock(
            f"Backbone x {depth}\nSharedBidirectionalAttentionBlock\nforward + reversed MultiheadAttention\nheads={cfg.model.num_heads}, FFN {cfg.model.d_model}->{cfg.model.ffn_dim}->{cfg.model.d_model}, dropout={cfg.model.dropout}\npooling={'on' if pool_flag else 'off'}",
            "#ffe3d5",
        ),
        DiagramBlock(
            f"Masked mean -> Linear {cfg.model.d_model}->{cfg.model.num_classes}\nclass logits",
            "#ececec",
        ),
    ]


def draw_pipeline(
    ax: plt.Axes,
    title: str,
    subtitle: str,
    blocks: list[DiagramBlock],
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=10, color="#333333")

    top = 0.90
    bottom = 0.06
    n = len(blocks)
    gap = 0.018
    box_h = (top - bottom - gap * (n - 1)) / n
    box_w = 0.82
    x0 = 0.09

    for idx, block in enumerate(blocks):
        y0 = top - (idx + 1) * box_h - idx * gap
        patch = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#2f3640",
            facecolor=block.color,
        )
        ax.add_patch(patch)
        ax.text(
            x0 + box_w / 2,
            y0 + box_h / 2,
            block.text,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#111111",
            wrap=True,
        )

        if idx < n - 1:
            next_y = top - (idx + 2) * box_h - (idx + 1) * gap + box_h
            arrow = FancyArrowPatch(
                (0.5, y0),
                (0.5, next_y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#444444",
            )
            ax.add_patch(arrow)


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def make_notes_markdown(
    cfg_path: Path,
    cfg: Event2VecConfig,
    model: Event2VecClassifier,
    paper_preset_key: str,
    summary_text: str,
) -> str:
    pooled_h, pooled_w = pooled_sensor_size(cfg)
    paper = PAPER_TABLE5[paper_preset_key]
    pool_flag = any(cfg.model.pool_after_each_block) if isinstance(cfg.model.pool_after_each_block, list) else bool(cfg.model.pool_after_each_block)
    param_count = count_parameters(model)

    difference_lines = [
        "The paper Figure 6 starts from raw events and optional clustering-derived intensity rho. This project inserts a preprocessing pipeline in `event2vec/data.py` that bins events to frames, average-pools the sensor, applies a spike encoding (`rate`, `latency`, or `delta`), and only then converts nonzero activations into `[x, y, t, p, rho]` tokens.",
        "The paper appendix discusses bidirectional variants of FoX and GLA attention. The local model in `event2vec/e2v.py` uses `nn.MultiheadAttention` inside `SharedBidirectionalAttentionBlock`, not FoX or GLA.",
        f"The local default config matches the paper's ASL-DVS scale fairly closely on backbone size: D={cfg.model.d_model}, Df={cfg.model.ffn_dim}, heads={cfg.model.num_heads}, depth={cfg.model.depth}.",
        f"The local default config disables pooling in the backbone (`pool_after_each_block={cfg.model.pool_after_each_block}`), which is aligned with the paper's ASL-DVS appendix setting but differs from the DVS Gesture preset.",
    ]

    bullet_block = "\n".join(f"- {line}" for line in difference_lines)
    return f"""# Event2Vec Architecture Comparison

Generated from:
- Paper: [{PAPER_TITLE}]({PAPER_URL})
- Local config: `{cfg_path}`

## Paper Reference

{PAPER_FIGURE_6_CAPTION}

Selected paper preset: `{paper_preset_key}` ({paper['label']})

- Embedding dimension `D={paper['D']}`
- FFN hidden dimension `Df={paper['Df']}`
- Attention heads `nhead={paper['nhead']}`
- Backbone depth `l={paper['l']}`
- Repeats `{paper['repeats']}`
- GPUs `{paper['ngpus']}`
- `lrmin={paper['lrmin']}`
- Note: {paper['notes']}

## Local Project Pipeline

- Data root: `{cfg.data.data_root}`
- Encoding: `{cfg.data.encoding}`
- Sensor size: `{tuple(cfg.data.sensor_size)}`
- Pool kernel: `{tuple(cfg.data.pool_kernel)}`
- Pooled sensor size: `{(pooled_h, pooled_w)}`
- Time steps: `{cfg.data.num_steps}`
- Max tokens: `{cfg.data.max_tokens}`
- Model classes: `{cfg.model.num_classes}`
- Model dimensions: `D={cfg.model.d_model}`, `Df={cfg.model.ffn_dim}`, `heads={cfg.model.num_heads}`, `depth={cfg.model.depth}`
- Backbone pooling enabled: `{pool_flag}`
- Parameter count: `{param_count:,}`

## Key Differences

{bullet_block}

## torchinfo Summary

```text
{summary_text.rstrip()}
```
"""


def build_comparison_rows(
    cfg: Event2VecConfig,
    paper_preset_key: str,
    model: Event2VecClassifier,
) -> list[ComparisonRow]:
    paper = PAPER_TABLE5[paper_preset_key]
    pooled_h, pooled_w = pooled_sensor_size(cfg)
    pool_flag = any(cfg.model.pool_after_each_block) if isinstance(cfg.model.pool_after_each_block, list) else bool(cfg.model.pool_after_each_block)
    return [
        ComparisonRow(
            stage="Input representation",
            paper="Raw events (x, y, t, p), with rho available when event clustering is used.",
            local="Raw DVS events are read first, but the model does not consume them directly.",
            status="Partial",
            implication="Both start from events, but the local model only sees derived tokens after extra preprocessing.",
        ),
        ComparisonRow(
            stage="Pre-token preprocessing",
            paper="Figure 6 shows direct Event2Vec processing; appendix text mentions clustering for long streams, not frame/bin conversion.",
            local=(
                f"Events are binned into {cfg.data.num_steps} temporal frames, split into ON/OFF channels, "
                f"average-pooled from {cfg.data.sensor_size[0]}x{cfg.data.sensor_size[1]} to {pooled_h}x{pooled_w}, "
                f"then encoded with '{cfg.data.encoding}'."
            ),
            status="Different",
            implication="This is the largest architectural difference. The project front-end is not the paper's direct raw-event path.",
        ),
        ComparisonRow(
            stage="Token formation",
            paper="Event tokens conceptually come from events themselves and include rho scaling when available.",
            local="Only nonzero encoded spikes become tokens [x, y, t_norm, p, rho], then token count is capped.",
            status="Different",
            implication="The local token stream is sparser and already transformed before Event2Vec sees it.",
        ),
        ComparisonRow(
            stage="Spatial embedding",
            paper="Linear 3 -> D/4 -> ReLU -> Linear D/4 -> D/2 -> ReLU -> Linear D/2 -> D.",
            local=(
                f"Same shape in `SpatialEmbedding`: Linear 3 -> {model.event2vec.spatial.fc1.out_features} -> "
                f"{model.event2vec.spatial.fc2.out_features} -> {cfg.model.d_model}, with LayerNorm after each linear."
            ),
            status="Match",
            implication="This part is very close to the Figure 6 reference.",
        ),
        ComparisonRow(
            stage="Temporal embedding",
            paper="Conv1d 1 -> D/4 -> ReLU -> Conv1d D/4 -> D/2 -> ReLU -> Conv1d D/2 -> D over delta-t.",
            local=(
                f"Same channel progression in `TemporalEmbedding`: Conv1d 1 -> {model.event2vec.temporal.conv1.out_channels} -> "
                f"{model.event2vec.temporal.conv2.out_channels} -> {cfg.model.d_model} over delta-t, with LayerNorm after each conv."
            ),
            status="Match",
            implication="This also closely follows Figure 6, although the local code uses standard Conv1d as noted in `e2v.py`.",
        ),
        ComparisonRow(
            stage="Event2Vec fusion",
            paper="V = (Vs + Vt) * (log rho + 1).",
            local="Same formula in `Event2Vec.forward`.",
            status="Match",
            implication="The core Event2Vec token fusion is aligned.",
        ),
        ComparisonRow(
            stage="Backbone block",
            paper=(
                f"Backbone x l with Self-Attention + FFN. For {paper['label']}: "
                f"D={paper['D']}, Df={paper['Df']}, heads={paper['nhead']}, depth={paper['l']}."
            ),
            local=(
                f"`SharedBidirectionalAttentionBlock` x {cfg.model.depth}: forward + reversed `nn.MultiheadAttention`, "
                f"fusion linear, FFN {cfg.model.d_model}->{cfg.model.ffn_dim}->{cfg.model.d_model}."
            ),
            status="Partial",
            implication="The size matches the ASL-DVS paper preset, but the attention implementation differs from the appendix's FoX/GLA discussion.",
        ),
        ComparisonRow(
            stage="Pooling in backbone",
            paper="Optional; enabled for DVS Gesture, disabled for ASL-DVS and DVS-Lip.",
            local=f"`pool_after_each_block={cfg.model.pool_after_each_block}`.",
            status="Match",
            implication="Your default config matches the ASL-DVS paper preset here.",
        ),
        ComparisonRow(
            stage="Readout / head",
            paper="Backbone output feeds a linear classification head.",
            local="Masked mean pooling across tokens, then Linear 64->24.",
            status="Partial",
            implication="Functionally similar classifier output, but the local readout explicitly averages token features first.",
        ),
        ComparisonRow(
            stage="Overall verdict",
            paper="Direct Event2Vec classification pipeline from Figure 6 / Appendix A.2.",
            local="Paper-like Event2Vec encoder/backbone attached to a project-specific preprocessing front-end.",
            status="Different",
            implication="Your project is best described as Event2Vec-inspired or partially paper-aligned, not a strict Figure 6 reproduction.",
        ),
    ]


def make_alignment_markdown(
    cfg_path: Path,
    cfg: Event2VecConfig,
    paper_preset_key: str,
    rows: list[ComparisonRow],
) -> str:
    header = [
        "# Event2Vec Side-by-Side Alignment",
        "",
        f"Paper reference: [{PAPER_TITLE}]({PAPER_URL})",
        "Checked against the latest arXiv version available on February 5, 2026 (v5), including Figure 6 and Appendix A.2/Table 5.",
        f"Local config: `{cfg_path}`",
        "",
        "## Fast Read",
        "",
        "- The Event2Vec embedding core in this project is close to the paper.",
        "- The data front-end in this project is not the paper front-end.",
        "- So the backbone is paper-like, but the full end-to-end pipeline is not a strict Figure 6 reproduction.",
        "",
        "## Row-by-Row Comparison",
        "",
        "| Stage | Paper Figure 6 / Appendix | Local project | Status | Why it matters |",
        "| --- | --- | --- | --- | --- |",
    ]
    body = [
        f"| {row.stage} | {row.paper} | {row.local} | {row.status} | {row.implication} |"
        for row in rows
    ]
    footer = [
        "",
        "## Local Defaults",
        "",
        f"- Encoding: `{cfg.data.encoding}`",
        f"- Sensor size: `{tuple(cfg.data.sensor_size)}`",
        f"- Pool kernel: `{tuple(cfg.data.pool_kernel)}`",
        f"- Time steps: `{cfg.data.num_steps}`",
        f"- Max tokens: `{cfg.data.max_tokens}`",
        f"- Model dims: `D={cfg.model.d_model}`, `Df={cfg.model.ffn_dim}`, `heads={cfg.model.num_heads}`, `depth={cfg.model.depth}`",
    ]
    return "\n".join(header + body + footer) + "\n"


def build_alignment_figure(output_path: Path, rows: list[ComparisonRow]) -> None:
    n = len(rows)
    fig_h = max(8.5, 1.18 * (n + 1))
    fig, ax = plt.subplots(figsize=(22, fig_h), constrained_layout=True)
    ax.axis("off")

    columns = ["Stage", "Paper", "Local project", "Status", "Why it matters"]
    cell_text = [[row.stage, row.paper, row.local, row.status, row.implication] for row in rows]
    col_widths = [0.13, 0.25, 0.26, 0.08, 0.28]

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colLoc="left",
        cellLoc="left",
        colWidths=col_widths,
        bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)

    status_colors = {
        "Match": "#dff5e1",
        "Partial": "#fff1d6",
        "Different": "#ffe3d5",
    }

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#4b5563")
        cell.set_linewidth(0.9)
        if row_idx == 0:
            cell.set_facecolor("#d8ecff")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            if col_idx == 3:
                status = cell.get_text().get_text()
                cell.set_facecolor(status_colors.get(status, "#f3f4f6"))
                cell.set_text_props(weight="bold", color="#111111")
            else:
                cell.set_facecolor("#ffffff")

    ax.set_title("Event2Vec Paper vs Local Project: Row-by-Row Alignment", fontsize=18, fontweight="bold", pad=14)
    fig.text(
        0.5,
        0.975,
        "Green = close match, amber = similar idea but not identical, red = materially different stage.",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_figure(
    output_path: Path,
    cfg: Event2VecConfig,
    paper_preset_key: str,
    model: Event2VecClassifier,
) -> None:
    paper = PAPER_TABLE5[paper_preset_key]
    fig, axes = plt.subplots(1, 2, figsize=(15, 18), constrained_layout=True)
    draw_pipeline(
        axes[0],
        title="Paper Figure 6 Reference",
        subtitle=f"{paper['label']} preset from Appendix Table 5",
        blocks=make_paper_blocks(paper),
    )
    draw_pipeline(
        axes[1],
        title="Local Project Pipeline",
        subtitle="event2vec/data.py + event2vec/e2v.py + default_config.json",
        blocks=make_local_blocks(cfg, model),
    )
    fig.suptitle("Event2Vec Architecture Comparison", fontsize=20, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        textwrap.fill(
            "Left: recreated Figure 6 architecture from the paper appendix. "
            "Right: the actual local pipeline in this project, including preprocessing before Event2Vec tokens are formed. "
            "This makes it easier to compare the canonical paper design against the implementation you are currently training.",
            width=140,
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-vs-local Event2Vec architecture comparison.")
    parser.add_argument(
        "--config",
        default="event2vec/default_config.json",
        help="Path to the local Event2Vec config JSON.",
    )
    parser.add_argument(
        "--paper-preset",
        choices=sorted(PAPER_TABLE5),
        default="asl_dvs",
        help="Dataset preset from the paper appendix Table 5 to display on the reference side.",
    )
    parser.add_argument(
        "--output-dir",
        default="Analytics/architecture",
        help="Directory where image and summary files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    model = build_model(cfg)
    summary_text = make_summary_text(model, cfg)
    comparison_rows = build_comparison_rows(cfg, args.paper_preset, model)

    png_path = output_dir / "event2vec_architecture_comparison.png"
    svg_path = output_dir / "event2vec_architecture_comparison.svg"
    txt_path = output_dir / "event2vec_model_summary.txt"
    md_path = output_dir / "event2vec_architecture_notes.md"
    align_png_path = output_dir / "event2vec_alignment_matrix.png"
    align_md_path = output_dir / "event2vec_alignment_matrix.md"

    build_figure(png_path, cfg, args.paper_preset, model)
    build_figure(svg_path, cfg, args.paper_preset, model)
    build_alignment_figure(align_png_path, comparison_rows)
    write_text(txt_path, summary_text)
    write_text(md_path, make_notes_markdown(cfg_path, cfg, model, args.paper_preset, summary_text))
    write_text(align_md_path, make_alignment_markdown(cfg_path, cfg, args.paper_preset, comparison_rows))

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {align_png_path}")
    print(f"Wrote {align_md_path}")


if __name__ == "__main__":
    main()
