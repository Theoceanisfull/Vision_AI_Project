#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = RUNS_DIR / "analysis" / "latest_runs"

ENCODING_COLORS = {
    "rate": "#2F6BFF",
    "latency": "#F08A24",
    "delta": "#159A78",
}

SERIES_COLORS = {
    "SCNN base": "#6C757D",
    "SCNN deep": "#343A40",
    "SCNN x2 base": "#6FAE7C",
    "SCNN x2 deep": "#1F7A3D",
    "Event2Vec 5e": "#98B8FF",
    "Event2Vec 10e": "#5B8CFF",
    "Event2Vec 50e": "#153EBA",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_percent(value: float) -> float:
    return value * 100.0


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return value
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_metrics_txt(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        data[key.strip()] = parse_scalar(raw_value)
    return data


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_event2vec_50_runs() -> list[dict[str, Any]]:
    summary_path = RUNS_DIR / "event2vec_50_epochs" / "encoding_comparison" / "summary.json"
    summary = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for run in summary["runs"]:
        history_path = Path(run["run_dir"]) / "history.json"
        history = load_json(history_path)
        selection = run["selection"]
        test = run["test"]
        rows.append(
            {
                "encoding": run["encoding"],
                "run_dir": run["run_dir"],
                "best_epoch": selection["best_epoch"],
                "best_val_acc": selection["best_val_metrics"]["acc"],
                "best_val_loss": selection["best_val_metrics"]["loss"],
                "best_val_token_count": selection["best_val_metrics"]["token_count"],
                "test_acc": test["test_acc"],
                "test_loss": test["test_loss"],
                "test_token_count": test["test_token_count"],
                "test_spike_count": test["test_spike_count"],
                "last_test_acc": test["last_test_acc"],
                "last_test_loss": test["last_test_loss"],
                "history": history,
            }
        )
    return rows


def load_summary_encoding(summary_path: Path, encoding: str) -> dict[str, Any]:
    summary = load_json(summary_path)
    for run in summary["runs"]:
        if run["encoding"] == encoding:
            test = run["test"]
            return {
                "test_acc": test["test_acc"],
                "test_loss": test["test_loss"],
                "test_spike_count": test["test_spike_count"],
                "run_dir": run["run_dir"],
            }
    raise KeyError(f"Encoding {encoding!r} not found in {summary_path}")


def load_rate_progression() -> list[dict[str, Any]]:
    event2vec_5 = load_summary_encoding(
        RUNS_DIR / "event2vec" / "encoding_comparison" / "summary.json",
        "rate",
    )
    event2vec_50 = load_summary_encoding(
        RUNS_DIR / "event2vec_50_epochs" / "encoding_comparison" / "summary.json",
        "rate",
    )
    scnn_base = load_summary_encoding(
        RUNS_DIR / "scnn" / "base_encoding_comparison" / "summary.json",
        "rate",
    )
    scnn_deep = load_summary_encoding(
        RUNS_DIR / "scnn" / "deep_encoding_comparison" / "summary.json",
        "rate",
    )
    event2vec_10 = load_metrics_txt(RUNS_DIR / "event2vec_10_epochs" / "rate" / "metrics.txt")
    scnn_x2_base = load_metrics_txt(RUNS_DIR / "scnn_x2_epochs" / "base_rate" / "metrics.txt")
    scnn_x2_deep = load_metrics_txt(RUNS_DIR / "scnn_x2_epochs" / "deep_rate" / "metrics.txt")

    return [
        {
            "label": "SCNN base",
            "family": "SCNN",
            "epochs": "baseline",
            "test_acc": scnn_base["test_acc"],
            "test_loss": scnn_base["test_loss"],
            "test_spike_count": scnn_base["test_spike_count"],
        },
        {
            "label": "SCNN deep",
            "family": "SCNN",
            "epochs": "baseline",
            "test_acc": scnn_deep["test_acc"],
            "test_loss": scnn_deep["test_loss"],
            "test_spike_count": scnn_deep["test_spike_count"],
        },
        {
            "label": "SCNN x2 base",
            "family": "SCNN",
            "epochs": "x2",
            "test_acc": float(scnn_x2_base["test_acc"]),
            "test_loss": float(scnn_x2_base["test_loss"]),
            "test_spike_count": float(scnn_x2_base["test_spike_count"]),
        },
        {
            "label": "SCNN x2 deep",
            "family": "SCNN",
            "epochs": "x2",
            "test_acc": float(scnn_x2_deep["test_acc"]),
            "test_loss": float(scnn_x2_deep["test_loss"]),
            "test_spike_count": float(scnn_x2_deep["test_spike_count"]),
        },
        {
            "label": "Event2Vec 5e",
            "family": "Event2Vec",
            "epochs": 5,
            "test_acc": event2vec_5["test_acc"],
            "test_loss": event2vec_5["test_loss"],
            "test_spike_count": event2vec_5["test_spike_count"],
        },
        {
            "label": "Event2Vec 10e",
            "family": "Event2Vec",
            "epochs": 10,
            "test_acc": float(event2vec_10["test_acc"]),
            "test_loss": float(event2vec_10["test_loss"]),
            "test_spike_count": float(event2vec_10["test_spike_count"]),
        },
        {
            "label": "Event2Vec 50e",
            "family": "Event2Vec",
            "epochs": 50,
            "test_acc": event2vec_50["test_acc"],
            "test_loss": event2vec_50["test_loss"],
            "test_spike_count": event2vec_50["test_spike_count"],
        },
    ]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_scnn_log(log_path: Path) -> dict[str, list[dict[str, float]]]:
    encoding_pattern = re.compile(r"^=== Training encoding: ([a-z_]+)")
    metrics_pattern = re.compile(
        r"train_loss=(?P<train_loss>[0-9.]+), "
        r"train_acc=(?P<train_acc>[0-9.]+), "
        r"val_loss=(?P<val_loss>[0-9.]+), "
        r"val_acc=(?P<val_acc>[0-9.]+), "
        r"train_spikes=(?P<train_spikes>[0-9.]+), "
        r"val_spikes=(?P<val_spikes>[0-9.]+)"
    )
    histories: dict[str, list[dict[str, float]]] = {}
    current_encoding: str | None = None
    epoch_indices: dict[str, int] = {}

    for line in log_path.read_text().splitlines():
        encoding_match = encoding_pattern.search(line)
        if encoding_match:
            current_encoding = encoding_match.group(1)
            histories.setdefault(current_encoding, [])
            epoch_indices.setdefault(current_encoding, 0)
            continue

        metrics_match = metrics_pattern.search(line)
        if metrics_match and current_encoding is not None:
            epoch_indices[current_encoding] += 1
            point = {key: float(value) for key, value in metrics_match.groupdict().items()}
            point["epoch"] = float(epoch_indices[current_encoding])
            histories[current_encoding].append(point)

    return histories


def load_live_scnn_latency() -> list[dict[str, Any]]:
    launch_dir = RUNS_DIR / "scnn" / "launch"
    rows: list[dict[str, Any]] = []

    for pid_path in sorted(launch_dir.glob("*.pid")):
        pid = int(pid_path.read_text().strip())
        if not process_alive(pid):
            continue

        log_path = pid_path.with_suffix(".log")
        if not log_path.exists():
            continue

        label = "SCNN base latency" if "train_three_base" in pid_path.name else "SCNN deep latency"
        parsed = parse_scnn_log(log_path)
        latency_points = parsed.get("latency", [])
        if not latency_points:
            continue

        for point in latency_points:
            rows.append(
                {
                    "label": label,
                    "epoch": int(point["epoch"]),
                    "train_loss": point["train_loss"],
                    "train_acc": point["train_acc"],
                    "val_loss": point["val_loss"],
                    "val_acc": point["val_acc"],
                    "train_spikes": point["train_spikes"],
                    "val_spikes": point["val_spikes"],
                    "log_path": str(log_path),
                }
            )

    rows.sort(key=lambda row: (row["label"], row["epoch"]))
    return rows


def annotate_bars(ax: plt.Axes, values: list[float], suffix: str = "", decimals: int = 2) -> None:
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_event2vec_metric_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["test_acc"], reverse=True)
    labels = [row["encoding"].title() for row in ordered]
    x = np.arange(len(ordered))
    colors = [ENCODING_COLORS[row["encoding"]] for row in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.patch.set_facecolor("#FAF7F2")

    acc_values = [to_percent(row["test_acc"]) for row in ordered]
    loss_values = [row["test_loss"] for row in ordered]
    token_values = [row["test_token_count"] for row in ordered]

    axes[0].bar(x, acc_values, color=colors, edgecolor="#1F2937")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    annotate_bars(axes[0], acc_values, suffix="%")

    axes[1].bar(x, loss_values, color=colors, edgecolor="#1F2937")
    axes[1].set_title("Test Loss")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_xticks(x, labels)
    annotate_bars(axes[1], loss_values, decimals=3)

    axes[2].bar(x, token_values, color=colors, edgecolor="#1F2937")
    axes[2].set_title("Token / Spike Count")
    axes[2].set_ylabel("Average Count")
    axes[2].set_xticks(x, labels)
    annotate_bars(axes[2], token_values, decimals=1)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("Latest Completed Event2Vec Runs (50 Epochs)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_event2vec_efficiency_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    fig.patch.set_facecolor("#FAF7F2")

    for row in rows:
        x_value = row["test_token_count"]
        y_value = to_percent(row["test_acc"])
        color = ENCODING_COLORS[row["encoding"]]
        ax.scatter(x_value, y_value, s=220, color=color, edgecolor="#1F2937", linewidth=1.0)
        ax.annotate(
            f"{row['encoding']} ({row['test_loss']:.3f} loss)",
            (x_value, y_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Accuracy vs Token Budget", fontsize=15, fontweight="bold")
    ax.set_xlabel("Average Test Token / Spike Count")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_event2vec_learning_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#FAF7F2")
    metric_specs = [
        ("train_acc", "Train Accuracy", axes[0, 0], True),
        ("val_acc", "Validation Accuracy", axes[0, 1], True),
        ("train_loss", "Train Loss", axes[1, 0], False),
        ("val_loss", "Validation Loss", axes[1, 1], False),
    ]

    for row in rows:
        history = row["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        color = ENCODING_COLORS[row["encoding"]]
        for metric_key, title, ax, is_accuracy in metric_specs:
            values = np.array(history[metric_key], dtype=float)
            if is_accuracy:
                values = values * 100.0
            ax.plot(epochs, values, label=row["encoding"], color=color, linewidth=2.2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            if is_accuracy:
                ax.set_ylabel("Accuracy (%)")
            else:
                ax.set_ylabel("Loss")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Event2Vec 50-Epoch Training Dynamics", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rate_progression_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    colors = [SERIES_COLORS[row["label"]] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor("#FAF7F2")

    acc_values = [to_percent(row["test_acc"]) for row in rows]
    loss_values = [row["test_loss"] for row in rows]

    axes[0].bar(x, acc_values, color=colors, edgecolor="#1F2937")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Rate-Encoding Progression Across Model Families")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bars(axes[0], acc_values, suffix="%")

    axes[1].bar(x, loss_values, color=colors, edgecolor="#1F2937")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    annotate_bars(axes[1], loss_values, decimals=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_live_latency_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#FAF7F2")
    palette = {
        "SCNN base latency": "#B45F06",
        "SCNN deep latency": "#8E3B46",
    }

    for label, points in grouped.items():
        epochs = [point["epoch"] for point in points]
        val_acc = [point["val_acc"] * 100.0 for point in points]
        val_loss = [point["val_loss"] for point in points]
        color = palette.get(label, "#333333")
        axes[0].plot(epochs, val_acc, marker="o", linewidth=2.2, color=color, label=label)
        axes[1].plot(epochs, val_loss, marker="o", linewidth=2.2, color=color, label=label)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Completed Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Completed Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Active SCNN Latency Jobs (Latest Logged Epoch Summaries)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    event2vec_rows: list[dict[str, Any]],
    rate_progression: list[dict[str, Any]],
    live_latency_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    event2vec_by_encoding = {row["encoding"]: row for row in event2vec_rows}
    rate = event2vec_by_encoding["rate"]
    latency = event2vec_by_encoding["latency"]
    delta = event2vec_by_encoding["delta"]

    e2v_5 = next(row for row in rate_progression if row["label"] == "Event2Vec 5e")
    e2v_10 = next(row for row in rate_progression if row["label"] == "Event2Vec 10e")
    e2v_50 = next(row for row in rate_progression if row["label"] == "Event2Vec 50e")
    scnn_deep = next(row for row in rate_progression if row["label"] == "SCNN deep")
    scnn_x2_deep = next(row for row in rate_progression if row["label"] == "SCNN x2 deep")

    latency_gain = to_percent(latency["test_acc"] - rate["test_acc"])
    latency_token_multiplier = latency["test_token_count"] / rate["test_token_count"]
    delta_gap = to_percent(rate["test_acc"] - delta["test_acc"])
    e2v_5_to_10 = to_percent(e2v_10["test_acc"] - e2v_5["test_acc"])
    e2v_10_to_50 = to_percent(e2v_50["test_acc"] - e2v_10["test_acc"])
    scnn_gain = to_percent(scnn_x2_deep["test_acc"] - scnn_deep["test_acc"])

    lines = [
        "# Latest Run Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Event2Vec 50-Epoch Snapshot",
        "",
        "| Encoding | Best Epoch | Test Acc | Test Loss | Test Tokens | Best Val Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in sorted(event2vec_rows, key=lambda item: item["test_acc"], reverse=True):
        lines.append(
            "| "
            f"{row['encoding']} | "
            f"{row['best_epoch']} | "
            f"{to_percent(row['test_acc']):.2f}% | "
            f"{row['test_loss']:.4f} | "
            f"{row['test_token_count']:.2f} | "
            f"{to_percent(row['best_val_acc']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Rate-Encoding Progression",
            "",
            "| Run | Test Acc | Test Loss | Spike Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for row in rate_progression:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{to_percent(row['test_acc']):.2f}% | "
            f"{row['test_loss']:.4f} | "
            f"{row['test_spike_count']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Active SCNN Latency Jobs",
            "",
            "| Job | Last Completed Epoch | Val Acc | Val Loss | Val Spikes |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    if live_latency_rows:
        latest_rows: dict[str, dict[str, Any]] = {}
        for row in live_latency_rows:
            latest_rows[row["label"]] = row
        for label, row in sorted(latest_rows.items()):
            lines.append(
                "| "
                f"{label} | "
                f"{row['epoch']} | "
                f"{row['val_acc'] * 100.0:.2f}% | "
                f"{row['val_loss']:.4f} | "
                f"{row['val_spikes']:.2f} |"
            )
    else:
        lines.append("| No active SCNN latency jobs detected | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Event2Vec latency is the accuracy leader at {to_percent(latency['test_acc']):.2f}%, beating Event2Vec rate by {latency_gain:.2f} percentage points, but it spends {latency_token_multiplier:.2f}x more tokens/spikes.",
            f"- Event2Vec delta is the most token-efficient encoding, but it trails Event2Vec rate by {delta_gap:.2f} percentage points, which is a large accuracy tax for the smaller token budget.",
            f"- Event2Vec rate improves strongly with longer training: +{e2v_5_to_10:.2f} points from 5 to 10 epochs, then another +{e2v_10_to_50:.2f} points from 10 to 50 epochs.",
            f"- SCNN deep rate improves by {scnn_gain:.2f} points between the first deep run and the x2-epoch deep run, so additional training helped substantially, but it still trails Event2Vec 50e rate by {to_percent(e2v_50['test_acc'] - scnn_x2_deep['test_acc']):.2f} points.",
            "- The live SCNN latency jobs look unhealthy so far: validation accuracy is near chance-level and the validation loss remains high, which usually points to an encoding or optimization mismatch rather than simple undertraining.",
            "",
            "## Generated Charts",
            "",
            "- `event2vec_50_test_metrics.png`",
            "- `event2vec_50_efficiency.png`",
            "- `event2vec_50_learning_curves.png`",
            "- `rate_encoding_progression.png`",
            "- `scnn_live_latency_progress.png`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    event2vec_rows = load_event2vec_50_runs()
    rate_progression = load_rate_progression()
    live_latency_rows = load_live_scnn_latency()

    write_csv(
        OUTPUT_DIR / "event2vec_50_metrics.csv",
        [
            {
                "encoding": row["encoding"],
                "best_epoch": row["best_epoch"],
                "best_val_acc": row["best_val_acc"],
                "best_val_loss": row["best_val_loss"],
                "best_val_token_count": row["best_val_token_count"],
                "test_acc": row["test_acc"],
                "test_loss": row["test_loss"],
                "test_token_count": row["test_token_count"],
                "test_spike_count": row["test_spike_count"],
                "last_test_acc": row["last_test_acc"],
                "last_test_loss": row["last_test_loss"],
            }
            for row in event2vec_rows
        ],
        [
            "encoding",
            "best_epoch",
            "best_val_acc",
            "best_val_loss",
            "best_val_token_count",
            "test_acc",
            "test_loss",
            "test_token_count",
            "test_spike_count",
            "last_test_acc",
            "last_test_loss",
        ],
    )

    write_csv(
        OUTPUT_DIR / "rate_progression.csv",
        rate_progression,
        ["label", "family", "epochs", "test_acc", "test_loss", "test_spike_count"],
    )

    write_csv(
        OUTPUT_DIR / "scnn_live_latency.csv",
        live_latency_rows,
        [
            "label",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "train_spikes",
            "val_spikes",
            "log_path",
        ],
    )

    save_event2vec_metric_chart(event2vec_rows, OUTPUT_DIR / "event2vec_50_test_metrics.png")
    save_event2vec_efficiency_chart(event2vec_rows, OUTPUT_DIR / "event2vec_50_efficiency.png")
    save_event2vec_learning_curves(event2vec_rows, OUTPUT_DIR / "event2vec_50_learning_curves.png")
    save_rate_progression_chart(rate_progression, OUTPUT_DIR / "rate_encoding_progression.png")
    save_live_latency_chart(live_latency_rows, OUTPUT_DIR / "scnn_live_latency_progress.png")
    build_report(event2vec_rows, rate_progression, live_latency_rows, OUTPUT_DIR / "report.md")

    print(f"Wrote analysis bundle to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
