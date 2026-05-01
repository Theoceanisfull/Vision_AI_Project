from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = PROJECT_ROOT / "Analytics"

ENCODINGS = ["rate", "latency", "delta"]
CSV_BY_ENCODING = {
    "rate": ANALYTICS_DIR / "rate_progression.csv",
    "latency": ANALYTICS_DIR / "latency_progression.csv",
    "delta": ANALYTICS_DIR / "delta_progression.csv",
}
LABEL_ORDER = ["SCNN base", "SCNN deep"]
DISPLAY_NAMES = {
    "rate": "Rate",
    "latency": "Latency",
    "delta": "Delta",
}
COLORS = {
    "SCNN base": "#4C78A8",
    "SCNN deep": "#F58518",
}


def load_baseline_scnn_rows() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}

    for encoding, csv_path in CSV_BY_ENCODING.items():
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["family"] == "SCNN" and row["epochs"] == "baseline"]

        by_label = {row["label"]: row for row in rows}
        missing = [label for label in LABEL_ORDER if label not in by_label]
        if missing:
            raise ValueError(f"Missing rows for {encoding}: {missing}")

        data[encoding] = {}
        for label in LABEL_ORDER:
            row = by_label[label]
            data[encoding][label] = {
                "test_acc": float(row["test_acc"]),
                "test_loss": float(row["test_loss"]),
                "test_spike_count": float(row["test_spike_count"]),
            }

    return data


def annotate_bars(ax: plt.Axes, bars, *, fmt: str, y_offset: float) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            format(height, fmt),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_metric_plot(
    metric_key: str,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    value_fmt: str,
    convert=None,
) -> None:
    data = load_baseline_scnn_rows()
    x = np.arange(len(ENCODINGS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    all_values = []
    for label in LABEL_ORDER:
        values = []
        for encoding in ENCODINGS:
            value = data[encoding][label][metric_key]
            if convert is not None:
                value = convert(value)
            values.append(value)
        all_values.extend(values)
        bars = ax.bar(
            x + (-width / 2 if label == "SCNN base" else width / 2),
            values,
            width=width,
            label=label,
            color=COLORS[label],
        )
        ymax = max(all_values) if all_values else 1.0
        annotate_bars(ax, bars, fmt=value_fmt, y_offset=max(ymax * 0.015, 0.02))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [DISPLAY_NAMES[encoding] for encoding in ENCODINGS])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    if metric_key == "test_acc":
        ax.set_ylim(0, max(all_values) * 1.18)
    else:
        ax.set_ylim(0, max(all_values) * 1.18 if all_values else 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_combined_plot(output_path: Path) -> None:
    data = load_baseline_scnn_rows()
    x = np.arange(len(ENCODINGS))
    width = 0.34

    metrics = [
        ("test_acc", "Test Accuracy (%)", lambda v: v * 100.0, ".1f"),
        ("test_loss", "Test Loss", None, ".3f"),
        ("test_spike_count", "Average Spike Count", None, ".2f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    for ax, (metric_key, ylabel, convert, fmt) in zip(axes, metrics):
        all_values = []
        for label in LABEL_ORDER:
            values = []
            for encoding in ENCODINGS:
                value = data[encoding][label][metric_key]
                if convert is not None:
                    value = convert(value)
                values.append(value)
            all_values.extend(values)
            bars = ax.bar(
                x + (-width / 2 if label == "SCNN base" else width / 2),
                values,
                width=width,
                label=label,
                color=COLORS[label],
            )
            ymax = max(all_values) if all_values else 1.0
            annotate_bars(ax, bars, fmt=fmt, y_offset=max(ymax * 0.02, 0.02))

        ax.set_title(ylabel)
        ax.set_xticks(x, [DISPLAY_NAMES[encoding] for encoding in ENCODINGS])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(all_values) * 1.22 if all_values else 1)

    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("SCNN Base vs Deep Across Rate, Latency, and Delta", fontsize=14, y=1.10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(output_path: Path) -> None:
    data = load_baseline_scnn_rows()
    lines = [
        "# SCNN Base vs Deep",
        "",
        "This table compares the baseline SCNN and deep SCNN across the three encodings using the saved baseline analysis artifacts.",
        "",
        "| Encoding | Model | Test Accuracy | Test Loss | Average Spike Count |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for encoding in ENCODINGS:
        for label in LABEL_ORDER:
            row = data[encoding][label]
            lines.append(
                f"| {DISPLAY_NAMES[encoding]} | {label} | {row['test_acc'] * 100:.2f}% | {row['test_loss']:.4f} | {row['test_spike_count']:.2f} |"
            )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    make_metric_plot(
        "test_acc",
        title="SCNN Base vs Deep: Test Accuracy by Encoding",
        ylabel="Test Accuracy (%)",
        output_path=ANALYTICS_DIR / "scnn_base_vs_deep_accuracy.png",
        value_fmt=".1f",
        convert=lambda v: v * 100.0,
    )
    make_metric_plot(
        "test_loss",
        title="SCNN Base vs Deep: Test Loss by Encoding",
        ylabel="Test Loss",
        output_path=ANALYTICS_DIR / "scnn_base_vs_deep_loss.png",
        value_fmt=".3f",
    )
    make_metric_plot(
        "test_spike_count",
        title="SCNN Base vs Deep: Spike Count by Encoding",
        ylabel="Average Spike Count",
        output_path=ANALYTICS_DIR / "scnn_base_vs_deep_spikes.png",
        value_fmt=".2f",
    )
    make_combined_plot(ANALYTICS_DIR / "scnn_base_vs_deep_combined.png")
    write_summary_markdown(ANALYTICS_DIR / "scnn_base_vs_deep_summary.md")

    print(ANALYTICS_DIR / "scnn_base_vs_deep_accuracy.png")
    print(ANALYTICS_DIR / "scnn_base_vs_deep_loss.png")
    print(ANALYTICS_DIR / "scnn_base_vs_deep_spikes.png")
    print(ANALYTICS_DIR / "scnn_base_vs_deep_combined.png")
    print(ANALYTICS_DIR / "scnn_base_vs_deep_summary.md")


if __name__ == "__main__":
    main()
