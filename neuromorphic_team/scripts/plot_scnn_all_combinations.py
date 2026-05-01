from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = PROJECT_ROOT / "Analytics"

CSV_BY_ENCODING = {
    "rate": ANALYTICS_DIR / "rate_progression.csv",
    "latency": ANALYTICS_DIR / "latency_progression.csv",
    "delta": ANALYTICS_DIR / "delta_progression.csv",
}

GROUP_ORDER = {
    "rate": ["SCNN base", "SCNN deep", "SCNN x2 base", "SCNN x2 deep"],
    "latency": ["SCNN base", "SCNN deep"],
    "delta": ["SCNN base", "SCNN deep"],
}

DISPLAY_NAMES = {
    "rate": "Rate",
    "latency": "Latency",
    "delta": "Delta",
}

SHORT_LABELS = {
    "SCNN base": "Base",
    "SCNN deep": "Deep",
    "SCNN x2 base": "x2 Base",
    "SCNN x2 deep": "x2 Deep",
}

COLORS = {
    "SCNN base": "#4C78A8",
    "SCNN deep": "#F58518",
    "SCNN x2 base": "#54A24B",
    "SCNN x2 deep": "#E45756",
}

# Exact counts are recoverable for the first three from the checked-in configs.
# The original x2 deep run folder is no longer present locally, so the saved
# analytics only preserve it as an extended "x2" run rather than a numeric count.
EPOCH_TAGS = {
    "SCNN base": "5e",
    "SCNN deep": "10e",
    "SCNN x2 base": "10e",
    "SCNN x2 deep": "10e",
}

METRICS = {
    "accuracy": {
        "row_key": "test_acc",
        "scale": 100.0,
        "title": "SCNN Accuracy Across All Completed Encoding Combinations",
        "ylabel": "Test Accuracy (%)",
        "value_fmt": "{value:.1f}%",
        "summary_title": "# SCNN Accuracy Across All Completed Combinations",
        "summary_value_name": "Test Accuracy",
        "summary_value_fmt": "{value:.2f}%",
        "output_name": "scnn_all_combinations_accuracy",
    },
    "spike_count": {
        "row_key": "test_spike_count",
        "scale": 1.0,
        "title": "SCNN Spike Count Across All Completed Encoding Combinations",
        "ylabel": "Average Test Spike Count",
        "value_fmt": "{value:.2f}",
        "summary_title": "# SCNN Spike Count Across All Completed Combinations",
        "summary_value_name": "Average Test Spike Count",
        "summary_value_fmt": "{value:.2f}",
        "output_name": "scnn_all_combinations_spike_count",
    },
}


def load_rows() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}

    for encoding, csv_path in CSV_BY_ENCODING.items():
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["family"] == "SCNN"]

        data[encoding] = {
            row["label"]: {
                "test_acc": float(row["test_acc"]),
                "test_loss": float(row["test_loss"]),
                "test_spike_count": float(row["test_spike_count"]),
            }
            for row in rows
        }

    return data


def build_series(metric_name: str):
    metric = METRICS[metric_name]
    data = load_rows()
    bars = []
    x = 0
    group_centers = []
    separators = []

    for encoding, labels in GROUP_ORDER.items():
        group_start = x
        for label in labels:
            row = data[encoding].get(label)
            if row is None:
                continue
            bars.append(
                {
                    "x": x,
                    "encoding": encoding,
                    "label": label,
                    "value": row[metric["row_key"]] * metric["scale"],
                    "tick": f"{SHORT_LABELS[label]}\n{EPOCH_TAGS[label]}",
                }
            )
            x += 1
        group_end = x - 1
        group_centers.append(((group_start + group_end) / 2.0, DISPLAY_NAMES[encoding]))
        separators.append(x - 0.5)
        x += 1

    return bars, group_centers, separators[:-1]


def make_plot(metric_name: str, output_path: Path) -> None:
    metric = METRICS[metric_name]
    bars, group_centers, separators = build_series(metric_name)
    x_positions = [bar["x"] for bar in bars]
    heights = [bar["value"] for bar in bars]
    colors = [COLORS[bar["label"]] for bar in bars]
    tick_labels = [bar["tick"] for bar in bars]

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    bar_container = ax.bar(x_positions, heights, color=colors, width=0.78)

    for patch, bar in zip(bar_container, bars):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + 1.2,
            metric["value_fmt"].format(value=bar["value"]),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for xpos in separators:
        ax.axvline(xpos, color="#D0D0D0", linewidth=1.0)

    ymax = max(heights) if heights else 1.0
    for center, label in group_centers:
        ax.text(center, -0.14 * ymax, label, ha="center", va="top", fontsize=11, fontweight="bold")

    handles = []
    seen_labels = set()
    for bar in bars:
        label = bar["label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=COLORS[label], label=f"{label} ({EPOCH_TAGS[label]})")
        )

    ax.set_title(metric["title"], fontsize=15)
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(x_positions, tick_labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.22)
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(metric_name: str, output_path: Path) -> None:
    metric = METRICS[metric_name]
    bars, _, _ = build_series(metric_name)
    lines = [
        metric["summary_title"],
        "",
        "This chart combines every completed SCNN encoding run saved in the current project artifacts.",
        "",
        f"| Encoding | Model | Epoch Tag | {metric['summary_value_name']} |",
        "| --- | --- | --- | ---: |",
    ]

    for bar in bars:
        lines.append(
            f"| {DISPLAY_NAMES[bar['encoding']]} | {bar['label']} | {EPOCH_TAGS[bar['label']]} | {metric['summary_value_fmt'].format(value=bar['value'])} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    for metric_name, metric in METRICS.items():
        stem = metric["output_name"]
        chart_path = ANALYTICS_DIR / f"{stem}.png"
        summary_path = ANALYTICS_DIR / f"{stem}.md"

        make_plot(metric_name, chart_path)
        write_summary(metric_name, summary_path)

        print(chart_path)
        print(summary_path)


if __name__ == "__main__":
    main()
