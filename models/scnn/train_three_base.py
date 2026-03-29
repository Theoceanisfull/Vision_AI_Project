from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SNNConfig, default_config
from .t_loop import run_training

ENCODINGS = ("rate", "latency", "delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3 base Conv-SNN models: rate, latency, delta")
    parser.add_argument(
        "--base-config",
        type=str,
        default="models/scnn/default_config.json",
        help="Path to a base JSON config",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="runs/scnn/base_encoding_comparison/summary.json",
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke run (1 epoch, tiny subset) for all encodings",
    )
    return parser.parse_args()


def load_config(path: str) -> SNNConfig:
    p = Path(path)
    if p.exists():
        return SNNConfig.from_json(p)
    return default_config()


def apply_quick_overrides(cfg: SNNConfig) -> None:
    cfg.train.epochs = 1
    cfg.data.batch_size = 4
    cfg.data.num_steps = 8
    cfg.train.max_train_batches = 2
    cfg.train.max_val_batches = 1
    cfg.train.max_test_batches = 1


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)

    records: list[dict] = []

    for enc in ENCODINGS:
        cfg = SNNConfig.from_dict(base_cfg.to_dict())
        cfg.model.architecture = "base"
        cfg.data.encoding = enc
        cfg.result.run_name = f"base_{enc}" if not args.quick else f"base_{enc}_quick"

        if args.quick:
            apply_quick_overrides(cfg)

        print(f"\n=== Training encoding: {enc} ===")
        result = run_training(cfg)

        record = {
            "encoding": enc,
            "run_dir": result["run_dir"],
            "test": result["test"],
        }
        records.append(record)

    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
