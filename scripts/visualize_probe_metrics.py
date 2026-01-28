#!/usr/bin/env python3
"""
Visualize linear probe results.

Usage:
  python scripts/visualize_probe_val_metrics.py path/to/results.json metric_name

Example:
  python scripts/visualize_probe_val_metrics.py outputs/probe_results.json cv_auc_mean
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def _extract_rows(training_info: dict) -> list[dict]:
    rows = []
    for rec in training_info.get("all_results", []):
        position = rec.get("position")
        layer = rec.get("layer")
        target = rec.get("target")
        metrics = rec.get("metrics", {}) or {}

        row = {"position": position, "layer": layer, "target": target}
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str, help="Path to results.json (training_info).")
    parser.add_argument("metric", type=str, help="Metric name, e.g. cv_auc_mean, test_auc, cv_logloss_mean.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for images (default: рядом с json).")
    parser.add_argument("--fmt", type=str, default="png", help="Image format: png/pdf/svg.")
    parser.add_argument("--title_prefix", type=str, default="", help="Optional title prefix.")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    training_info = json.loads(results_path.read_text(encoding="utf-8"))

    rows = _extract_rows(training_info)
    if not rows:
        raise SystemExit("No rows found in training_info['all_results'].")

    metric = args.metric
    available_metrics = sorted({k for r in rows for k in r.keys()})
    if metric not in available_metrics:
        raise SystemExit(
            f"Metric '{metric}' not found. Available keys include:\n"
            + ", ".join([m for m in available_metrics if m not in {"fold_auc", "fold_logloss"}])
        )

    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # target -> position -> list of (layer, metric_value)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get(metric) is None:
            continue
        grouped[r["target"]][r["position"]].append((r["layer"], r[metric]))

    for target, pos_dict in grouped.items():
        plt.figure(figsize=(8, 5))

        for position, pairs in sorted(pos_dict.items(), key=lambda x: x[0]):
            pairs = sorted(pairs, key=lambda x: x[0])
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            plt.plot(xs, ys, marker="o", linewidth=2, label=f"pos={position}")

        plt.xlabel("Layer")
        plt.ylabel(metric)
        title = f"{args.title_prefix}{target}: {metric} vs layer"
        plt.title(title.strip())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"probe_{target}_{metric}.{args.fmt}"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
