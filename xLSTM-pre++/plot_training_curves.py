#!/usr/bin/env python
"""Plot training curves from training_history.json."""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def has_metric(history, key):
    return any(key in record for record in history)


def plot_series(ax, epochs, history, keys, title, ylabel=None):
    plotted = False
    for key in keys:
        if has_metric(history, key):
            ax.plot(epochs, [record.get(key, 0.0) for record in history], label=key, linewidth=2)
            plotted = True
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def save_single_plot(output_path, epochs, history, keys, title, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, epochs, history, keys, title, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def resolve_output_dir(history_path: Path, output_arg: Optional[str]) -> Path:
    if output_arg is None:
        return history_path.with_name(f"{history_path.stem}_plots")
    output_path = Path(output_arg).resolve()
    if output_path.suffix:
        return output_path.parent / output_path.stem
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--history", type=str, required=True, help="Path to training_history.json")
    parser.add_argument("--output", type=str, default=None, help="Output directory or output file prefix")
    args = parser.parse_args()

    history_path = Path(args.history).resolve()
    with open(history_path, "r") as file_obj:
        history = json.load(file_obj)
    if not history:
        raise ValueError("history 为空，无法绘图")

    epochs = [record["epoch"] for record in history]
    output_dir = resolve_output_dir(history_path, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("total_loss.png", ["train_loss", "val_loss"], "Total Loss", "Loss"),
        ("reg_loss.png", ["train_reg", "val_reg"], "Regression Loss", "Loss"),
        ("cls_loss.png", ["train_cls", "val_cls"], "Classification Loss", "Loss"),
        ("end_loss.png", ["train_end", "val_end"], "Endpoint Loss", "Loss"),
        ("cls_soft_loss.png", ["train_cls_soft", "val_cls_soft"], "Soft Classification Loss", "Loss"),
        ("cls_hard_loss.png", ["train_cls_hard", "val_cls_hard"], "Hard Classification Loss", "Loss"),
        ("inertia_loss.png", ["train_inertia", "val_inertia"], "Inertia Loss", "Loss"),
        ("ctrl_smooth_loss.png", ["train_ctrl_smooth", "val_ctrl_smooth"], "Control Smoothness Loss", "Loss"),
        ("ade_at_1.png", ["ADE@1"], "ADE@1", "ADE"),
        ("fde_at_1.png", ["FDE@1"], "FDE@1", "FDE"),
        ("minade_at_k.png", ["minADE@K"], "minADE@K", "minADE"),
        ("minfde_at_k.png", ["minFDE@K"], "minFDE@K", "minFDE"),
        ("safety_metrics.png", ["Intrusion@1", "minIntrusion@K", "MR@2m"], "Safety Metrics", "Metric"),
    ]

    saved_files = []
    for filename, keys, title, ylabel in plot_specs:
        if not any(has_metric(history, key) for key in keys):
            continue
        output_path = output_dir / filename
        save_single_plot(output_path, epochs, history, keys, title, ylabel=ylabel)
        saved_files.append(output_path.name)

    print(f"已保存 {len(saved_files)} 张曲线图到: {output_dir}")
    for filename in saved_files:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
