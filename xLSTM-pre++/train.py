#!/usr/bin/env python
"""Train xLSTM-pre++ model variants."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xlstm_prepp.data import create_dataset, get_display_name, load_config, resolve_path
from xlstm_prepp.runtime import create_model
from xlstm_prepp.training.trainer import Trainer


def _make_loader(dataset, collate_fn, batch_size: int, shuffle: bool, num_workers: int, use_pin_memory: bool, prefetch_factor: int):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train xLSTM-pre++ variants")
    parser.add_argument("--config", type=str, default="configs/GAT-xLSTM-K-I++.yaml", help="Config path")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    project_root = Path(__file__).resolve().parent
    args.config = resolve_path(args.config, str(project_root))
    if args.resume is not None:
        args.resume = resolve_path(args.resume, str(project_root))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    config = load_config(args.config)
    print(f"加载配置: {args.config}")
    print(f"创建模型: {get_display_name(config)}")

    train_dataset, collate_fn = create_dataset(config, split="train")
    val_dataset, _ = create_dataset(config, split="val")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 64)
    num_workers = train_config.get("num_workers", 4)
    prefetch_factor = train_config.get("prefetch_factor", 2)
    use_pin_memory = args.device.startswith("cuda")
    train_loader = _make_loader(train_dataset, collate_fn, batch_size, True, num_workers, use_pin_memory, prefetch_factor)
    val_loader = _make_loader(val_dataset, collate_fn, batch_size, False, num_workers, use_pin_memory, prefetch_factor)

    model = create_model(config)
    num_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"模型参数量: {num_params:,}")

    trainer = Trainer(model, config, device=args.device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"恢复训练: {args.resume}")

    trainer.train(train_loader, val_loader)
    print("训练完成")


if __name__ == "__main__":
    main()
