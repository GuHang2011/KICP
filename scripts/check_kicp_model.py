# scripts/check_kicp_model.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 KICP 主模型是否能完整前向传播和反向传播。

运行示例：
    python scripts/check_kicp_model.py --csv data/processed/politifact.csv --batch_size 1

如果显存不足：
    python scripts/check_kicp_model.py --csv data/processed/politifact.csv --batch_size 1 --device cpu

预期输出：
    logits shape: torch.Size([1, 2])
    probs shape : torch.Size([1, 2])
    loss        : 一个正常数字
    反向传播成功
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_first_grad_parameter(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            return name, param.grad.shape
    return None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="处理后的 CSV 文件路径，例如 data/processed/politifact.csv",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size，默认 1。CLIP-Large 很大，建议先用 1。",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=2,
        help="只测试前多少条样本，默认 2。",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP 模型名称或本地路径。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备。",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("当前运行设备：", device)

    print()
    print("=============== 创建 Dataset ===============")

    dataset = KICPDataset(
        csv_path=args.csv,
        image_size=336,
        max_samples=args.max_samples,
    )

    print()
    print("=============== 创建 Collator ===============")

    collator = KICPCollator(
        clip_model_name=args.clip_model_name,
        max_length=77,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print()
    print("=============== 创建 KICPModel ===============")

    model = KICPModel(
        clip_model_name=args.clip_model_name,
        class_names=["real", "fake"],
        hidden_dim=768,
        prompt_length=16,
        prompt_init_std=0.02,
        coattention_heads=4,
        coattention_dropout=0.1,
        coattention_ffn_dim=2048,
        classifier_temperature=0.07,
        freeze_clip=True,
    )

    model = model.to(device)
    model.train()

    print()
    print("参数统计：")
    print(f"总参数量：{count_total_parameters(model):,}")
    print(f"可训练参数量：{count_trainable_parameters(model):,}")

    print()
    print("=============== 读取一个 batch ===============")

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    print("输入 shape：")
    print("input_ids      :", input_ids.shape)
    print("attention_mask :", attention_mask.shape)
    print("pixel_values   :", pixel_values.shape)
    print("labels         :", labels.shape)
    print("labels 内容    :", labels)

    print()
    print("=============== 执行 KICP 前向传播 ===============")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_features=True,
    )

    print("outputs keys:")
    print(outputs.keys())

    print()
    print("logits shape:")
    print(outputs["logits"].shape)

    print()
    print("probs shape:")
    print(outputs["probs"].shape)

    print()
    print("loss:")
    print(outputs["loss"].item())

    print()
    print("fused_features shape:")
    print(outputs["fused_features"].shape)

    print()
    print("class_features shape:")
    print(outputs["class_features"].shape)

    print()
    print("logits:")
    print(outputs["logits"])

    print()
    print("probs:")
    print(outputs["probs"])

    print()
    print("=============== 反向传播检查 ===============")

    loss = outputs["loss"]
    loss.backward()

    grad_name, grad_shape = find_first_grad_parameter(model)

    if grad_name is not None:
        print("至少有一个可训练参数获得梯度：")
        print(f"{grad_name}, grad shape = {grad_shape}")
        print("反向传播成功。")
    else:
        print("没有找到梯度，请检查模型。")

    print()
    print("=============== KICP 主模型检查完成 ===============")
    print("如果你看到：")
    print("logits shape = [batch_size, 2]")
    print("probs shape  = [batch_size, 2]")
    print("fused_features shape = [batch_size, 768]")
    print("class_features shape = [2, 768]")
    print("并且反向传播成功，说明本步骤成功。")


if __name__ == "__main__":
    main()