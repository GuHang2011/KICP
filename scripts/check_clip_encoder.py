# scripts/check_clip_encoder.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 CLIP 特征提取模块是否能正常运行。

运行示例：
    python scripts/check_clip_encoder.py --csv data/processed/politifact.csv

    python scripts/check_clip_encoder.py --csv data/processed/gossipcop.csv

输出：
    text_seq_features shape
    image_seq_features shape
    text_pooled_features shape
    image_pooled_features shape

如果这些 shape 正常，说明 CLIP 特征提取模块可以使用。
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.clip_encoder import KICPCLIPEncoder


def count_trainable_parameters(model):
    """
    统计可训练参数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    """
    统计总参数量。
    """
    return sum(p.numel() for p in model.parameters())


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
        default=2,
        help="batch size，默认 2。CLIP-Large 较大，新手建议先用 2。",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=4,
        help="只测试前多少条样本，默认 4。",
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
        help="运行设备。auto 表示有 GPU 用 GPU，否则用 CPU。",
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
    print("=============== 创建 CLIP Encoder ===============")

    encoder = KICPCLIPEncoder(
        clip_model_name=args.clip_model_name,
        hidden_dim=768,
        freeze_clip=True,
        normalize_features=True,
    )

    encoder = encoder.to(device)
    encoder.eval()

    total_params = count_total_parameters(encoder)
    trainable_params = count_trainable_parameters(encoder)

    print()
    print("参数统计：")
    print(f"总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")
    print("说明：CLIP 主体已经冻结，但如果 vision hidden size 需要投影到 768，projection 层仍然可训练。")

    print()
    print("=============== 读取一个 batch ===============")

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    print("输入 shape：")
    print("input_ids      :", input_ids.shape)
    print("attention_mask :", attention_mask.shape)
    print("pixel_values   :", pixel_values.shape)

    print()
    print("=============== 执行 CLIP 前向传播 ===============")

    with torch.no_grad():
        features = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    print()
    print("输出 feature keys:")
    print(features.keys())

    print()
    print("text_seq_features shape:")
    print(features["text_seq_features"].shape)

    print()
    print("image_seq_features shape:")
    print(features["image_seq_features"].shape)

    print()
    print("text_pooled_features shape:")
    print(features["text_pooled_features"].shape)

    print()
    print("image_pooled_features shape:")
    print(features["image_pooled_features"].shape)

    print()
    print("=============== CLIP Encoder 检查完成 ===============")
    print("如果你看到：")
    print("text_seq_features  : [batch_size, text_length, 768]")
    print("image_seq_features : [batch_size, image_tokens, 768]")
    print("text_pooled_features  : [batch_size, 768]")
    print("image_pooled_features : [batch_size, 768]")
    print("说明本步骤成功。")


if __name__ == "__main__":
    main()