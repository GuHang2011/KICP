# scripts/check_dataloader.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 Dataset + Collator + DataLoader 是否能正常工作。

运行示例：
    python scripts/check_dataloader.py --csv data/processed/politifact.csv

    python scripts/check_dataloader.py --csv data/processed/gossipcop.csv

输出内容：
    input_ids shape
    attention_mask shape
    pixel_values shape
    labels shape

如果这些 shape 正常，说明数据已经可以送入 CLIP 模型。
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator


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
        default=4,
        help="batch size，默认 4",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help="只测试前多少条样本，默认 8",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP 模型名称或本地路径",
    )

    args = parser.parse_args()

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

    print()
    print("=============== 创建 DataLoader ===============")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print("DataLoader 创建成功。")

    print()
    print("=============== 读取一个 batch ===============")

    batch = next(iter(dataloader))

    print("batch keys:")
    print(batch.keys())

    print()
    print("input_ids shape:")
    print(batch["input_ids"].shape)

    print()
    print("attention_mask shape:")
    print(batch["attention_mask"].shape)

    print()
    print("pixel_values shape:")
    print(batch["pixel_values"].shape)

    print()
    print("labels shape:")
    print(batch["labels"].shape)

    print()
    print("ids:")
    print(batch["ids"])

    print()
    print("texts 前 2 条:")
    for text in batch["texts"][:2]:
        print("-", text[:100])

    print()
    print("labels:")
    print(batch["labels"])

    print()
    print("=============== DataLoader 检查完成 ===============")
    print("如果你看到下面这些 shape，说明本步骤成功：")
    print("input_ids:      [batch_size, text_length]")
    print("attention_mask: [batch_size, text_length]")
    print("pixel_values:   [batch_size, 3, 336, 336]")
    print("labels:         [batch_size]")


if __name__ == "__main__":
    main()