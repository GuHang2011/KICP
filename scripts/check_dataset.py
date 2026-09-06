# scripts/check_dataset.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 datasets/kicp_dataset.py 是否能正常读取数据。

运行示例：
    python scripts/check_dataset.py --csv data/processed/politifact.csv
    python scripts/check_dataset.py --csv data/processed/gossipcop.csv
"""

import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------
# 关键修复：
# 把项目根目录 D:/KICP_project 加入 Python 搜索路径。
# 否则从 scripts/ 目录运行时，Python 找不到 datasets 包。
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="处理后的 CSV 文件路径，例如 data/processed/politifact.csv",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="只检查前多少条样本，默认 5 条",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    dataset = KICPDataset(
        csv_path=str(csv_path),
        image_size=336,
        max_samples=args.max_samples,
    )

    print()
    print("=============== 开始检查样本 ===============")

    for i in range(len(dataset)):
        sample = dataset[i]

        print(f"\n样本 index = {i}")
        print(f"id    : {sample['id']}")
        print(f"text  : {sample['text'][:100]}...")
        print(f"label : {sample['label']}")
        print(f"image : {sample['image']}")
        print(f"image size: {sample['image'].size}")

    print()
    print("=============== Dataset 检查完成 ===============")
    print("如果你看到每条样本都有 id、text、label、image size=(336, 336)，说明 Dataset 类正常。")


if __name__ == "__main__":
    main()