# train.py
# -*- coding: utf-8 -*-

"""
KICP 训练脚本。

运行示例：

1. PolitiFact 小样本 smoke test：
    python train.py --csv data/processed/politifact.csv --dataset_name politifact --max_samples 20 --epochs 1 --batch_size 1 --device cpu

2. GossipCop 小样本 smoke test：
    python train.py --csv data/processed/gossipcop.csv --dataset_name gossipcop --max_samples 20 --epochs 1 --batch_size 1 --device cpu

说明：
    当前阶段先用小样本测试训练流程。
    等训练流程跑通后，再扩大 max_samples、epochs。
"""

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_train_val_csv(
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    max_samples: int,
    val_ratio: float,
    seed: int,
):
    """
    从 data/processed/*.csv 中划分 train / val。

    注意：
        你的 processed CSV 是先 fake 后 real。
        如果直接取前 max_samples 条，会全是 fake。
        所以这里先按 label 分层抽样，再划分训练集和验证集。
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = ["id", "text", "image_path", "label"]
    df = df[required_columns].copy()

    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df["label"] = df["label"].astype(int)

    print("原始数据标签统计：")
    print(df["label"].value_counts().sort_index())
    print()

    if max_samples is not None and max_samples > 0 and max_samples < len(df):
        sampled_parts = []

        labels = sorted(df["label"].unique().tolist())
        per_class = max_samples // len(labels)

        if per_class < 2:
            raise ValueError("max_samples 太小，每个类别至少建议 2 条以上。")

        for label in labels:
            part = df[df["label"] == label]

            n = min(per_class, len(part))

            sampled_part = part.sample(
                n=n,
                random_state=seed,
            )

            sampled_parts.append(sampled_part)

        df = pd.concat(sampled_parts, axis=0)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        print(f"使用小样本训练，max_samples={max_samples}")
        print("采样后标签统计：")
        print(df["label"].value_counts().sort_index())
        print()

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_path = output_dir / f"{dataset_name}_train.csv"
    val_path = output_dir / f"{dataset_name}_val.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

    print("训练集标签统计：")
    print(train_df["label"].value_counts().sort_index())
    print()

    print("验证集标签统计：")
    print(val_df["label"].value_counts().sort_index())
    print()

    print(f"训练集文件：{train_path}")
    print(f"验证集文件：{val_path}")
    print()

    return str(train_path), str(val_path)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch}",
        ncols=100,
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_features=False,
        )

        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs["probs"], dim=-1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
            }
        )

    avg_loss = total_loss / max(len(dataloader), 1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc="Evaluate",
        ncols=100,
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_features=False,
        )

        loss = outputs["loss"]

        total_loss += loss.item()

        preds = torch.argmax(outputs["probs"], dim=-1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_f1,
    output_dir,
    filename,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "best_f1": best_f1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )

    print(f"模型已保存：{save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="处理后的 CSV，例如 data/processed/politifact.csv",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="数据集名称，例如 politifact 或 gossipcop",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="小样本训练数量。先用 20 测试流程。",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("当前运行设备：", device)
    print()

    split_dir = Path(args.output_dir) / "splits"

    train_csv, val_csv = build_train_val_csv(
        csv_path=args.csv,
        dataset_name=args.dataset_name,
        output_dir=str(split_dir),
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("=============== 创建 Dataset ===============")

    train_dataset = KICPDataset(
        csv_path=train_csv,
        image_size=336,
        max_samples=None,
    )

    val_dataset = KICPDataset(
        csv_path=val_csv,
        image_size=336,
        max_samples=None,
    )

    print()
    print("=============== 创建 Collator ===============")

    collator = KICPCollator(
        clip_model_name=args.clip_model_name,
        max_length=77,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
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
        use_knowledge=True,
        knowledge_bank_size=64,
        top_k_text=5,
        top_k_image=5,
    )

    model = model.to(device)

    print()
    print("参数统计：")
    print(f"总参数量：{count_total_parameters(model):,}")
    print(f"可训练参数量：{count_trainable_parameters(model):,}")
    print()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=============== 开始训练 ===============")

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print()
        print("训练结果：")
        print(
            f"loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}"
        )

        print("验证结果：")
        print(
            f"loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"f1={val_metrics['f1']:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_f1=best_f1,
            output_dir=checkpoint_dir,
            filename="last.pt",
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_f1=best_f1,
                output_dir=checkpoint_dir,
                filename="best.pt",
            )

            print(f"当前 best_f1 更新为：{best_f1:.4f}")

    print()
    print("=============== 训练完成 ===============")
    print(f"best_f1 = {best_f1:.4f}")
    print(f"最佳模型保存位置：{checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()