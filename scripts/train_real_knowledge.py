# scripts/train_real_knowledge.py
# -*- coding: utf-8 -*-

"""
使用真实 Wikidata 知识向量训练 KICP 模型。

特点：
    1. 支持 full CSV：
        data/processed/politifact_full.csv
        data/processed/gossipcop_full.csv

    2. 支持真实 Wikidata 知识向量：
        knowledge/wikidata_knowledge_politifact_embeddings.pt

    3. 训练时模型内部会执行：
        text Top-K 知识检索
        image Top-K 知识检索
        知识向量加权融合

    4. 自动保存：
        best.pt
        last.pt
        train_log.csv
        args.json
        train/val split csv
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def balanced_sample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.copy()

    labels = sorted(df["label"].unique().tolist())
    n_labels = len(labels)

    base = max_samples // n_labels
    remainder = max_samples % n_labels

    parts = []

    for i, label in enumerate(labels):
        label_df = df[df["label"] == label].copy()
        n = base + (1 if i < remainder else 0)
        n = min(n, len(label_df))

        if n > 0:
            parts.append(label_df.sample(n=n, random_state=seed))

    sampled = pd.concat(parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return sampled


def build_splits(
    csv_path: str,
    output_dir: Path,
    dataset_name: str,
    max_samples: int,
    val_ratio: float,
    seed: int,
):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["id", "text", "image_path", "label"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列：{col}，当前列名：{list(df.columns)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = balanced_sample(
        df=df,
        max_samples=max_samples,
        seed=seed,
    )

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    split_dir = output_dir / "splits"
    ensure_dir(split_dir)

    train_path = split_dir / f"{dataset_name}_train.csv"
    val_path = split_dir / f"{dataset_name}_val.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

    print()
    print("=============== 数据划分完成 ===============")
    print(f"原始 CSV：{csv_path}")
    print(f"训练集：{train_path}，样本数={len(train_df)}")
    print(f"验证集：{val_path}，样本数={len(val_df)}")
    print("训练集标签统计：")
    print(train_df["label"].value_counts().sort_index())
    print("验证集标签统计：")
    print(val_df["label"].value_counts().sort_index())
    print("============================================")
    print()

    return train_path, val_path


def build_dataloader(
    csv_path: Path,
    clip_model_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    dataset = KICPDataset(str(csv_path))

    collator = KICPCollator(
        clip_model_name=clip_model_name,
        max_length=77,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return loader


def move_batch_to_device(batch, device):
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            return_features=False,
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(outputs["probs"], dim=-1)

        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        progress.set_postfix(loss=float(loss.item()))

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, desc="Eval", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            return_features=False,
        )

        loss = outputs["loss"]

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(outputs["probs"], dim=-1)

        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--knowledge_embedding_path",
        type=str,
        required=True,
        help="真实 Wikidata 知识向量 .pt 文件。",
    )

    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--top_k_text", type=int, default=5)
    parser.add_argument("--top_k_image", type=int, default=3)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)

    with open(log_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    train_path, val_path = build_splits(
        csv_path=args.csv,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = build_dataloader(
        csv_path=train_path,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = build_dataloader(
        csv_path=val_path,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = KICPModel(
        clip_model_name=args.clip_model_name,
        class_names=["real", "fake"],
        hidden_dim=768,
        prompt_length=16,
        coattention_heads=4,
        use_knowledge=True,
        top_k_text=args.top_k_text,
        top_k_image=args.top_k_image,
        knowledge_embedding_path=args.knowledge_embedding_path,
        freeze_clip=True,
    )

    model.to(device)

    total_params, trainable_params = count_parameters(model)

    print()
    print("=============== 模型信息 ===============")
    print(f"knowledge_mode: {model.knowledge_mode}")
    print(f"knowledge_embedding_path: {args.knowledge_embedding_path}")
    print(f"top_k_text: {args.top_k_text}")
    print(f"top_k_image: {args.top_k_image}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")
    print("========================================")
    print()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    logs = []

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "best_f1": best_f1,
        }

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            row["best_f1"] = best_f1

            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                checkpoint_dir / "best.pt",
            )

            print(f"保存 best.pt，best_f1={best_f1:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            },
            checkpoint_dir / "last.pt",
        )

        logs.append(row)
        pd.DataFrame(logs).to_csv(log_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        print(
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"train_f1={train_metrics['f1_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}, "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

    print()
    print("=============== 训练完成 ===============")
    print(f"best_f1: {best_f1:.4f}")
    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"last checkpoint: {checkpoint_dir / 'last.pt'}")
    print(f"train log: {log_dir / 'train_log.csv'}")
    print("========================================")


if __name__ == "__main__":
    main()
