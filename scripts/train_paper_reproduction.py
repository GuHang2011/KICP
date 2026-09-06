# scripts/train_paper_reproduction.py
# -*- coding: utf-8 -*-

"""
KICP 论文参数复现实验训练脚本。

用途：
    按论文实验设置训练 KICP：
        - CLIP ViT-L/14@336px
        - CLIP 主体冻结
        - hidden_dim = 768
        - prompt_length = 16
        - coattention_heads = 8
        - dropout = 0.6
        - AdamW
        - lr = 3e-5
        - weight_decay = 1e-3
        - epochs = 20
        - full supervision: 8:2 stratified split
        - validation best checkpoint

适用数据：
    data/processed/politifact_full.csv
    data/processed/gossipcop_full.csv

适用知识库：
    knowledge/wikidata_knowledge_politifact_embeddings.pt
    knowledge/wikidata_knowledge_gossipcop_embeddings.pt

输出：
    outputs/xxx/checkpoints/best.pt
    outputs/xxx/checkpoints/last.pt
    outputs/xxx/logs/train_log.csv
    outputs/xxx/logs/args.json
    outputs/xxx/splits/xxx_train.csv
    outputs/xxx/splits/xxx_val.csv

说明：
    本脚本是“工程复现版”的论文参数训练脚本。
    它使用当前项目已经构建好的真实新闻正文、真实图片和 Wikidata 知识向量。
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

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


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def balanced_sample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """max_samples=0 表示使用全部样本；其他情况按类别均衡采样。"""
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


def build_full_supervision_splits(
    csv_path: str,
    output_dir: Path,
    dataset_name: str,
    max_samples: int,
    val_ratio: float,
    seed: int,
) -> Tuple[Path, Path]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = ["id", "text", "image_path", "label"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列：{col}，当前列名：{list(df.columns)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    if len(df) == 0:
        raise ValueError("CSV 中没有可用样本。")

    df = balanced_sample(df=df, max_samples=max_samples, seed=seed)

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
    print("=============== Full Supervision 数据划分完成 ===============")
    print(f"输入 CSV：{csv_path}")
    print(f"总样本数：{len(df)}")
    print(f"训练集：{train_path}，样本数={len(train_df)}")
    print(f"验证集：{val_path}，样本数={len(val_df)}")
    print("训练集标签统计：")
    print(train_df["label"].value_counts().sort_index())
    print("验证集标签统计：")
    print(val_df["label"].value_counts().sort_index())
    print("============================================================")
    print()

    return train_path, val_path


def build_dataloader(csv_path: Path, clip_model_name: str, batch_size: int, shuffle: bool, num_workers: int):
    dataset = KICPDataset(str(csv_path))
    collator = KICPCollator(clip_model_name=clip_model_name, max_length=77)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )
    return loader


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

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
    f1_binary = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1_macro, "f1_binary": f1_binary}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, epoch: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    progress = tqdm(dataloader, desc=f"Eval Epoch {epoch}", leave=False)

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
    f1_binary = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1_macro, "f1_binary": f1_binary}


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, best_f1: float, args: argparse.Namespace) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_f1": best_f1,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--knowledge_embedding_path", type=str, required=True)

    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--prompt_length", type=int, default=16)
    parser.add_argument("--coattention_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    parser.add_argument("--top_k_text", type=int, default=5)
    parser.add_argument("--top_k_image", type=int, default=5)

    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

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

    train_path, val_path = build_full_supervision_splits(
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
        hidden_dim=args.hidden_dim,
        prompt_length=args.prompt_length,
        prompt_init_std=0.02,
        coattention_heads=args.coattention_heads,
        coattention_dropout=args.dropout,
        classifier_temperature=0.07,
        freeze_clip=True,
        use_knowledge=True,
        top_k_text=args.top_k_text,
        top_k_image=args.top_k_image,
        knowledge_embedding_path=args.knowledge_embedding_path,
    )
    model.to(device)

    total_params, trainable_params = count_parameters(model)

    print()
    print("================ KICP 论文复现实验参数 ================")
    print(f"dataset_name              : {args.dataset_name}")
    print(f"csv                       : {args.csv}")
    print(f"knowledge_embedding_path  : {args.knowledge_embedding_path}")
    print(f"knowledge_mode            : {model.knowledge_mode}")
    print(f"clip_model_name           : {args.clip_model_name}")
    print(f"hidden_dim                : {args.hidden_dim}")
    print(f"prompt_length             : {args.prompt_length}")
    print(f"coattention_heads         : {args.coattention_heads}")
    print(f"dropout                   : {args.dropout}")
    print(f"epochs                    : {args.epochs}")
    print(f"lr                        : {args.lr}")
    print(f"weight_decay              : {args.weight_decay}")
    print(f"top_k_text                : {args.top_k_text}")
    print(f"top_k_image               : {args.top_k_image}")
    print("freeze_clip               : True")
    print(f"total_params              : {total_params:,}")
    print(f"trainable_params          : {trainable_params:,}")
    print("======================================================")
    print()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = -1
    logs = []

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"================ Epoch {epoch}/{args.epochs} ================")
        train_metrics = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device, epoch=epoch)
        val_metrics = evaluate(model=model, dataloader=val_loader, device=device, epoch=epoch)

        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", model=model, epoch=epoch, best_f1=best_f1, args=args)
            print(f"保存 best.pt：epoch={epoch}, best_f1={best_f1:.4f}")

        save_checkpoint(checkpoint_dir / "last.pt", model=model, epoch=epoch, best_f1=best_f1, args=args)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_f1_binary": train_metrics["f1_binary"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_binary": val_metrics["f1_binary"],
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "is_best": int(improved),
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        print(
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"train_f1_macro={train_metrics['f1_macro']:.4f}, "
            f"train_f1_binary={train_metrics['f1_binary']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}, "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}, "
            f"val_f1_binary={val_metrics['f1_binary']:.4f} | "
            f"best_f1={best_f1:.4f}, best_epoch={best_epoch}"
        )

    summary = {
        "dataset_name": args.dataset_name,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "csv": args.csv,
        "knowledge_embedding_path": args.knowledge_embedding_path,
        "output_dir": str(output_dir),
        "train_log": str(log_dir / "train_log.csv"),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
    }
    with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("================ 训练完成 ================")
    print(f"best_f1       : {best_f1:.4f}")
    print(f"best_epoch    : {best_epoch}")
    print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"last checkpoint: {checkpoint_dir / 'last.pt'}")
    print(f"train log       : {log_dir / 'train_log.csv'}")
    print(f"summary         : {log_dir / 'summary.json'}")
    print("==========================================")


if __name__ == "__main__":
    main()
