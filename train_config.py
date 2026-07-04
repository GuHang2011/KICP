# train_config.py
# -*- coding: utf-8 -*-

"""
使用 YAML 配置文件训练 KICP。

运行示例：

    python train_config.py --config configs/politifact_smoke.yaml

    python train_config.py --config configs/gossipcop_smoke.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import (
    set_seed,
    build_train_val_csv,
    save_checkpoint,
    count_total_parameters,
    count_trainable_parameters,
)

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def load_config(config_path: str):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def train_one_epoch(model, dataloader, optimizer, device, epoch):
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

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径，例如 configs/politifact_smoke.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(train_cfg.get("seed", 42))

    device_name = train_cfg.get("device", "auto")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=============== 当前配置 ===============")
    print(f"配置文件：{args.config}")
    print(f"数据集：{dataset_cfg['dataset_name']}")
    print(f"原始 CSV：{dataset_cfg['csv']}")
    print(f"输出目录：{train_cfg['output_dir']}")
    print(f"训练设备：{device}")
    print("=======================================")
    print()

    split_dir = Path(train_cfg["output_dir"]) / "splits"

    train_csv, val_csv = build_train_val_csv(
        csv_path=dataset_cfg["csv"],
        dataset_name=dataset_cfg["dataset_name"],
        output_dir=str(split_dir),
        max_samples=dataset_cfg.get("max_samples", 100),
        val_ratio=dataset_cfg.get("val_ratio", 0.2),
        seed=train_cfg.get("seed", 42),
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
        clip_model_name=model_cfg["clip_model_name"],
        max_length=77,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print()
    print("=============== 创建 KICPModel ===============")

    model = KICPModel(
        clip_model_name=model_cfg["clip_model_name"],
        class_names=model_cfg["class_names"],
        hidden_dim=model_cfg["hidden_dim"],
        prompt_length=model_cfg["prompt_length"],
        prompt_init_std=model_cfg["prompt_init_std"],
        coattention_heads=model_cfg["coattention_heads"],
        coattention_dropout=model_cfg["coattention_dropout"],
        coattention_ffn_dim=model_cfg["coattention_ffn_dim"],
        classifier_temperature=model_cfg["classifier_temperature"],
        freeze_clip=model_cfg["freeze_clip"],
        use_knowledge=model_cfg["use_knowledge"],
        knowledge_bank_size=model_cfg["knowledge_bank_size"],
        top_k_text=model_cfg["top_k_text"],
        top_k_image=model_cfg["top_k_image"],
    )

    model = model.to(device)

    print()
    print("参数统计：")
    print(f"总参数量：{count_total_parameters(model):,}")
    print(f"可训练参数量：{count_trainable_parameters(model):,}")
    print()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint_dir = Path(train_cfg["output_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0

    print("=============== 开始训练 ===============")

    for epoch in range(1, train_cfg["epochs"] + 1):
        print()
        print(f"========== Epoch {epoch}/{train_cfg['epochs']} ==========")

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