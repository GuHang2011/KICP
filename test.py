# test.py
# -*- coding: utf-8 -*-

"""
KICP 测试脚本。

运行示例：

1. 测试 PolitiFact 验证集：
    python test.py --test_csv outputs/splits/politifact_val.csv --checkpoint outputs/checkpoints/best.pt --device cpu

2. 如果用 GPU：
    python test.py --test_csv outputs/splits/politifact_val.csv --checkpoint outputs/checkpoints/best.pt --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    all_ids = []
    all_texts = []
    all_labels = []
    all_preds = []
    all_probs_real = []
    all_probs_fake = []

    progress_bar = tqdm(
        dataloader,
        desc="Testing",
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
        probs = outputs["probs"]
        preds = torch.argmax(probs, dim=-1)

        total_loss += loss.item()

        all_ids.extend(batch["ids"])
        all_texts.extend(batch["texts"])
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs_real.extend(probs[:, 0].cpu().numpy().tolist())
        all_probs_fake.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_binary = f1_score(all_labels, all_preds, average="binary", pos_label=1)

    results_df = pd.DataFrame(
        {
            "id": all_ids,
            "text": all_texts,
            "label": all_labels,
            "pred": all_preds,
            "prob_real": all_probs_real,
            "prob_fake": all_probs_fake,
        }
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "labels": all_labels,
        "preds": all_preds,
        "results_df": results_df,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="测试集 CSV，例如 outputs/splits/politifact_val.csv",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint，例如 outputs/checkpoints/best.pt",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/test_predictions.csv",
        help="预测结果保存路径",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("当前运行设备：", device)
    print()

    test_csv = Path(args.test_csv)
    checkpoint_path = Path(args.checkpoint)

    if not test_csv.exists():
        raise FileNotFoundError(f"找不到测试 CSV：{test_csv}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    print("=============== 创建测试 Dataset ===============")

    test_dataset = KICPDataset(
        csv_path=str(test_csv),
        image_size=336,
        max_samples=None,
    )

    print()
    print("=============== 创建 Collator ===============")

    collator = KICPCollator(
        clip_model_name=args.clip_model_name,
        max_length=77,
    )

    test_loader = DataLoader(
        test_dataset,
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

    print()
    print("=============== 加载 checkpoint ===============")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)

    print(f"已加载模型：{checkpoint_path}")
    print(f"checkpoint epoch：{checkpoint.get('epoch', 'unknown')}")
    print(f"checkpoint best_f1：{checkpoint.get('best_f1', 'unknown')}")

    print()
    print("=============== 开始测试 ===============")

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    print()
    print("=============== 测试结果 ===============")
    print(f"loss      : {metrics['loss']:.4f}")
    print(f"accuracy  : {metrics['accuracy']:.4f}")
    print(f"f1_macro  : {metrics['f1_macro']:.4f}")
    print(f"f1_binary : {metrics['f1_binary']:.4f}")

    print()
    print("分类报告：")
    print(
        classification_report(
            metrics["labels"],
            metrics["preds"],
            target_names=["real", "fake"],
            digits=4,
            zero_division=0,
        )
    )

    print("混淆矩阵：")
    print(confusion_matrix(metrics["labels"], metrics["preds"]))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics["results_df"].to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"预测结果已保存：{output_csv}")


if __name__ == "__main__":
    main()