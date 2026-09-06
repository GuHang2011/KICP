# scripts/test_real_knowledge.py
# -*- coding: utf-8 -*-

"""
测试真实 Wikidata 知识增强版 KICP 模型。

输入：
    --test_csv outputs/politifact_real_knowledge/splits/politifact_real_knowledge_val.csv
    --checkpoint outputs/politifact_real_knowledge/checkpoints/best.pt
    --knowledge_embedding_path knowledge/wikidata_knowledge_politifact_embeddings.pt

输出：
    评估指标 + 预测 CSV
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def move_batch_to_device(batch, device):
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_ids = []
    all_texts = []
    all_labels = []
    all_preds = []
    all_probs = []

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Test"):
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            return_features=False,
        )

        probs = outputs["probs"]
        preds = torch.argmax(probs, dim=-1)

        total_loss += outputs["loss"].item() * batch["labels"].size(0)

        all_ids.extend(batch["ids"])
        all_texts.extend(batch["texts"])
        all_labels.extend(batch["labels"].detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)

    return {
        "ids": all_ids,
        "texts": all_texts,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "loss": avg_loss,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--knowledge_embedding_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--top_k_text", type=int, default=5)
    parser.add_argument("--top_k_image", type=int, default=3)

    args = parser.parse_args()

    device = torch.device(args.device)

    dataset = KICPDataset(args.test_csv)

    collator = KICPCollator(
        clip_model_name=args.clip_model_name,
        max_length=77,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    print()
    print("=============== Checkpoint 加载信息 ===============")
    print(f"checkpoint: {args.checkpoint}")
    print(f"epoch: {checkpoint.get('epoch')}")
    print(f"best_f1: {checkpoint.get('best_f1')}")
    print(f"missing keys: {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")
    print("===================================================")
    print()

    model.to(device)

    result = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    labels = result["labels"]
    preds = result["preds"]
    probs = result["probs"]

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_binary = f1_score(labels, preds, average="binary", zero_division=0)

    print()
    print("=============== 测试结果 ===============")
    print(f"loss: {result['loss']:.4f}")
    print(f"accuracy: {acc:.4f}")
    print(f"f1_macro: {f1_macro:.4f}")
    print(f"f1_binary: {f1_binary:.4f}")
    print()
    print("classification_report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=["real", "fake"],
            zero_division=0,
        )
    )
    print("confusion_matrix:")
    print(confusion_matrix(labels, preds))
    print("========================================")
    print()

    output_df = pd.DataFrame(
        {
            "id": result["ids"],
            "text": result["texts"],
            "label": labels,
            "pred": preds,
            "prob_real": [p[0] for p in probs],
            "prob_fake": [p[1] for p in probs],
        }
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    metrics_path = output_csv.with_suffix(".metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": result["loss"],
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_binary": f1_binary,
                "checkpoint": args.checkpoint,
                "knowledge_embedding_path": args.knowledge_embedding_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"预测结果已保存：{output_csv}")
    print(f"指标 JSON 已保存：{metrics_path}")


if __name__ == "__main__":
    main()
