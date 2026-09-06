# scripts/check_kicp_model_real_knowledge.py
# -*- coding: utf-8 -*-

"""
测试 KICP 主模型是否能够接入真实 Wikidata 知识向量。

要求：
    1. 已存在 data/processed/politifact_full.csv
    2. 已存在 knowledge/wikidata_knowledge_politifact_embeddings.pt
"""

import sys
from pathlib import Path

# 确保从 scripts 目录运行时也能导入 datasets 和 models 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel


def main():
    csv_path = PROJECT_ROOT / "data" / "processed" / "politifact_full.csv"
    knowledge_path = PROJECT_ROOT / "knowledge" / "wikidata_knowledge_politifact_embeddings.pt"

    dataset = KICPDataset(str(csv_path))
    collator = KICPCollator(
        clip_model_name="openai/clip-vit-large-patch14-336",
        max_length=77,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(dataloader))

    model = KICPModel(
        clip_model_name="openai/clip-vit-large-patch14-336",
        class_names=["real", "fake"],
        hidden_dim=768,
        prompt_length=16,
        coattention_heads=4,
        use_knowledge=True,
        top_k_text=5,
        top_k_image=3,
        knowledge_embedding_path=str(knowledge_path),
        freeze_clip=True,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            return_features=True,
        )

    print("knowledge_mode:", model.knowledge_mode)
    print("logits:", outputs["logits"].shape)
    print("probs:", outputs["probs"].shape)
    print("loss:", outputs["loss"].item())
    print("multimodal_features:", outputs["multimodal_features"].shape)
    print("knowledge_features:", outputs["knowledge_features"].shape)
    print("final_features:", outputs["final_features"].shape)
    print("text_top_indices:", outputs["text_top_indices"].shape)
    print("image_top_indices:", outputs["image_top_indices"].shape)

    retriever = model.knowledge_retriever
    text_meta = retriever.get_metadata_by_indices(outputs["text_top_indices"])
    image_meta = retriever.get_metadata_by_indices(outputs["image_top_indices"])

    print()
    print("当前样本 ID:", batch["ids"][0])
    print("当前样本文本:", batch["texts"][0][:200])

    print()
    print("text Top-5 知识：")
    for item in text_meta[0]:
        print(
            item.get("knowledge_id"),
            item.get("qid"),
            item.get("label"),
            "-",
            item.get("description"),
        )

    print()
    print("image Top-3 知识：")
    for item in image_meta[0]:
        print(
            item.get("knowledge_id"),
            item.get("qid"),
            item.get("label"),
            "-",
            item.get("description"),
        )

    print()
    print("KICP 真实知识接入测试通过。")


if __name__ == "__main__":
    main()
