# scripts/check_real_knowledge_retriever.py
# -*- coding: utf-8 -*-

"""
测试真实 Wikidata 知识检索模块。
这个脚本只测试 models/real_knowledge_retriever.py，不需要读取数据集。
"""

import sys
from pathlib import Path

# 确保从 scripts 目录运行时也能导入 models 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.real_knowledge_retriever import RealKnowledgeRetriever


def main():
    knowledge_path = PROJECT_ROOT / "knowledge" / "wikidata_knowledge_politifact_embeddings.pt"

    retriever = RealKnowledgeRetriever(
        knowledge_embedding_path=str(knowledge_path),
        hidden_dim=768,
        text_top_k=5,
        image_top_k=3,
        temperature=0.07,
        normalize=True,
    )

    text_query = torch.randn(2, 768)
    image_query = torch.randn(2, 768)

    outputs = retriever(
        text_query_features=text_query,
        image_query_features=image_query,
    )

    print("knowledge_embeddings:", retriever.knowledge_embeddings.shape)
    print("text_knowledge_features:", outputs["text_knowledge_features"].shape)
    print("image_knowledge_features:", outputs["image_knowledge_features"].shape)
    print("knowledge_features:", outputs["knowledge_features"].shape)
    print("text_top_indices:", outputs["text_top_indices"].shape)
    print("image_top_indices:", outputs["image_top_indices"].shape)

    text_meta = retriever.get_metadata_by_indices(outputs["text_top_indices"])
    image_meta = retriever.get_metadata_by_indices(outputs["image_top_indices"])

    print()
    print("第 1 个样本 text Top-5 知识：")
    for item in text_meta[0]:
        print(
            item.get("knowledge_id"),
            item.get("qid"),
            item.get("label"),
            "-",
            item.get("description"),
        )

    print()
    print("第 1 个样本 image Top-3 知识：")
    for item in image_meta[0]:
        print(
            item.get("knowledge_id"),
            item.get("qid"),
            item.get("label"),
            "-",
            item.get("description"),
        )

    print()
    print("真实知识检索模块测试通过。")


if __name__ == "__main__":
    main()
