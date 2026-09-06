# scripts/check_knowledge_retriever.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 KnowledgeRetriever 是否正常工作。

运行：
    python scripts/check_knowledge_retriever.py

预期输出：
    text_knowledge_features shape  = [2, 768]
    image_knowledge_features shape = [2, 768]
    knowledge_features shape       = [2, 768]
    text_topk_indices shape        = [2, 5]
    image_topk_indices shape       = [2, 5]
    反向传播成功
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.knowledge_retriever import KnowledgeRetriever


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=============== 创建随机输入 ===============")

    batch_size = 2
    hidden_dim = 768

    text_pooled_features = torch.randn(
        batch_size,
        hidden_dim,
        requires_grad=True,
    )

    image_pooled_features = torch.randn(
        batch_size,
        hidden_dim,
        requires_grad=True,
    )

    print("text_pooled_features shape:")
    print(text_pooled_features.shape)

    print("image_pooled_features shape:")
    print(image_pooled_features.shape)

    print()
    print("=============== 创建 KnowledgeRetriever ===============")

    retriever = KnowledgeRetriever(
        hidden_dim=768,
        knowledge_bank_size=64,
        top_k_text=5,
        top_k_image=5,
        init_std=0.02,
        temperature=0.07,
        normalize=True,
    )

    print(retriever)

    print()
    print("可训练参数量：")
    print(f"{count_trainable_parameters(retriever):,}")

    print()
    print("=============== 前向传播 ===============")

    outputs = retriever(
        text_pooled_features=text_pooled_features,
        image_pooled_features=image_pooled_features,
    )

    print("outputs keys:")
    print(outputs.keys())

    print()
    print("text_knowledge_features shape:")
    print(outputs["text_knowledge_features"].shape)

    print()
    print("image_knowledge_features shape:")
    print(outputs["image_knowledge_features"].shape)

    print()
    print("knowledge_features shape:")
    print(outputs["knowledge_features"].shape)

    print()
    print("text_topk_indices shape:")
    print(outputs["text_topk_indices"].shape)

    print("text_topk_indices:")
    print(outputs["text_topk_indices"])

    print()
    print("image_topk_indices shape:")
    print(outputs["image_topk_indices"].shape)

    print("image_topk_indices:")
    print(outputs["image_topk_indices"])

    print()
    print("text_topk_weights:")
    print(outputs["text_topk_weights"])

    print()
    print("image_topk_weights:")
    print(outputs["image_topk_weights"])

    print()
    print("=============== 反向传播检查 ===============")

    fake_loss = outputs["knowledge_features"].sum()
    fake_loss.backward()

    print("knowledge_bank.grad 是否存在：")
    print(retriever.knowledge_bank.grad is not None)

    print("text_pooled_features.grad 是否存在：")
    print(text_pooled_features.grad is not None)

    print("image_pooled_features.grad 是否存在：")
    print(image_pooled_features.grad is not None)

    print()
    print("=============== Knowledge Retriever 检查完成 ===============")
    print("如果你看到：")
    print("text_knowledge_features shape  = [2, 768]")
    print("image_knowledge_features shape = [2, 768]")
    print("knowledge_features shape       = [2, 768]")
    print("text_topk_indices shape        = [2, 5]")
    print("image_topk_indices shape       = [2, 5]")
    print("三个 grad 检查都为 True")
    print("说明本步骤成功。")


if __name__ == "__main__":
    main()