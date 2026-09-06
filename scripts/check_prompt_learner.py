# scripts/check_prompt_learner.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 Continuous Prompt 模块是否正常。

运行：
    python scripts/check_prompt_learner.py

预期输出：
    prompt_vectors shape: torch.Size([16, 768])
    class_embeddings shape: torch.Size([2, 768])
    class_features shape: torch.Size([2, 768])
"""

import sys
from pathlib import Path

import torch

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.prompt_learner import ContinuousPromptLearner


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=============== 创建 Continuous Prompt 模块 ===============")

    class_names = ["real", "fake"]

    prompt_learner = ContinuousPromptLearner(
        class_names=class_names,
        prompt_length=16,
        prompt_dim=768,
        init_std=0.02,
        normalize=True,
    )

    print(prompt_learner)

    print()
    print("=============== 参数检查 ===============")
    print("类别名称：", class_names)
    print("prompt_vectors shape:")
    print(prompt_learner.prompt_vectors.shape)

    print("class_embeddings shape:")
    print(prompt_learner.class_embeddings.shape)

    trainable_params = count_trainable_parameters(prompt_learner)
    print(f"可训练参数量：{trainable_params:,}")

    expected_params = 16 * 768 + 2 * 768
    print(f"理论参数量：{expected_params:,}")

    print()
    print("=============== 前向传播检查 ===============")

    class_features = prompt_learner()

    print("class_features shape:")
    print(class_features.shape)

    print()
    print("class_features 前 2 行、前 5 个值：")
    print(class_features[:, :5])

    print()
    print("=============== 梯度检查 ===============")

    # 构造一个假的 loss，检查 prompt 是否能反向传播
    fake_loss = class_features.sum()
    fake_loss.backward()

    print("prompt_vectors.grad 是否存在：")
    print(prompt_learner.prompt_vectors.grad is not None)

    print("class_embeddings.grad 是否存在：")
    print(prompt_learner.class_embeddings.grad is not None)

    print()
    print("=============== Continuous Prompt 检查完成 ===============")
    print("如果你看到：")
    print("prompt_vectors shape = [16, 768]")
    print("class_embeddings shape = [2, 768]")
    print("class_features shape = [2, 768]")
    print("两个 grad 都是 True")
    print("说明本步骤成功。")


if __name__ == "__main__":
    main()