# scripts/check_co_attention.py
# -*- coding: utf-8 -*-

"""
作用：
    测试 Co-Attention 模块是否正常工作。

测试内容：
    1. 随机生成文本特征和图像特征；
    2. 输入 CoAttentionBlock；
    3. 检查输出 shape；
    4. 检查是否可以反向传播。

运行：
    python scripts/check_co_attention.py

预期输出：
    text_features shape  : torch.Size([2, 23, 768])
    image_features shape : torch.Size([2, 577, 768])
    fused_features shape : torch.Size([2, 768])
    反向传播成功
"""

import sys
from pathlib import Path

import torch

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.co_attention import CoAttentionBlock


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=============== 创建随机输入 ===============")

    batch_size = 2
    text_length = 23
    image_tokens = 577
    hidden_dim = 768

    # 模拟 CLIP 文本 token 特征
    text_features = torch.randn(
        batch_size,
        text_length,
        hidden_dim,
    )

    # 模拟 CLIP 图像 patch 特征
    image_features = torch.randn(
        batch_size,
        image_tokens,
        hidden_dim,
    )

    print("text_features shape:")
    print(text_features.shape)

    print("image_features shape:")
    print(image_features.shape)

    print()
    print("=============== 创建 CoAttentionBlock ===============")

    # Stage 1 多模态融合可以用 4 heads
    co_attention = CoAttentionBlock(
        hidden_dim=768,
        num_heads=4,
        dropout=0.1,
        ffn_dim=2048,
    )

    print(co_attention)

    trainable_params = count_trainable_parameters(co_attention)

    print()
    print("可训练参数量：")
    print(f"{trainable_params:,}")

    print()
    print("=============== 前向传播 ===============")

    # KICP 中一般写作：
    # H_M = CoAttention(image_features, text_features)
    fused_features = co_attention(
        a_features=image_features,
        b_features=text_features,
    )

    print("fused_features shape:")
    print(fused_features.shape)

    print()
    print("=============== 中间结果检查 ===============")

    details = co_attention(
        a_features=image_features,
        b_features=text_features,
        return_details=True,
    )

    print("details keys:")
    print(details.keys())

    print("h_ab shape:")
    print(details["h_ab"].shape)

    print("h_ba shape:")
    print(details["h_ba"].shape)

    print("h_ab_pooled shape:")
    print(details["h_ab_pooled"].shape)

    print("h_ba_pooled shape:")
    print(details["h_ba_pooled"].shape)

    print()
    print("=============== 反向传播检查 ===============")

    fake_loss = fused_features.sum()
    fake_loss.backward()

    has_grad = False
    for name, param in co_attention.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            print(f"参数有梯度：{name}, grad shape={param.grad.shape}")
            break

    print()
    print("是否至少有一个参数成功获得梯度：")
    print(has_grad)

    print()
    print("=============== Co-Attention 检查完成 ===============")
    print("如果你看到：")
    print("text_features shape  = [2, 23, 768]")
    print("image_features shape = [2, 577, 768]")
    print("fused_features shape = [2, 768]")
    print("是否至少有一个参数成功获得梯度 = True")
    print("说明本步骤成功。")


if __name__ == "__main__":
    main()