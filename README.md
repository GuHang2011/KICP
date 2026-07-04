# KICP Project

本项目是一个基于 CLIP、Prompt Learning、Co-Attention 和知识检索的多模态假新闻检测实验工程，主要面向 GossipCop 和 PolitiFact 数据集。

仓库只保留代码、配置和目录占位文件；真实数据、图片、知识向量、训练输出和本地虚拟环境不上传到 GitHub。

## 项目结构

```text
KICP_project/
  configs/                 # smoke test 配置
  datasets/                # Dataset 与 Collator
  models/                  # KICP 模型组件
  scripts/                 # 数据处理、知识库构建、训练与检查脚本
  data/                    # 本地数据目录，不上传真实数据
  knowledge/               # 本地知识库与向量目录，不上传生成文件
  train.py                 # 基础训练入口
  test.py                  # 测试入口
  train_config.py          # 配置化训练入口
  requirements.txt         # Python 依赖
```

## 环境安装

建议使用 Python 3.9 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU，请按本机 CUDA 版本安装对应的 PyTorch。

## 数据准备

本仓库不包含真实数据和图片。请自行准备数据，并放到以下目录：

```text
data/
  processed/
    gossipcop.csv
    politifact.csv
    gossipcop_full.csv
    politifact_full.csv
  images/
    gossipcop/
    politifact/
knowledge/
  wikidata_knowledge_gossipcop_embeddings.pt
  wikidata_knowledge_politifact_embeddings.pt
```

训练 CSV 至少需要包含：

```text
id,text,image_path,label
```

其中 `image_path` 是图片路径，`label` 是二分类标签。

## 快速运行

小样本 smoke test：

```bash
python train.py --csv data/processed/politifact.csv --dataset_name politifact --max_samples 20 --epochs 1 --batch_size 1 --device cpu

python train.py --csv data/processed/gossipcop.csv --dataset_name gossipcop --max_samples 20 --epochs 1 --batch_size 1 --device cpu
```

论文参数复现实验：

```bash
python scripts/train_paper_reproduction.py ^
  --csv data/processed/gossipcop_full.csv ^
  --dataset_name gossipcop_kicp_full_paper ^
  --output_dir outputs/gossipcop_kicp_full_paper ^
  --knowledge_embedding_path knowledge/wikidata_knowledge_gossipcop_embeddings.pt ^
  --max_samples 0 ^
  --epochs 20 ^
  --batch_size 1 ^
  --device cpu ^
  --lr 0.00003 ^
  --weight_decay 0.001 ^
  --dropout 0.6 ^
  --coattention_heads 8 ^
  --prompt_length 16 ^
  --top_k_text 5 ^
  --top_k_image 5
```

```bash
python scripts/train_paper_reproduction.py ^
  --csv data/processed/politifact_full.csv ^
  --dataset_name politifact_kicp_full_paper ^
  --output_dir outputs/politifact_kicp_full_paper ^
  --knowledge_embedding_path knowledge/wikidata_knowledge_politifact_embeddings.pt ^
  --max_samples 0 ^
  --epochs 20 ^
  --batch_size 1 ^
  --device cpu ^
  --lr 0.00003 ^
  --weight_decay 0.001 ^
  --dropout 0.6 ^
  --coattention_heads 8 ^
  --prompt_length 16 ^
  --top_k_text 5 ^
  --top_k_image 5
```

## GitHub 上传说明

上传前请查看 [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md)。重点是不要上传 `.venv/`、`data/` 中的真实数据、`knowledge/` 中的生成文件和 `outputs/`。
