# GitHub 上传前清理说明

本项目目录原来同时包含源码、本地虚拟环境、真实数据、图片、知识库向量和训练输出。上传 GitHub 时建议只上传可复现代码和轻量配置，不上传大文件、缓存、虚拟环境和可能涉及版权的数据。

## 建议保留并上传

- `README.md`：项目说明、安装方式和运行命令。
- `requirements.txt`：Python 依赖。
- `train.py`、`test.py`、`train_config.py`：主训练、测试和配置入口。
- `run_*.bat`、`setup.bat`：Windows 下的运行辅助脚本。
- `scripts/`：数据构建、知识库构建、训练复现实验、检查脚本。
- `models/`：KICP 模型、CLIP 编码器、Prompt Learner、知识检索等模型代码。
- `datasets/`：数据集读取和 collator 代码。
- `configs/`：小样本 smoke test 配置。
- `.gitignore`：防止大文件和本地产物误上传。
- `data/.gitkeep`、`data/raw/.gitkeep`、`data/processed/.gitkeep`、`data/images/.gitkeep`、`knowledge/.gitkeep`：只保留空目录结构，方便别人知道数据应放在哪里。

## 已清理/不建议上传

- `.venv/`：本地虚拟环境，别人应使用 `requirements.txt` 重新安装。
- `.idea/`：PyCharm/IDEA 本地配置。
- `__pycache__/`、`*.pyc`：Python 缓存文件。
- `outputs/`：训练日志、切分文件、checkpoint 等运行结果。
- `data/images/`、`data/raw/`、`data/processed/` 中的真实数据：体积较大，且可能涉及数据许可问题。
- `knowledge/*.csv`、`knowledge/*.pt`、`knowledge/cache/`：由脚本生成或缓存的知识库文件和 CLIP 向量。
- `models/*.bak`：临时备份文件，保留最终 `.py` 源码即可。

如果后续需要重新训练，请从原始数据源重新准备数据，或从你本地备份/压缩包中恢复数据。

## 上传前推荐步骤

```bash
git init
git add .
git status
```

检查 `git status` 里不应出现 `.venv/`、`data/images/`、`outputs/`、`knowledge/*.pt` 等大文件。

```bash
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```

## 数据目录格式

请自行准备 FakeNewsNet / GossipCop / PolitiFact 数据，并整理成以下路径：

```text
data/
  raw/
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

CSV 至少需要包含这些列：

```text
id,text,image_path,label
```
