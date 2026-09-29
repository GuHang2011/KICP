# KICP · Multimodal Fake News Detection

A research implementation for exploring CLIP features, co-attention, continuous prompt vectors, and knowledge retrieval on GossipCop and PolitiFact data.

The repository includes model components, data preparation scripts, training/evaluation entry points, and component checks. Datasets, CLIP weights, generated knowledge embeddings, and trained checkpoints are external assets. These instructions describe the current code; they do not establish reproduction of published results.

**中文概述：** 面向多模态假新闻检测的实验工程，包含 CLIP 特征提取、协同注意力、连续提示向量和知识检索模块。仓库提供代码与配置，真实数据、模型权重和实验产物需单独准备。

## Start here

| Goal | Entry point |
| --- | --- |
| Understand the model | [Model assembly](models/kicp_model.py) and [repository guide](docs/REPOSITORY_GUIDE.md) |
| Inspect the data interface | [Dataset](datasets/kicp_dataset.py) |
| Check components without model downloads | [Minimal checks](#minimal-checks) |
| Run a small training job | [train.py](train.py) or [train_config.py](train_config.py) |
| Train with external knowledge | [train_real_knowledge.py](scripts/train_real_knowledge.py) |
| Prepare files for sharing | [Upload guide](GITHUB_UPLOAD_GUIDE.md) |

## Implementation

- **CLIP encoder:** extracts text token/image patch sequences and pooled features. The default is `openai/clip-vit-large-patch14-336`; the supplied training entry points freeze CLIP parameters.
- **Co-attention:** combines text and image sequences into a shared representation.
- **Prompt learner:** averages trainable prompt vectors and adds that context to trainable class embeddings for classification. It does not inject prompts into the CLIP text encoder.
- **Knowledge retrieval:** supports a simulated trainable bank or a local file of CLIP-encoded Wikidata text embeddings. With knowledge enabled, omitting the embedding file selects the simulated bank.

See the [code map and experiment notes](docs/REPOSITORY_GUIDE.md) for entry-point differences and historical scripts.

## Installation

Use Python 3.9 or newer in an isolated environment. [requirements.txt](requirements.txt) specifies minimum versions, not a locked or exhaustively tested environment. For GPU use, install a matching PyTorch/torchvision build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the remaining requirements.

```sh
git clone https://github.com/GuHang2011/KICP.git
cd KICP
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, `.venv\Scripts\activate.bat` in Windows Command Prompt, or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -r requirements.txt
```

Run commands from the repository root. CLIP-dependent commands use Hugging Face `from_pretrained` and may download processor files and weights on first use. Where available, `--clip_model_name` also accepts a compatible local model directory. A one-epoch CPU run still loads CLIP-Large and is not a lightweight installation test.

## Minimal checks

These scripts use synthetic tensors and installed dependencies, with no dataset, CLIP download, or knowledge file:

```sh
python scripts/check_prompt_learner.py
python scripts/check_co_attention.py
python scripts/check_knowledge_retriever.py
```

They print shapes and gradient diagnostics. Inspect those values: these are component inspection scripts, not a full correctness or benchmark suite.

After preparing a CSV, inspect samples without loading CLIP:

```sh
python scripts/check_dataset.py --csv data/processed/politifact.csv --max_samples 5
```

The following additionally needs CLIP processor files and weights, downloaded or cached:

```sh
python scripts/check_clip_encoder.py --csv data/processed/politifact.csv --max_samples 2 --batch_size 1 --device cpu
```

## Data and external assets

Prepare data locally in this layout:

```text
data/
  processed/
    gossipcop.csv
    politifact.csv
    gossipcop_full.csv       # optional prepared article data
    politifact_full.csv
  images/
    gossipcop/
    politifact/
knowledge/
  wikidata_knowledge_gossipcop_embeddings.pt
  wikidata_knowledge_politifact_embeddings.pt
```

Training CSVs require `id,text,image_path,label`, with `0 = real` and `1 = fake`. Relative image paths resolve from the working directory. Knowledge construction additionally expects `title`; see the [preparation workflow](docs/REPOSITORY_GUIDE.md#data-and-knowledge-preparation).

The loader substitutes a white image for missing or unreadable files. Check image coverage separately before interpreting an experiment as using real images. The local merge helper deliberately uses placeholder images for initial pipeline checks.

Raw news, images, external model weights, generated knowledge files, and training outputs are not distributed here. Obtain data from its original sources and follow their access and use conditions. Knowledge embeddings must match the selected CLIP representation and model dimension; retain the embedding file when evaluating a real-knowledge checkpoint.

## Small training runs

With prepared data and CLIP assets available, these commands use a **simulated trainable knowledge bank**. Each CSV needs enough examples of both classes for the stratified train/validation split.

```sh
python train.py --csv data/processed/politifact.csv --dataset_name politifact --output_dir outputs/politifact_smoke --max_samples 20 --epochs 1 --batch_size 1 --device cpu
python train.py --csv data/processed/gossipcop.csv --dataset_name gossipcop --output_dir outputs/gossipcop_smoke --max_samples 20 --epochs 1 --batch_size 1 --device cpu
```

Alternatively, the supplied YAML examples select up to 100 samples and three epochs on CPU:

```sh
python train_config.py --config configs/politifact_smoke.yaml
python train_config.py --config configs/gossipcop_smoke.yaml
```

For **external Wikidata knowledge embeddings**, use the dedicated entry point:

```sh
python scripts/train_real_knowledge.py --csv data/processed/politifact_full.csv --dataset_name politifact_real --output_dir outputs/politifact_real --knowledge_embedding_path knowledge/wikidata_knowledge_politifact_embeddings.pt --max_samples 20 --epochs 1 --batch_size 1 --device cpu
```

Training writes split CSVs and checkpoints beneath the selected output directory. Real-knowledge training also writes `logs/train_log.csv`. Use separate output directories to retain each run. Small-sample runs check the workflow; they do not establish generalization performance.

## Evaluation and experiment records

Pair [test.py](test.py) with the default `train.py` architecture. Pair [test_real_knowledge.py](scripts/test_real_knowledge.py) with real-knowledge training and its original embedding file. Match architecture settings at evaluation time: the scripts do not automatically reconstruct every setting from a checkpoint.

The existing `*_paper_reproduction.py` filenames are retained as experiment entry points. Their names do not certify equivalence to a paper. The [repository guide](docs/REPOSITORY_GUIDE.md#training-and-evaluation-entry-points) documents their configurable architecture and matching evaluator.

For reportable results, retain the exact revision, resolved dependencies, model identifier, data provenance, image coverage, split CSVs, seed, full configuration, and evaluation outputs. Reserve an independent test set; training creates train/validation splits, not a separate benchmark test set.
