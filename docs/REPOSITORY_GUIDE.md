# Repository guide

This guide maps the current KICP source tree to its inputs, entry points, and generated artifacts. It is based on source inspection; it does not report trained-model results or certify paper reproduction.

## Code map

| Path | Responsibility |
| --- | --- |
| [datasets/kicp_dataset.py](../datasets/kicp_dataset.py) | Read `id,text,image_path,label`; load RGB images; substitute white images when files are unavailable |
| [datasets/kicp_collator.py](../datasets/kicp_collator.py) | Apply CLIPProcessor; tokenize text with maximum length 77; build batches |
| [models/clip_encoder.py](../models/clip_encoder.py) | Load CLIP; extract and project sequence/pooled features |
| [models/co_attention.py](../models/co_attention.py) | Cross-modal attention and feature fusion |
| [models/prompt_learner.py](../models/prompt_learner.py) | Trainable prompt vectors and class embeddings |
| [models/knowledge_retriever.py](../models/knowledge_retriever.py) | Retrieval from a simulated trainable bank |
| [models/real_knowledge_retriever.py](../models/real_knowledge_retriever.py) | Retrieval from a local embedding tensor and metadata |
| [models/kicp_model.py](../models/kicp_model.py) | Assemble components, select knowledge mode, and compute classification outputs |
| [configs/](../configs/) | GossipCop and PolitiFact examples for train_config.py |
| [scripts/](../scripts/) | Data preparation, checks, and additional experiment entry points |
| [data/](../data/), [knowledge/](../knowledge/) | Local asset locations; generated content is ignored by Git |

The prompt implementation adds the mean prompt vector to learned class embeddings; it does not insert prompts into CLIP token embeddings. Freezing CLIP's original parameters still leaves added projection layers trainable. These distinctions matter when comparing this implementation with another method.

## Included and external assets

| Asset | Availability / requirement |
| --- | --- |
| Python source, YAML examples, dependency list | Included |
| GossipCop/PolitiFact news and images | Prepare locally; absent from the public code tree |
| CLIP processor files and weights | Loaded through Hugging Face or a compatible local directory |
| Wikidata retrieval output and search cache | Generated through network requests |
| CLIP-encoded knowledge .pt files | Generate locally; absent from the public code tree |
| Trained checkpoints, splits, logs, predictions | Generated per run; excluded by .gitignore |

The default backbone is `openai/clip-vit-large-patch14-336`. Knowledge files contain `embeddings` of shape `[N, D]`, metadata, and encoding information. `D` must equal the configured hidden dimension (768 in supplied examples). Use the same CLIP representation for queries and knowledge encoding. RealKnowledgeRetriever registers its embedding tensor with `persistent=False`, so checkpoints do not contain it: keep the original embedding file with the experiment record.

## Data and knowledge preparation

Paths are relative to the repository root. Training requires `id,text,image_path,label`, with 0 = real and 1 = fake. Knowledge construction additionally reads `title`. Missing or unreadable images become white images, so a successful forward pass does not establish image availability.

| Script | Purpose and prerequisite |
| --- | --- |
| [merge_fakenewsnet_local.py](../scripts/merge_fakenewsnet_local.py) | Merge locally acquired FakeNewsNet CSVs; uses placeholder image paths for initial checks |
| [build_full_articles.py](../scripts/build_full_articles.py) | Retrieve article text and images from news URLs; includes network-error and placeholder fallbacks |
| [build_wikidata_knowledge.py](../scripts/build_wikidata_knowledge.py) | Extract candidate entity strings, query Wikidata, and write knowledge rows plus a cache |
| [filter_wikidata_knowledge.py](../scripts/filter_wikidata_knowledge.py) | Filter retrieved knowledge before encoding |
| [encode_wikidata_knowledge_clip.py](../scripts/encode_wikidata_knowledge_clip.py) | Encode knowledge_text with CLIP and save embeddings plus metadata |

The Wikidata builder requires `--csv` and `--output`; its default `--max_rows 100` processes a subset. Set `--max_rows 0` for a full pass. Entity matching uses search results, not verified entity annotations; inspect retrieval records and acquisition failures before drawing research conclusions.

After preparing politifact_full.csv:

```sh
python scripts/build_wikidata_knowledge.py --csv data/processed/politifact_full.csv --output knowledge/wikidata_knowledge_politifact.csv --max_rows 100
```

After inspecting and filtering that output, encode the filtered CSV:

```sh
python scripts/encode_wikidata_knowledge_clip.py --input_csv knowledge/wikidata_knowledge_politifact_filtered.csv --output_pt knowledge/wikidata_knowledge_politifact_embeddings.pt --batch_size 4 --device cpu
```

Retrieval contacts Wikidata; encoding loads CLIP processor files and weights and may download them. Review each script's `--help` before a run. The merge helper has fixed dataset paths rather than a general CLI; inspect its paths first. Retain source attribution and data-specific usage conditions with locally acquired assets.

## Training and evaluation entry points

| Training | Knowledge mode / configuration | Matching evaluation |
| --- | --- | --- |
| [train.py](../train.py) | Simulated bank; 768 dimensions, 16 prompt vectors, four heads; CLI controls data/training | [test.py](../test.py) with the default architecture |
| [train_config.py](../train_config.py) | YAML-configured components; does not forward knowledge_embedding_path; supplied examples use the simulated bank | test.py only when model settings match its fixed architecture |
| [train_real_knowledge.py](../scripts/train_real_knowledge.py) | Required embedding file; 768 dimensions, 16 prompt vectors, four heads; configurable top-k | [test_real_knowledge.py](../scripts/test_real_knowledge.py) |
| [train_paper_reproduction.py](../scripts/train_paper_reproduction.py) | Required embedding file; exposes dimension, prompt length, heads, dropout, and top-k | [test_paper_reproduction.py](../scripts/test_paper_reproduction.py) with matching settings |

The historical paper_reproduction entry point defaults to 20 epochs, eight heads, dropout 0.6, prompt length 16, learning rate 3e-5, weight decay 1e-3, and top-k 5 per modality. These are code defaults; they do not establish equivalence to an independently verified paper.

For the README's default smoke run, inspect validation predictions with:

```sh
python test.py --test_csv outputs/politifact_smoke/splits/politifact_val.csv --checkpoint outputs/politifact_smoke/checkpoints/best.pt --output_csv outputs/politifact_smoke/validation_predictions.csv --batch_size 1 --device cpu
```

This reuses the validation partition used for model selection; it is not an independent test result. For final evaluation, supply an independently reserved CSV.

Real-knowledge evaluation needs the same embedding file and top-k settings used during training:

```sh
python scripts/test_real_knowledge.py --test_csv outputs/politifact_real/splits/politifact_real_val.csv --checkpoint outputs/politifact_real/checkpoints/best.pt --knowledge_embedding_path knowledge/wikidata_knowledge_politifact_embeddings.pt --output_csv outputs/politifact_real/validation_predictions.csv --batch_size 1 --device cpu --top_k_text 5 --top_k_image 3
```

Inspect missing/unexpected checkpoint-key diagnostics. Some evaluators use `strict=False`; a completed load does not demonstrate that every trained parameter matched. Align architecture arguments rather than mixing checkpoints between entry-point families.

Training creates stratified train/validation splits. Supply enough examples of each class. Positive `--max_samples` values select a subset and `0` disables the training cap; this convention does not apply directly to KICPDataset(max_samples=0), which takes the first zero rows.

## Checks and prerequisites

| Check in scripts/ | Required assets | Interpretation |
| --- | --- | --- |
| check_prompt_learner.py | Installed dependencies | Synthetic shapes and gradient diagnostics |
| check_co_attention.py | Installed dependencies | Synthetic fusion and backward-pass diagnostics |
| check_knowledge_retriever.py | Installed dependencies | Simulated-bank retrieval and gradient diagnostics |
| check_dataset.py | Prepared CSV; images for meaningful visual data | Sample inspection; no CLIP download |
| check_dataloader.py | CSV and CLIP processor files | Tokenization/batching; processor files may download |
| check_clip_encoder.py | CSV, processor files, CLIP weights | Encoder forward pass; weights may download |
| check_kicp_model.py, check_kicp_model_with_knowledge.py | CSV, processor files, CLIP weights | Model diagnostics; inspect knowledge mode in each script |
| check_real_knowledge_retriever.py | Local PolitiFact embedding file | Retrieval with synthetic query features |
| check_kicp_model_real_knowledge.py | Fixed PolitiFact full CSV, embedding file, CLIP assets | Full-model diagnostics with external knowledge |

The final two scripts use fixed PolitiFact paths. Many checks print diagnostics rather than assert every expected value. They do not replace evaluation on a defined data split.

## Historical and maintenance scripts

| File(s) | Current observation / suggested maintenance |
| --- | --- |
| scripts/build_full_articles.py and build_full_articles_fixed.py | Identical contents at this documentation review. Prefer the unsuffixed entry point; verify compatibility before consolidating. |
| scripts/patch_kicp_model_clip_arg.py | Historical source-rewriting helper with a .bak backup. It is not an installation step and should not run routinely on the current model. |
| run_train_*.bat and run_test.bat | Windows wrappers with preset paths; inspect them or use the README's explicit Python commands. |
| setup.bat | Existing Windows setup helper; the README provides an explicit isolated-environment workflow. |
| *_paper_reproduction.py | Retained experiment filenames; document actual settings and evidence for each run. |

No research code is deleted or refactored in this documentation cleanup. Dependencies use lower bounds; retain resolved versions and consider a tested lock file in a separately validated change.

## Recording an experiment

Save the revision, environment versions, data source/acquisition date, image coverage and fallback count, deduplication policy, split CSVs, seed, full arguments/configuration, CLIP identifier, knowledge-file hash, training log, and evaluation outputs. Match knowledge construction to the evaluation protocol and record which data informed it. Report repeated-run variation when available, and distinguish validation selection from independent testing.

Consult the [upload guide](../GITHUB_UPLOAD_GUIDE.md) and [ignore rules](../.gitignore) before sharing assets. This guide introduces no new license grant or ownership claim over external data, CLIP weights, or Wikidata-derived assets.
