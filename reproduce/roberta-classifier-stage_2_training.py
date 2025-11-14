# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup

# %%
import os
from pathlib import Path

from datasets import load_dataset, load_from_disk

sys.path.insert(0, str(Path("/mnt/shared-fs/lindenbauer/RouteLLM/reproduce")))
from utils import run_bert_classifier_training, format_indices

# %%
cache_dir = "/s3/lindenbauer/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

project_root = Path("/mnt/shared-fs/lindenbauer/RouteLLM")
script_path = project_root / "reproduce" / "train_roberta.py"
models_root = Path("/s3/lindenbauer/RouteLLM/models")
models_root.mkdir(parents=True, exist_ok=True)

# %%
from datasets import load_from_disk, DatasetDict
cache_dir = "/s3/lindenbauer/.cache/datasets"
d_arena_path = os.path.join(cache_dir, "d_arena")

d_arena = load_from_disk(d_arena_path)

# %%
d_arena

# %% [markdown]
# # MMLU trained model checkpoint

# %%
mmlu_dataset = load_dataset("routellm/mmlu_battles")

# %%
mmlu_dataset

# %% [markdown]
# ## Map dataset labels

# %%
run_bert_classifier_training(
    train_indices=None,
    output_subdir="d_mmlu_cls",
    max_steps=2500,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_mmlu_cls",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2, 
    freeze_encoder=True,
)
