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
import sys
import shutil
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

sys.path.insert(0, str(Path("/mnt/shared-fs/lindenbauer/RouteLLM/reproduce")))
from utils import run_bert_classifier_training

# %%
cache_dir = "/s3/lindenbauer/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

project_root = Path("/mnt/shared-fs/lindenbauer/RouteLLM")
script_path = project_root / "reproduce" / "train_roberta.py"
models_root = Path("/s3/lindenbauer/RouteLLM/models")
# Keep label metadata consistent with D_arena mappings
LABEL_NAMES = {0: "strong_wins", 1: "tie", 2: "weak_wins"}
STRONG_MODEL_IDS = {"gpt-4-1106-preview"}
BASE_MODEL_NAME = "FacebookAI/xlm-roberta-base"
PRETRAINED_STAGE1_CHECKPOINT = Path(
    "/s3/lindenbauer/RouteLLM/models/d_arena-cls-balanced_data-bs128/checkpoint-1900/"
)

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

# %% [markdown]
# ## Process MMLU Battles

# %%
mmlu_dataset = load_dataset("routellm/mmlu_battles")


# %%
def map_mmlu_battle(example):
    """Convert tie/win signal into the 3-way preference label plus prompt cleanup."""
    example["prompt"] = example["prompt"].strip()

    if example["winner_tie"] == 1:
        example["labels"] = 1
        return example

    if example["winner_model_a"] == 1:
        winning_model = example["model_a"]
    else:
        winning_model = example["model_b"]

    example["labels"] = 0 if winning_model in STRONG_MODEL_IDS else 2
    return example


train_split = mmlu_dataset["train"]
processed_dataset = train_split.map(
    map_mmlu_battle,
    desc="Mapping MMLU outcomes to labels"
)

print(f"Processed dataset size: {len(processed_dataset)}")
label_counts = Counter(processed_dataset["labels"])
print("\nLabel distribution:")
for label in sorted(label_counts.keys()):
    count = label_counts[label]
    pct = 100 * count / len(processed_dataset)
    print(f"  {LABEL_NAMES[label]}: {count} ({pct:.1f}%)")

sample = processed_dataset[0]
print("\nSample processed example:")
print(f"  prompt: {sample['prompt'][:80]}...")
print(f"  labels: {sample['labels']} ({LABEL_NAMES[sample['labels']]})")

# %% [markdown]
# ## Remove unused columns

# %%
processed_dataset = processed_dataset.select_columns(["prompt", "labels"])

print(f"\nFinal columns: {processed_dataset.column_names}")

# %% [markdown]
# ## Create train/validation split

# %%
from datasets import ClassLabel

# First, cast labels to ClassLabel type (required for stratified splitting)
processed_dataset = processed_dataset.cast_column(
    "labels",
    ClassLabel(names=["strong_wins", "tie", "weak_wins"])
)

split_dataset = processed_dataset.train_test_split(
    test_size=0.1,
    stratify_by_column="labels",
    seed=42
)

split_dataset = DatasetDict(
    train=split_dataset["train"],
    validation=split_dataset["test"],
)

print(f"\nSplit sizes:")
print(f"  train: {len(split_dataset['train'])}")
print(f"  validation: {len(split_dataset['validation'])}")

for split_name in ("train", "validation"):
    split_counts = Counter(split_dataset[split_name]["labels"])
    print(f"\n{split_name.title()} label distribution:")
    for label in sorted(split_counts.keys()):
        count = split_counts[label]
        pct = 100 * count / len(split_dataset[split_name])
        print(f"  {LABEL_NAMES[label]}: {count} ({pct:.1f}%)")

mmlu_dataset_path = os.path.join(
    os.environ["HF_DATASETS_CACHE"],
    "mmlu_battles_stage2"
)

if os.path.exists(mmlu_dataset_path):
    shutil.rmtree(mmlu_dataset_path)

split_dataset.save_to_disk(mmlu_dataset_path)

# %% [markdown]
# ## Train classifier
# We'll start by training on the data as it is.

# %%
mmlu_dataset_path = os.path.join(
    os.environ["HF_DATASETS_CACHE"],
    "mmlu_battles_stage2"
)
mmlu_dataset = load_from_disk(mmlu_dataset_path)

# %%
run_bert_classifier_training(
    dataset_path=mmlu_dataset_path,
    model_name_or_path=PRETRAINED_STAGE1_CHECKPOINT,
    tokenizer_name_or_path=BASE_MODEL_NAME,
    train_indices=None,
    output_subdir="d_mmlu_cls-raw",
    max_steps=200,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=20,
    save_steps=100,
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_mmlu_cls-raw",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2, 
    freeze_encoder=True,
)

# %% [markdown]
# This again results in predicting the majority class. Let's check the authors checkpoint again.

# %%
from typing import Optional

def evaluate_checkpoint(
    model_path: str,
    dataset_path: str = mmlu_dataset_path,
    num_labels: int = 3,
    max_length: int = 512,
    per_device_eval_batch_size: int = 128,
    cuda_device: Optional[int] = 0,
) -> dict:
    """
    Evaluate a HuggingFace model checkpoint on the D_arena validation set.

    Args:
        model_path: Path to the HuggingFace model checkpoint
        dataset_path: Path to the dataset on disk
        num_labels: Number of labels for classification (default: 3 for strong_wins/tie/weak_wins)
        max_length: Maximum sequence length for tokenization
        per_device_eval_batch_size: Batch size for evaluation
        cuda_device: CUDA device index (None to use default)

    Returns:
        Dictionary containing evaluation metrics
    """
    import sys
    import torch
    from pathlib import Path
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_from_disk

    # Import compute_metrics from train_roberta.py
    sys.path.insert(0, str(Path("/mnt/shared-fs/lindenbauer/RouteLLM/reproduce")))
    from train_roberta import compute_metrics

    # Set CUDA device if specified
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # Load the model checkpoint
    print(f"Loading {model_path} checkpoint...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded")

    # Load our D_arena validation set
    input_dataset = load_from_disk(dataset_path)
    eval_dataset = input_dataset["validation"]

    print(f"\nEvaluation dataset size: {len(eval_dataset)}")

    # Tokenize the validation set
    print("\nTokenizing validation set...")
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=max_length)

    eval_dataset_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing validation set"
    )

    # Create a Trainer for evaluation only (no training)
    training_args = TrainingArguments(
        output_dir=f"/tmp/eval_{model_path.replace('/', '_')}",  # Temporary output dir
        per_device_eval_batch_size=per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = trainer.evaluate(eval_dataset=eval_dataset_tokenized)

    # Print results
    print("\n" + "=" * 80)
    print(f"Checkpoint Evaluation Results: {model_path}")
    print("=" * 80)

    label_names = {0: "strong_wins", 1: "tie", 2: "weak_wins"}

    for key, value in sorted(metrics.items()):
        # Format the output nicely
        if key.startswith("eval_accuracy_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) accuracy: {value:.4f}")
        elif key.startswith("eval_precision_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) precision: {value:.4f}")
        elif key.startswith("eval_recall_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) recall: {value:.4f}")
        elif key.startswith("eval_f1_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) F1: {value:.4f}")
        elif key.startswith("eval_support_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) support: {value}")
        elif key.startswith("eval_pred_count_class_"):
            class_idx = int(key.split("_")[-1])
            print(f"Class {class_idx} ({label_names[class_idx]}) predicted count: {value}")
        elif "macro" in key:
            print(f"{key}: {value:.4f}")
        elif key == "eval_accuracy":
            print(f"\nOverall accuracy: {value:.4f}\n")
        else:
            print(f"{key}: {value}")

    print("=" * 80)

    return metrics


# %%
evaluate_checkpoint('routellm/bert_mmlu_augmented')

# %% [markdown]
# ## Balance the dataset
# We apply the same oversampling procedure as we did for D_arena to generate a balanced dataset.

# %%
mmlu_dataset

# %%
from sklearn.utils import resample
import numpy as np

mmlu_train = mmlu_dataset['train']
mmlu_val = mmlu_dataset['validation']

label_names = {0: "strong_wins", 1: "tie", 2: "weak_wins"}

# Get labels and indices
labels = np.array(mmlu_train["labels"])
unique_labels = np.unique(labels)

# Print original distribution
print("=" * 80)
print("Oversampling Training Set")
print("=" * 80)
print(f"Original class distribution:")
class_counts = {}
for label in unique_labels:
    count = (labels == label).sum()
    class_counts[label] = count
    print(f"  Class {label} ({label_names[label]}): {count} samples")

# Find majority class size
max_count = max(class_counts.values())

# Resample each class to match majority
balanced_indices = []
for label in unique_labels:
    # Get indices for this class
    class_indices = np.where(labels == label)[0].tolist()
    
    # Oversample to match majority class
    if len(class_indices) < max_count:
        resampled_indices = resample(
            class_indices,
            n_samples=max_count,
            replace=True,
            random_state=42
        )
    else:
        resampled_indices = class_indices
    
    balanced_indices.extend(resampled_indices)

# Shuffle and create balanced dataset
np.random.seed(42)
np.random.shuffle(balanced_indices)
mmlu_train_balanced = mmlu_train.select(balanced_indices)

# Print balanced distribution
print(f"\nBalanced class distribution:")
balanced_labels = np.array(mmlu_train_balanced.select(balanced_indices)["labels"])
for label in unique_labels:
    count = (balanced_labels == label).sum()
    pct = 100 * count / len(mmlu_train_balanced.select(balanced_indices))
    print(f"  Class {label} ({label_names[label]}): {count} samples ({pct:.1f}%)")

print(f"\nTraining set size: {len(mmlu_train)} → {len(mmlu_train_balanced)}")
print(f"Validation set size: {len(mmlu_val)} (unchanged)")


# %%
from datasets import DatasetDict

cache_dir = "/s3/lindenbauer/.cache/datasets"
mmlu_balanced_dataset_path = os.path.join(cache_dir, "mmlu_balanced")
os.makedirs(cache_dir, exist_ok=True)

# %%
# Save as DatasetDict with BALANCED train split and original validation split
mmlu_dataset_dict = DatasetDict({
    "train": mmlu_train_balanced,
    "validation": mmlu_val
})

mmlu_dataset_dict.save_to_disk(mmlu_balanced_dataset_path)

# %% [markdown]
# ## Train on fully balanced dataset

# %%
run_bert_classifier_training(
    dataset_path=mmlu_balanced_dataset_path,
    model_name_or_path=PRETRAINED_STAGE1_CHECKPOINT,
    tokenizer_name_or_path=BASE_MODEL_NAME,
    train_indices=None,
    output_subdir="d_mmlu_cls-balanced",
    max_steps=750,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=20,
    save_steps=100,
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_mmlu_cls-balanced",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    freeze_encoder=True,
)

# %% [markdown]
# ## Combine d_arena and mmlu_balanced datasets

# %%
from datasets import concatenate_datasets

# Load the preprocessed balanced datasets
cache_dir = "/s3/lindenbauer/.cache/datasets"
d_arena_path = os.path.join(cache_dir, "d_arena")
mmlu_balanced_path = os.path.join(cache_dir, "mmlu_balanced")

d_arena_dataset = load_from_disk(d_arena_path)
mmlu_balanced_dataset = load_from_disk(mmlu_balanced_path)

print("D Arena dataset:")
print(f"  Train size: {len(d_arena_dataset['train'])}")
print(f"  Validation size: {len(d_arena_dataset['validation'])}")

print("\nMMLU balanced dataset:")
print(f"  Train size: {len(mmlu_balanced_dataset['train'])}")
print(f"  Validation size: {len(mmlu_balanced_dataset['validation'])}")

# Combine train sets
combined_train = concatenate_datasets([
    d_arena_dataset["train"],
    mmlu_balanced_dataset["train"]
])

# Combine validation sets
combined_validation = concatenate_datasets([
    d_arena_dataset["validation"],
    mmlu_balanced_dataset["validation"]
])

print(f"\nCombined dataset:")
print(f"  Train size: {len(combined_train)}")
print(f"  Validation size: {len(combined_validation)}")

# Create combined DatasetDict
combined_dataset = DatasetDict({
    "train": combined_train,
    "validation": combined_validation
})

# Save the combined dataset
combined_dataset_path = os.path.join(cache_dir, "d_arena_mmlu_combined-balanced")
if os.path.exists(combined_dataset_path):
    shutil.rmtree(combined_dataset_path)

combined_dataset.save_to_disk(combined_dataset_path)
print(f"\nCombined dataset saved to: {combined_dataset_path}")


# %% [markdown]
# ## End-to-end train on fully balanced D_arena + D_gold (=D_mmlu)

# %%
run_bert_classifier_training(
    dataset_path=combined_dataset_path,
    model_name_or_path="FacebookAI/xlm-roberta-base",
    train_indices=None,
    output_subdir="d_arena_mmlu-cls-balanced",
    max_steps=4000,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_arena_mmlu-cls-balanced",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    freeze_encoder=True,
)

# %% [markdown]
# ## Combine skewed datasets, then oversample to balance
# This should avoid oversampling smaller classes from especially MMLU too aggressively.

# %%
cache_dir = "/s3/lindenbauer/.cache/datasets"
d_arena_path_raw = os.path.join(cache_dir, "d_arena-raw")
d_mmlu_path_raw 

# %%
