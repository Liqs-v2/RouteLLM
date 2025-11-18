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

# %%
import os
import random
import tempfile
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional

from datasets import load_dataset, load_from_disk
sys.path.insert(0, str(Path("/mnt/shared-fs/lindenbauer/RouteLLM/reproduce")))
from utils import run_bert_classifier_training, format_indices

# %% [markdown]
# ## Configure Cache

# %%
cache_dir = "/s3/lindenbauer/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# %% [markdown]
# ## Data

# %%
DATASET_NAME = "routellm/mmlu_battles"


# %%
dataset = load_dataset("routellm/mmlu_battles")


# %% [markdown]
# # Overfit experiments

# %%
def build_overfit_indices(dataset, seed: int = 1027) -> dict[str, list[int]]:
    if len(dataset) < 16 * 4:
        raise ValueError("Dataset too small to build four batch indices (needs at least 64 samples).")

    rng = random.Random(seed)
    ordered = list(range(len(dataset)))
    rng.shuffle(ordered)

    indices = {
        "single_sample": [ordered[0]],
        "small_subset": sorted(ordered[:4]),
        "four_batches": sorted(ordered[: 16 * 4]),
    }
    return indices


overfit_indices = build_overfit_indices(dataset['train'])
overfit_indices


# %%
project_root = Path("/mnt/shared-fs/lindenbauer/RouteLLM")
script_path = project_root / "reproduce" / "train_roberta.py"
models_root = Path("/s3/lindenbauer/RouteLLM/models")
models_root.mkdir(parents=True, exist_ok=True)

def run_overfit_training(
    scenario_key: str,
    *,
    output_subdir: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    max_steps: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    eval_strategy: str = "steps",
    report_to: Optional[str] = "wandb",
    cuda_device: Optional[int] = 0,
) -> None:
    indices = overfit_indices[scenario_key]
    index_arg = format_indices(indices)
    output_dir = models_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script_path),
        "--dataset-name",
        DATASET_NAME,
        "--train-split",
        "train",
        "--train-indices",
        index_arg,
        "--eval-indices",
        index_arg,
        "--output-dir",
        str(output_dir),
        "--model-name",
        "FacebookAI/xlm-roberta-base",
        "--eval-strategy",
        eval_strategy,
        "--per-device-train-batch-size",
        str(per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(per_device_eval_batch_size),
        "--eval-steps",
        str(eval_steps),
        "--max-steps",
        str(max_steps),
        "--save-steps",
        str(save_steps),
        "--logging-steps",
        str(logging_steps),
    ]

    if gradient_accumulation_steps != 1:
        command.extend(
            ["--gradient-accumulation-steps", str(gradient_accumulation_steps)]
        )
    if learning_rate is not None:
        command.extend(["--learning-rate", str(learning_rate)])
    if weight_decay is not None:
        command.extend(["--weight-decay", str(weight_decay)])
    if report_to is not None:
        command.extend(["--report-to", report_to])
    if cuda_device is not None:
        command.extend(["--cuda-device", str(cuda_device)])

    wandb_run_name = f"overfit_{scenario_key}"
    command.extend(["--wandb-run-name", wandb_run_name])

    print("Launching training:", " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    print(f"Training script finished with return code {result.returncode}")

    if result.stdout:
        print("Training stdout:")
        print(result.stdout)

    if result.stderr:
        print("Training stderr:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=command,
            output=result.stdout,
            stderr=result.stderr,
        )

    return result


# %% [markdown]
# ## Overfit to a Single Sample


# %%
run_overfit_training(
    scenario_key="single_sample",
    output_subdir="overfit_single-retry-after-refactor",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=10,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
)


# %% [markdown]
# ## Overfit to a Small Subset


# %%
run_overfit_training(
    scenario_key="small_subset",
    output_subdir="overfit_small_subset",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=200,
    save_steps=200,
    eval_steps=10,
    logging_steps=5,
    learning_rate=1e-5,
    weight_decay=0.01,
)


# %% [markdown]
# ## Overfit to Four Batches


# %%
run_overfit_training(
    scenario_key="four_batches",
    output_subdir="overfit_four_batches",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=400,
    save_steps=400,
    eval_steps=25,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
)

# %% [markdown]
# # D_Arena Dataset Processing
#
# Based on the RouteLLM paper, we create a preference dataset by:
# 1. Clustering models from the Chatbot Arena into 10 tiers based on leaderboard scores
# 2. Defining Tier 0-1 models as "strong" and Tier 2-9 as "weak"
# 3. Keeping ALL battles where both models are in our tier mapping (64 models)
# 4. Filtering by prompt length (>= 16 characters)
# 5. Labeling outcomes based on winner's tier: (strong_wins=0, tie=1, weak_wins=2)
#
# Note: This approach keeps ~87% of the original dataset (matching paper's 65k from 75k after validation holdout)
# The tier assignment is used for LABELING, not for filtering battles.

# %% [markdown]
# ## 1. Define Model Tiers

# %%
# Model tier assignments from the RouteLLM paper (Appendix A)
MODEL_TIERS = {
    # Tier 0
    "gpt-4-0125-preview": 0,
    "gpt-4-1106-preview": 0,
    # Tier 1
    "gpt-4-0314": 1,
    "gpt-4-0613": 1,
    "mistral-medium": 1,
    "claude-1": 1,
    "qwen1.5-72b-chat": 1,
    # Tier 2
    "claude-2.0": 2,
    "mixtral-8x7b-instruct-v0.1": 2,
    "claude-2.1": 2,
    "gemini-pro-dev-api": 2,
    "gpt-3.5-turbo-0314": 2,
    "gpt-3.5-turbo-0613": 2,
    "gemini-pro": 2,
    "gpt-3.5-turbo-0125": 2,
    "claude-instant-1": 2,
    "yi-34b-chat": 2,
    "starling-lm-7b-alpha": 2,
    "wizardlm-70b": 2,
    "vicuna-33b": 2,
    "tulu-2-dpo-70b": 2,
    "nous-hermes-2-mixtral-8x7b-dpo": 2,
    "llama-2-70b-chat": 2,
    "openchat-3.5": 2,
    # Tier 3
    "llama2-70b-steerlm-chat": 3,
    "pplx-70b-online": 3,
    "dolphin-2.2.1-mistral-7b": 3,
    "gpt-3.5-turbo-1106": 3,
    "deepseek-llm-67b-chat": 3,
    "openhermes-2.5-mistral-7b": 3,
    "openchat-3.5-0106": 3,
    "wizardlm-13b": 3,
    "mistral-7b-instruct-v0.2": 3,
    "solar-10.7b-instruct-v1.0": 3,
    "zephyr-7b-beta": 3,
    "zephyr-7b-alpha": 3,
    "codellama-34b-instruct": 3,
    "mpt-30b-chat": 3,
    "llama-2-13b-chat": 3,
    "vicuna-13b": 3,
    "qwen1.5-7b-chat": 3,
    "pplx-7b-online": 3,
    "falcon-180b-chat": 3,
    "llama-2-7b-chat": 3,
    "guanaco-33b": 3,
    "qwen-14b-chat": 3,
    # Tier 4
    "stripedhyena-nous-7b": 4,
    "mistral-7b-instruct": 4,
    "vicuna-7b": 4,
    "qwen1.5-4b-chat": 4,
    "palm-2": 4,
    # Tier 5
    "koala-13b": 5,
    "chatglm3-6b": 5,
    "gpt4all-13b-snoozy": 5,
    # Tier 6
    "mpt-7b-chat": 6,
    "RWKV-4-Raven-14B": 6,
    "chatglm2-6b": 6,
    "alpaca-13b": 6,
    "oasst-pythia-12b": 6,
    # Tier 7
    "fastchat-t5-3b": 7,
    "chatglm-6b": 7,
    # Tier 8
    "dolly-v2-12b": 8,
    "stablelm-tuned-alpha-7b": 8,
    # Tier 9
    "llama-13b": 9,
}

# Define strong and weak model sets
M_STRONG = set(model for model, tier in MODEL_TIERS.items() if tier in [0, 1])
M_WEAK = set(model for model, tier in MODEL_TIERS.items() if tier >= 2)

print(f"Strong models (Tier 0-1): {len(M_STRONG)} models")
print(f"Weak models (Tier 2-9): {len(M_WEAK)} models")
print(f"Total mapped models: {len(MODEL_TIERS)}")

# %% [markdown]
# ## 2. Load Raw Dataset

# %%
import json
from collections import Counter

# Use the 55k dataset as specified in config.example.yaml
# This matches what the paper used (though they mention 80k, which may have been an internal version)
raw_dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
print(f"Raw dataset size: {len(raw_dataset)}")
print(f"Columns: {raw_dataset.column_names}")
print("\nSample row:")
sample = raw_dataset[0]
for key in raw_dataset.column_names:
    if key == "prompt":
        print(f"  {key}: {sample[key][:100]}...")
    else:
        print(f"  {key}: {sample[key]}")


# %% [markdown]
# ## 3. Filter Battles by Model Coverage and Prompt Length

# %%
def get_first_turn(prompt_str):
    """Extract first turn from JSON-encoded prompt."""
    return json.loads(prompt_str)[0].strip()

def is_valid_battle(model_a, model_b, prompt_str):
    """
    Check if battle is valid for training.
    
    Criteria:
    1. Both models must be in our tier mapping (64 models)
    2. Battle must be between strong (tier 0-1) and weak (tier 2+) models
    3. Prompt must be >= 16 characters (after extracting first turn)
    
    This follows the paper's approach: only use battles between M_strong and M_weak.
    """
    # Both models must be in our tier mapping
    if model_a not in MODEL_TIERS or model_b not in MODEL_TIERS:
        return False
    
    # Check that one model is strong (tier 0-1) and the other is weak (tier 2+)
    tier_a = MODEL_TIERS[model_a]
    tier_b = MODEL_TIERS[model_b]
    
    is_a_strong = tier_a in [0, 1]
    is_b_strong = tier_b in [0, 1]
    
    # Must be exactly one strong and one weak (XOR)
    if is_a_strong == is_b_strong:
        return False
    
    # Extract first turn and check length
    try:
        first_turn = get_first_turn(prompt_str)
        if len(first_turn) < 16:
            return False
    except (json.JSONDecodeError, IndexError, KeyError):
        return False
    
    return True


def filter_battles(dataset):
    """Filter dataset to battles with known models and valid prompts."""
    def check_valid(example):
        return is_valid_battle(example["model_a"], example["model_b"], example["prompt"])
    
    filtered = dataset.filter(check_valid, desc="Filtering for valid battles")
    return filtered


filtered_dataset = filter_battles(raw_dataset)
print(f"Filtered dataset size: {len(filtered_dataset)}")
print(f"Retention: {100 * len(filtered_dataset) / len(raw_dataset):.1f}%")
print(f"Filtered out: {len(raw_dataset) - len(filtered_dataset)} samples")

# Sample from filtered
sample = filtered_dataset[0]
print("\nSample filtered battle:")
print(f"  model_a: {sample['model_a']}, tier: {MODEL_TIERS[sample['model_a']]}")
print(f"  model_b: {sample['model_b']}, tier: {MODEL_TIERS[sample['model_b']]}")
print(f"  winner_model_a: {sample['winner_model_a']}, winner_tie: {sample['winner_tie']}, winner_model_b: {sample['winner_model_b']}")

# %%
# Analyze model participation
all_models_in_filtered = []
for example in filtered_dataset:
    all_models_in_filtered.append(example['model_a'])
    all_models_in_filtered.append(example['model_b'])

model_participation = Counter(all_models_in_filtered)
print(f"\nModel participation in filtered dataset:")
print(f"Unique models: {len(model_participation)}")
print(f"Expected: {len(MODEL_TIERS)} models")

print(f"\nTop 10 most frequent models:")
for model, count in model_participation.most_common(10):
    tier = MODEL_TIERS[model]
    print(f"  {model} (tier {tier}): {count} battles")

# %% [markdown]
# ## 4. Extract Prompts and Map Outcomes

# %%
def map_outcome_to_label(example):
    """
    Map battle outcome to 3-class label based on winner's tier.
    
    Labels are assigned based on which tier won, regardless of matchup:
    - 0: A strong model (tier 0-1) won
    - 1: Tie
    - 2: A weak model (tier 2+) won
    
    Examples:
    - GPT-4 vs Mixtral, GPT-4 wins → label=0 (strong wins)
    - GPT-4 vs Mixtral, Mixtral wins → label=2 (weak wins)
    - Llama-7b vs Vicuna-7b, Llama wins → label=2 (weak wins, both are tier 3)
    - GPT-4 vs GPT-3.5, GPT-4 wins → label=0 (strong wins)
    """
    model_a = example["model_a"]
    model_b = example["model_b"]
    
    tier_a = MODEL_TIERS[model_a]
    tier_b = MODEL_TIERS[model_b]
    
    # Extract first turn as prompt
    example["prompt"] = get_first_turn(example["prompt"])
    
    # Map outcome based on winner's tier
    if example["winner_tie"] == 1:
        example["labels"] = 1  # tie
    elif example["winner_model_a"] == 1:
        # model_a won - check its tier
        example["labels"] = 0 if tier_a in [0, 1] else 2
    else:
        # model_b won - check its tier
        example["labels"] = 0 if tier_b in [0, 1] else 2
    
    return example


processed_dataset = filtered_dataset.map(
    map_outcome_to_label,
    desc="Mapping outcomes to labels"
)

print(f"Processed dataset size: {len(processed_dataset)}")

# Check label distribution
label_counts = Counter(processed_dataset["labels"])
print("\nLabel distribution:")
label_names = {0: "strong_wins", 1: "tie", 2: "weak_wins"}
for label in sorted(label_counts.keys()):
    count = label_counts[label]
    pct = 100 * count / len(processed_dataset)
    print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

# Sample
print("\nSample processed example:")
sample = processed_dataset[0]
print(f"  prompt: {sample['prompt'][:80]}...")
print(f"  labels: {sample['labels']} ({label_names[sample['labels']]})")

# %% [markdown]
# ## 5. Remove Unused Columns

# %%
# Keep only essential columns
d_arena = processed_dataset.select_columns(["prompt", "labels"])

print(f"D_arena dataset size: {len(d_arena)}")
print(f"D_arena columns: {d_arena.column_names}")

# Verify we can access required fields
sample = d_arena[0]
print(f"\nSample D_arena example:")
print(f"  prompt: {sample['prompt'][:80]}...")
print(f"  labels: {sample['labels']}")

# %% [markdown]
# ## 6. Create Train/Validation Split

# %%
from datasets import ClassLabel

# Create stratified 5k validation split as mentioned in the paper
# "we primarily use the 80K Chatbot Arena data for training our models, but hold out 5k samples for validation"

# First, cast labels to ClassLabel type (required for stratified splitting)
d_arena = d_arena.cast_column(
    "labels",
    ClassLabel(names=["strong_wins", "tie", "weak_wins"])
)

split_dataset = d_arena.train_test_split(
    test_size=0.1,
    stratify_by_column="labels",
    seed=42
)

d_arena_train = split_dataset["train"]
d_arena_val = split_dataset["test"]

print(f"\n=== Train/Validation Split ===")
print(f"Train size: {len(d_arena_train)}")
print(f"Validation size: {len(d_arena_val)}")

# Verify stratification worked
print(f"\nTrain label distribution:")
train_label_counts = Counter(d_arena_train["labels"])
for label in sorted(train_label_counts.keys()):
    count = train_label_counts[label]
    pct = 100 * count / len(d_arena_train)
    print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

print(f"\nValidation label distribution:")
val_label_counts = Counter(d_arena_val["labels"])
for label in sorted(val_label_counts.keys()):
    count = val_label_counts[label]
    pct = 100 * count / len(d_arena_val)
    print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

# %% [markdown]
# ## 6.5. Oversample Training Set to Balance Classes
#
# The training set has significant class imbalance (~51% strong_wins, ~8% tie, ~41% weak_wins).
# This causes models to collapse to always predicting the majority class.
# We oversample minority classes to create a balanced training set using scikit-learn's resample.

# %%
from sklearn.utils import resample
import numpy as np

# Get labels and indices
labels = np.array(d_arena_train["labels"])
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
d_arena_train_balanced = d_arena_train.select(balanced_indices)

# Print balanced distribution
print(f"\nBalanced class distribution:")
balanced_labels = np.array(d_arena_train_balanced["labels"])
for label in unique_labels:
    count = (balanced_labels == label).sum()
    pct = 100 * count / len(d_arena_train_balanced)
    print(f"  Class {label} ({label_names[label]}): {count} samples ({pct:.1f}%)")

print(f"\nTraining set size: {len(d_arena_train)} → {len(d_arena_train_balanced)}")
print(f"Validation set size: {len(d_arena_val)} (unchanged)")

# %% [markdown]
# ## 7. Save Processed Dataset

# %%
from datasets import DatasetDict

cache_dir = "/s3/lindenbauer/.cache/datasets"
d_arena_path = os.path.join(cache_dir, "d_arena")
os.makedirs(cache_dir, exist_ok=True)

# %%
# Save as DatasetDict with BALANCED train split and original validation split
d_arena_dict = DatasetDict({
    "train": d_arena_train_balanced,
    "validation": d_arena_val
})

d_arena_dict.save_to_disk(d_arena_path)
print(f"\nD_arena dataset saved to: {d_arena_path}")
print(f"  Train split: {len(d_arena_train_balanced)} examples (balanced via oversampling)")
print(f"  Validation split: {len(d_arena_val)} examples (original, unbalanced)")

# Verify save
from datasets import load_from_disk
d_arena_loaded = load_from_disk(d_arena_path)

print(f"\nLoaded D_arena from disk:")
print(f"  Train: {len(d_arena_loaded['train'])} examples")
print(f"  Validation: {len(d_arena_loaded['validation'])} examples")
print(f"  Columns: {d_arena_loaded['train'].column_names}")

# Verify training set is balanced
print(f"\nTrain split label distribution (balanced):")
train_balanced_labels = Counter(d_arena_loaded['train']['labels'])
for label in sorted(train_balanced_labels.keys()):
    count = train_balanced_labels[label]
    pct = 100 * count / len(d_arena_loaded['train'])
    print(f"  {label_names[label]}: {count} ({pct:.1f}%)")

# Final statistics
print("\n=== D_Arena Dataset Summary ===")
print(f"Original dataset size: {len(d_arena)}")
print(f"Original train size: {len(d_arena_train)}")
print(f"Balanced train size: {len(d_arena_train_balanced)} (oversampled minority classes)")
print(f"Validation size: {len(d_arena_val)} (unchanged)")
print(f"Source: lmsys/lmsys-arena-human-preference-55k (all battles, labeled by tier)")
    
print(f"\nNote: Training set is balanced via oversampling to prevent class collapse.")
print(f"Validation set preserves original class distribution for proper evaluation.")
print(f"Labels indicate which tier won (strong=0-1, weak=2+), not the matchup type.")

# %% [markdown]
# # D_arena Classifier Training

# %%
from datasets import load_from_disk, DatasetDict
cache_dir = "/s3/lindenbauer/.cache/datasets"
d_arena_path = os.path.join(cache_dir, "d_arena")

d_arena_loaded = load_from_disk(d_arena_path)


# %%
def build_subset_indices(
    dataset, 
    fraction: Optional[float] = None,
    n_indices: Optional[int] = None,
    seed: int = 42
) -> list[int]:
    """
    Build a random subset of indices from a dataset.
    
    Args:
        dataset: The dataset to sample from
        fraction: Fraction of data to keep (0.0 to 1.0). Mutually exclusive with n_indices.
        n_indices: Exact number of indices to sample. Mutually exclusive with fraction.
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled indices, sorted
    """
    # Validate that exactly one parameter is set
    if (fraction is None) == (n_indices is None):
        raise ValueError("Exactly one of 'fraction' or 'n_indices' must be set")
    
    total_size = len(dataset)
    
    # Calculate subset_size based on provided parameter
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Fraction must be in (0.0, 1.0], got {fraction}")
        subset_size = int(total_size * fraction)
    else:
        if n_indices <= 0:
            raise ValueError(f"n_indices must be positive, got {n_indices}")
        if n_indices > total_size:
            raise ValueError(f"n_indices ({n_indices}) cannot exceed dataset size ({total_size})")
        subset_size = n_indices
    
    rng = random.Random(seed)
    all_indices = list(range(total_size))
    rng.shuffle(all_indices)
    
    return sorted(all_indices[:subset_size])


# %% [markdown]
# ## D_arena Overfit Experiments
#
# These experiments test if the training pipeline can overfit to small datasets,
# which would indicate that the data loading and model training are working correctly.

# %%
def build_d_arena_overfit_indices(dataset, batch_size: int, n_batches: int, seed: int = 1027) -> dict[str, list[int]]:
    """
    Build overfitting indices for D_arena dataset.

    Args:
        dataset: The dataset to sample from
        batch_size: batch size to use for this experimetn
        n_batches: number of batches to overfit to
        seed: Random seed for reproducibility

    Returns:
        Dictionary with overfitting scenarios
    """
    indices = {
        "single_sample": build_subset_indices(dataset, n_indices=1, seed=seed),
        "four_samples": build_subset_indices(dataset, n_indices=4, seed=seed),
        f"{n_batches}_batches": build_subset_indices(dataset, n_indices=batch_size * n_batches, seed=seed),
    }
    return indices


def run_d_arena_overfit_training(
    scenario_key: str,
    *,
    output_subdir: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    max_steps: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    eval_strategy: str = "steps",
    report_to: str = "wandb",
    wandb_project: str = "routellm-bert-classifier",
    cuda_device: Optional[int] = 0,
    seed: int = 1027,
    n_batches: int = 4,
    freeze_encoder: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run overfitting experiments on D_arena dataset.
    """
    # Load D_arena dataset
    d_arena_loaded = load_from_disk(d_arena_path)
    train_dataset = d_arena_loaded["train"]

    # Build overfitting indices for this scenario
    overfit_indices = build_d_arena_overfit_indices(train_dataset, per_device_train_batch_size, n_batches, seed=seed)
    train_indices = overfit_indices[scenario_key]

    # Use same indices for evaluation
    eval_indices = train_indices

    # Run training
    result = run_bert_classifier_training(
        dataset_path=d_arena_path,
        train_split_name="train",
        eval_split_name="train",
        train_indices=train_indices,
        eval_indices=eval_indices,
        output_subdir=output_subdir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        report_to=report_to,
        wandb_project=wandb_project,
        wandb_run_name=f"d_arena_overfit_{scenario_key}",
        wandb_log_model="false",
        cuda_device=cuda_device,
        seed=seed,
        freeze_encoder=freeze_encoder,
    )

    return result


# %% [markdown]
# ### Overfit to Single Sample (Frozen Encoder)
#
# Test if frozen encoder can overfit to a single sample.
# This diagnostic checks if the architecture can learn at all with frozen weights.

# %%
run_d_arena_overfit_training(
    scenario_key="single_sample",
    output_subdir="d_arena-classifier_only-overfit_single-retry-refactor",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=10,
    save_steps=10,
    eval_steps=10,
    logging_steps=5,
    learning_rate=1e-3,
    weight_decay=0.0,
    freeze_encoder=True,
)

# %% [markdown]
# ### Overfit to Four Samples

# %%
run_d_arena_overfit_training(
    scenario_key="four_samples",
    output_subdir="d_arena-classifier_only-overfit_four",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=200,
    save_steps=200,
    eval_steps=10,
    logging_steps=5,
    learning_rate=1e-3,
    weight_decay=0.0,
    freeze_encoder=True,
)

# %% [markdown]
# ### Overfit to Four Batches (64 samples)

# %%
run_d_arena_overfit_training(
    scenario_key="4_batches",
    output_subdir="d_arena-classifier_only-overfit_4_batches",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=400,
    save_steps=400,
    eval_steps=25,
    logging_steps=10,
    learning_rate=1e-3,
    weight_decay=0.0,
    n_batches=4,
    freeze_encoder=True,
)

# %% [markdown]
# ## Full dataset
# Train the BERT classifier on the full D_arena dataset using the paper's hyperparameters:
# - ~2000 steps
# - Batch size 16
# - Max sequence length 512
# - Learning rate 1e-5
# - Weight decay 0.01
# - Full parameter fine-tuning

# %% [markdown]
# ## Training Setup 1: Quick Sanity Check (10% of data)
#
# Before running the full training, we do a quick sanity check to verify:
# - Data pipeline works correctly
# - Model learns (training loss decreases)
# - Eval metrics improve from random baseline
# - WandB logging works
#
# This run uses:
# - 10%/50% of training data (~5,900 samples)
# - Expected runtime: ~15-20 minutes

# %%
# Build 10% subset indices
d_arena_loaded_for_subset = load_from_disk(d_arena_path)
train_subset_10pct = build_subset_indices(d_arena_loaded_for_subset["train"], fraction=0.1, seed=42)
train_subset_50pct = build_subset_indices(d_arena_loaded_for_subset["train"], fraction=0.5, seed=42)

# %%
run_bert_classifier_training(
    train_indices=train_subset_10pct,
    output_subdir="d_arena-10pct-w_s",
    max_steps=500,
    per_device_train_batch_size=64,  # Increased from 16 to improve gradient estimates
    per_device_eval_batch_size=64,  # Larger for faster eval on H200
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=25, 
    save_steps=500,  # Save only at end
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_arena-10pct-w_s-bs64",  # Updated run name to reflect batch size
)

# %%
run_bert_classifier_training(
    train_indices=train_subset_50pct,
    output_subdir="d_arena-50pct-w_s",
    max_steps=1800,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=100, 
    save_steps=500,  # Save only at end
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_arena-50pct-w_s-bs128",
)

# %% [markdown]
# ## Training Setup 2: Full BERT Classifier (100% data, 2000 steps)
#
# Full training run following the paper's specifications:
# - Full D_arena training set (~59k samples)
# - 2000 steps (~0.54 epochs with batch size 16)
# - Eval every 100 steps (20 total evaluations)
# - Larger eval batch size (64) to reduce overhead on H200
# - Expected runtime: ~2-3 hours
#
# Paper specifications:
# - Batch size: 16
# - Max sequence length: 512
# - Learning rate: 1e-5
# - Weight decay: 0.01
# - Full parameter fine-tuning

# %%
run_bert_classifier_training(
    train_indices=None,
    output_subdir="d_arena_full-w_s",
    max_steps=1500,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_length=512,
    eval_steps=225, 
    save_steps=4500,  # Save only at end
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_arena-w_s-bs64",
)

# %% [markdown]
# ## Training Setup 3: Deviating from training configuration in paper
#
# Improvements over baseline to try:
# - Batch size 128 (vs 16) for better gradient estimates
# - Weight decay 0.02 (vs 0.01) for stronger regularization
# - Explicit warmup ratio configuration
# - Train for more steps
# - Cosine decay

# %% [markdown]
# ### Warmup

# %%
run_bert_classifier_training(
    train_indices=None,
    output_subdir="d_arena_full-w_s-d_arena-w_s-bs128_warmup01_wdecay005",
    max_steps=1250,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.05,
    max_length=512,
    eval_steps=125,
    save_steps=1250,  # Save only at end
    logging_steps=10,
    wandb_project="routellm-bert-classifier",
    wandb_run_name="d_arena-w_s-bs128_warmup0.1_wdecay0.05",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_total_limit=1,
)


# %% [markdown]
# # Evaluate RouteLLM's Released Checkpoint
#
# Load the authors' released `routellm/bert` checkpoint and evaluate it on our D_arena validation set.
# This will help us understand:
# 1. How well their checkpoint performs on our (potentially different) validation split
# 2. If the difference in training data affects performance
# 3. What metrics they were able to achieve (as a reference point)

# %%
def evaluate_checkpoint_on_d_arena(
    model_path: str,
    d_arena_path: str = d_arena_path,
    num_labels: int = 3,
    max_length: int = 512,
    per_device_eval_batch_size: int = 128,
    cuda_device: Optional[int] = 0,
) -> dict:
    """
    Evaluate a HuggingFace model checkpoint on the D_arena validation set.

    Args:
        model_path: Path to the HuggingFace model checkpoint
        d_arena_path: Path to the D_arena dataset on disk
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
    d_arena_loaded = load_from_disk(d_arena_path)
    eval_dataset = d_arena_loaded["validation"]

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


# Evaluate the RouteLLM checkpoint
metrics = evaluate_checkpoint_on_d_arena("routellm/bert")

# %%
metrics = evaluate_checkpoint_on_d_arena("routellm/bert_gpt4_augmented")

# %% [markdown]
# ## Training Setup 4: Tuning on balanced data
#
# Because my classifiers trained on full data actually beat both their snapshots by quite a bit, I will retrain and also change the metric for selecting the model to F1 as it seems like a better fit than eval loss or accuracy.
#
# Experiments to run
# 1. Train full classifier head
# 2. Train full model with just output proj classifier head (no feat. extraction)

# %%
run_bert_classifier_training(
    train_indices=None,
    output_subdir="d_arena-s_cls_full_model-balanced_data-bs128",
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
    wandb_run_name="d_arena-s_cls_full_model-balanced_data-bs128",
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2, 
    freeze_encoder=True,
)

# %%
