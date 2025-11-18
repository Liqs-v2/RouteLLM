#!/usr/bin/env python3
"""
Launch script for BERT classifier training on D_arena dataset.
Can be run in a screen session for long-running training jobs.

Usage:
    source .venv/bin/activate
    python reproduce/launch_training.py
"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset, load_from_disk


# Configure cache directories
cache_dir = "/s3/lindenbauer/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# Setup paths
project_root = Path("/mnt/shared-fs/lindenbauer/RouteLLM")
script_path = project_root / "reproduce" / "train_roberta.py"
models_root = Path("/s3/lindenbauer/RouteLLM/models")
models_root.mkdir(parents=True, exist_ok=True)

d_arena_path = os.path.join(cache_dir, "datasets", "d_arena")


def format_indices(indices: Iterable[int], threshold: int = 100) -> str:
    """
    Format indices as comma-separated string or save to temp file if too long.
    
    Args:
        indices: Iterable of integer indices
        threshold: If more than this many indices, save to file instead
    
    Returns:
        Either comma-separated string or '@filepath' reference
    """
    indices_list = list(indices)
    
    # If short enough, return as comma-separated string
    if len(indices_list) <= threshold:
        return ",".join(str(idx) for idx in indices_list)
    
    # Otherwise, write to temp file and return file reference
    temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='train_indices_')
    try:
        with os.fdopen(temp_fd, 'w') as f:
            for idx in indices_list:
                f.write(f"{idx}\n")
        return f"@{temp_path}"
    except:
        # Clean up temp file if writing failed
        os.unlink(temp_path)
        raise


def run_bert_classifier_training(
    dataset_path: Optional[str] = None,
    train_split_name: str = "train",
    eval_split_name: str = "validation",
    train_indices: Optional[Iterable[int]] = None,
    eval_indices: Optional[Iterable[int]] = None,
    output_subdir: str = "bert_classifier_d_arena",
    max_steps: int = 2000,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    eval_steps: int = 100,
    save_steps: int = 500,
    logging_steps: int = 10,
    eval_strategy: str = "steps",
    report_to: str = "wandb",
    wandb_project: str = "my-awesome-project",
    wandb_run_name: str = "bert_d_arena_full",
    cuda_device: Optional[int] = 0,
    seed: int = 42,
) -> subprocess.CompletedProcess:
    """
    Run full parameter SFT of RoBERTa classifier training on a specified dataset.
    
    This follows the paper's training setup:
    - BERT-base architecture (xlm-roberta-base)
    - Full parameter fine-tuning
    - Trained on preference data
    
    Args:
        dataset_path: Path to dataset on disk or HF hub name (defaults to d_arena_path)
        train_split_name: Name of training split
        eval_split_name: Name of evaluation split
        train_indices: Optional list of indices to select from train split
        eval_indices: Optional list of indices to select from eval split
        output_subdir: Subdirectory name under models_root for outputs
        max_steps: Maximum training steps
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_length: Maximum sequence length for tokenization
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        eval_strategy: Evaluation strategy ("steps", "epoch", or "no")
        report_to: Where to report metrics (e.g., "wandb")
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        cuda_device: CUDA device index
        seed: Random seed
    
    Returns:
        subprocess.CompletedProcess result
    """
    # Use default d_arena_path if not provided
    if dataset_path is None:
        dataset_path = d_arena_path
    
    # Load dataset from disk or hub to get size info
    if os.path.isdir(dataset_path):
        dataset_loaded = load_from_disk(dataset_path)
    else:
        dataset_loaded = load_dataset(dataset_path)
    
    output_dir = models_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        sys.executable,
        str(script_path),
        "--dataset-name",
        dataset_path,
        "--train-split",
        train_split_name,
        "--eval-split",
        eval_split_name,
        "--output-dir",
        str(output_dir),
        "--model-name",
        "FacebookAI/xlm-roberta-base",
        "--num-labels",
        "3",
        "--max-length",
        str(max_length),
        "--max-steps",
        str(max_steps),
        "--per-device-train-batch-size",
        str(per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(per_device_eval_batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--eval-strategy",
        eval_strategy,
        "--eval-steps",
        str(eval_steps),
        "--save-steps",
        str(save_steps),
        "--logging-steps",
        str(logging_steps),
        "--report-to",
        report_to,
        "--wandb-project",
        wandb_project,
        "--wandb-run-name",
        wandb_run_name,
        "--seed",
        str(seed),
    ]
    
    if train_indices is not None:
        command.extend(["--train-indices", format_indices(train_indices)])
    if eval_indices is not None:
        command.extend(["--eval-indices", format_indices(eval_indices)])
    if cuda_device is not None:
        command.extend(["--cuda-device", str(cuda_device)])
    
    # Calculate effective dataset sizes
    train_size = len(dataset_loaded[train_split_name])
    eval_size = len(dataset_loaded[eval_split_name])
    if train_indices is not None:
        train_size = len(list(train_indices))
    if eval_indices is not None:
        eval_size = len(list(eval_indices))
    
    print("=" * 80)
    print("BERT Classifier Training Configuration")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Train split: {train_split_name} ({train_size} examples)")
    print(f"Eval split: {eval_split_name} ({eval_size} examples)")
    print(f"Model: FacebookAI/xlm-roberta-base")
    print(f"Output: {output_dir}")
    print(f"\nTraining Hyperparameters:")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Save every: {save_steps} steps")
    print(f"\nWandB:")
    print(f"  Project: {wandb_project}")
    print(f"  Run name: {wandb_run_name}")
    print("=" * 80)
    
    print("\nLaunching training:", " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    print(f"\nTraining script finished with return code {result.returncode}")
    
    if result.stdout:
        print("\nTraining stdout:")
        print(result.stdout)
    
    if result.stderr:
        print("\nTraining stderr:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=command,
            output=result.stdout,
            stderr=result.stderr,
        )
    
    return result


def main():
    """Launch full BERT classifier training on D_arena dataset."""
    print("Starting full BERT classifier training on D_arena dataset")
    print(f"Dataset path: {d_arena_path}")
    print(f"Models output: {models_root}")
    print("")
    
    # Verify dataset exists
    if not os.path.exists(d_arena_path):
        print(f"ERROR: Dataset not found at {d_arena_path}")
        print("Please run the notebook cells to create the D_arena dataset first.")
        sys.exit(1)
    
    # Run full training (2000 steps, ~2-3 hours on H200)
    run_bert_classifier_training(
        output_subdir="bert_classifier_d_arena_full",
        max_steps=2000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,  # Larger for faster eval on H200
        learning_rate=1e-5,
        weight_decay=0.01,
        max_length=512,
        eval_steps=100,  # Evaluate every 100 steps
        save_steps=2000,
        logging_steps=10,
        wandb_project="routellm-bert-classifier",
        wandb_run_name="bert_d_arena_full_2000steps-55k_all_battles",
    )
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

