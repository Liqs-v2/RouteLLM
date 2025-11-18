import os
import random
import tempfile
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional

from datasets import load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = PROJECT_ROOT / "train_roberta.py"
MODELS_ROOT = Path("/s3/lindenbauer/RouteLLM/models")
D_ARENA_PATH = Path("/s3/lindenbauer/.cache/datasets/d_arena")

def format_indices(indices: Iterable[int], threshold: int = 100) -> str:
    """
    Format indices as comma-separated string or save to temp file if too long.
    
    Args:
        indices: Iterable of integer indices
        threshold: If more than this many indices, save to file instead
    
    Returns:
        Either comma-separated string or '@filepath' reference JA
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
    *,
    model_name_or_path: Path | str = "FacebookAI/xlm-roberta-base",
    tokenizer_name_or_path: Path | str | None = None,
    script_path: Path | str = SCRIPT_PATH,
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
    wandb_log_model: str = "end",
    cuda_device: Optional[int] = 0,
    seed: int = 42,
    lr_scheduler_type: str = "linear",
    warmup_ratio: Optional[float] = 0.1,
    warmup_steps: Optional[int] = None,
    save_total_limit: Optional[int] = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: Optional[str] = None,
    greater_is_better: bool = False,
    freeze_encoder: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run full parameter SFT of RoBERTa classifier training on a specified dataset.
    
    This follows the paper's training setup:
    - BERT-base architecture (xlm-roberta-base)
    - Full parameter fine-tuning
    - Trained on preference data
    
    Args:
        dataset_path: Path to dataset on disk or HF hub name (defaults to `D_ARENA_PATH`)
        train_split_name: Name of training split
        eval_split_name: Name of evaluation split
        model_name_or_path: Hugging Face model identifier or local checkpoint path
        tokenizer_name_or_path: Hugging Face tokenizer identifier/path (defaults to model)
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
        wandb_log_model: When to log model to wandb ("end", "checkpoint", "false")
        cuda_device: CUDA device index
        seed: Random seed
        lr_scheduler_type: Learning rate scheduler strategy (defaults to linear decay)
        warmup_ratio: Fraction of training steps used for warmup (ignored if warmup_steps is set)
        warmup_steps: Explicit number of warmup steps (overrides warmup_ratio if provided)
        save_total_limit: Maximum number of checkpoints to keep (None = keep all)
        load_best_model_at_end: Whether to load best checkpoint at end of training
        metric_for_best_model: Metric to track for best model (e.g., 'eval_loss', 'eval_accuracy')
        greater_is_better: Whether higher metric values are better (False for loss, True for accuracy)
        freeze_encoder: Whether to freeze encoder parameters and only train the classifier head
    
    Returns:
        subprocess.CompletedProcess result
    """
    # Use default D_arena path if not provided
    if dataset_path is None:
        dataset_path = str(D_ARENA_PATH)

    script_path = Path(script_path)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_path_str = str(dataset_path)
    model_name_or_path_str = str(model_name_or_path)
    tokenizer_name_or_path_str = (
        str(tokenizer_name_or_path)
        if tokenizer_name_or_path is not None
        else None
    )
    
    # Load dataset from disk or hub to get size info
    if os.path.isdir(dataset_path_str):
        dataset_loaded = load_from_disk(dataset_path_str)
    else:
        dataset_loaded = load_dataset(dataset_path_str)
    
    output_dir = MODELS_ROOT / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        sys.executable,
        str(script_path),
        "--dataset-name",
        dataset_path_str,
        "--train-split",
        train_split_name,
        "--eval-split",
        eval_split_name,
        "--output-dir",
        str(output_dir),
        "--model-name",
        model_name_or_path_str,
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
        "--wandb-log-model",
        wandb_log_model,
    ]
    
    if tokenizer_name_or_path_str is not None:
        command.extend(["--tokenizer-name", tokenizer_name_or_path_str])
    if lr_scheduler_type is not None:
        command.extend(["--lr-scheduler-type", lr_scheduler_type])
    if warmup_steps is not None:
        command.extend(["--warmup-steps", str(warmup_steps)])
    elif warmup_ratio is not None:
        command.extend(["--warmup-ratio", str(warmup_ratio)])
    if save_total_limit is not None:
        command.extend(["--save-total-limit", str(save_total_limit)])
    if load_best_model_at_end:
        command.extend(["--load-best-model-at-end"])
        if metric_for_best_model:
            command.extend(["--metric-for-best-model", metric_for_best_model])
        if greater_is_better:
            command.extend(["--greater-is-better"])
    if freeze_encoder:
        command.extend(["--freeze-encoder"])
    
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
    print(f"Dataset: {dataset_path_str}")
    print(f"Train split: {train_split_name} ({train_size} examples)")
    print(f"Eval split: {eval_split_name} ({eval_size} examples)")
    print(f"Model: {model_name_or_path_str}")
    if tokenizer_name_or_path_str is not None:
        print(f"Tokenizer: {tokenizer_name_or_path_str}")
    print(f"Output: {output_dir}")
    print(f"\nTraining Hyperparameters:")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  LR scheduler: {lr_scheduler_type}")
    if warmup_steps is not None:
        print(f"  Warmup steps: {warmup_steps}")
    else:
        print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Save every: {save_steps} steps")
    if save_total_limit is not None:
        print(f"  Save total limit: {save_total_limit}")
    if load_best_model_at_end:
        print(f"  Load best model at end: True")
        print(f"  Best model metric: {metric_for_best_model or 'eval_loss'}")
        print(f"  Greater is better: {greater_is_better}")
    print(f"\nWandB:")
    print(f"  Project: {wandb_project}")
    print(f"  Run name: {wandb_run_name}")
    print(f"  Log model: {wandb_log_model}")
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