#!/usr/bin/env python3
"""
Train an XLM-RoBERTa sequence classification model outside of a notebook process.

This script mirrors the single-sample training flow from `roberta-classifier.ipynb`,
but executes in its own process to ensure GPU resources are released once training
completes. The script is parameterised so it can be reused for other splits or
training argument configurations.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


LABEL_COLUMNS = ["winner_model_a", "winner_tie", "winner_model_b"]


def parse_indices(indices: Optional[str]) -> Optional[List[int]]:
    """
    Parse indices from a comma-separated string or a file.
    
    If indices starts with '@', treat the rest as a file path containing
    one integer per line. Otherwise, parse as comma-separated integers.
    """
    if indices is None:
        return None
    
    # If starts with '@', read from file
    if indices.startswith('@'):
        filepath = indices[1:]  # Remove '@' prefix
        with open(filepath, 'r') as f:
            result = [int(line.strip()) for line in f if line.strip()]
        return result or None
    
    # Otherwise parse as comma-separated string
    cleaned = [segment.strip() for segment in indices.split(",")]
    result = [int(segment) for segment in cleaned if segment]
    return result or None


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


def add_label(example):
    for idx, column in enumerate(LABEL_COLUMNS):
        if example.get(column):
            example["labels"] = idx
            return example
    raise ValueError(
        f"Could not determine label for example with id={example.get('id')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a router classifier.")
    parser.add_argument(
        "--dataset-name",
        default="routellm/mmlu_battles",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split name to use for training.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Optional split name to use for evaluation (defaults to train split).",
    )
    parser.add_argument(
        "--train-indices",
        default=None,
        help="Comma separated list of integer indices or @filepath containing one index per line.",
    )
    parser.add_argument(
        "--eval-indices",
        default=None,
        help="Comma separated list of integer indices or @filepath containing one index per line.",
    )
    parser.add_argument(
        "--model-name",
        default="FacebookAI/xlm-roberta-base",
        help="Pretrained model identifier.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=3,
        help="Number of output labels.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and the final model will be written.",
    )
    parser.add_argument(
        "--eval-strategy",
        default="no",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy passed to TrainingArguments.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Run evaluation every N steps (only when eval_strategy='steps').",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=500,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="Learning rate scheduler type (passed to TrainingArguments).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max sequence length for tokenization.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index to bind to before training.",
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Destination for Trainer reporting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--wandb-project",
        default='my-awesome-project',
        help="Optional WANDB project name (sets WANDB_PROJECT env var).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional WANDB run name (sets WANDB_NAME env var).",
    )
    parser.add_argument(
        "--wandb-log-model",
        default='end',
        help="Optional WANDB_LOG_MODEL setting.",
    )
    parser.add_argument(
        "--wandb-watch",
        type=bool,
        default=False,
        help="Optional WANDB_WATCH setting.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        default=False,
        help="Load best model at end of training based on metric_for_best_model.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default=None,
        help="Metric to use for selecting best model (e.g., 'eval_loss', 'eval_accuracy'). Only used if load_best_model_at_end is True.",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        default=False,
        help="Whether higher metric values are better (False for loss, True for accuracy). Only used if load_best_model_at_end is True.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help="Freeze encoder parameters and only train the classifier head.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.warmup_steps is not None and args.warmup_ratio is not None:
        raise ValueError("Specify at most one of warmup_steps or warmup_ratio")

    # Set HuggingFace cache directory
    cache_dir = "/s3/lindenbauer/.cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

    # Control GPU visibility using CUDA_VISIBLE_DEVICES (must be done before GPU initialization)
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    if args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    if args.wandb_watch:
        os.environ["WANDB_WATCH"] = args.wandb_watch

    tokenizer_kwargs = {"padding": "max_length", "truncation": True}
    if args.max_length is not None:
        tokenizer_kwargs["max_length"] = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset from either local disk or HuggingFace Hub
    if os.path.isdir(args.dataset_name):
        print(f"Loading dataset from local disk: {args.dataset_name}")
        dataset = load_from_disk(args.dataset_name)
    else:
        print(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], **tokenizer_kwargs)

    def prepare_split(split_name: str, indices: Optional[List[int]]):
        raw_split = dataset[split_name]
        base_split = raw_split.map(
            tokenize_function,
            batched=True,
            desc=f"Tokenising {split_name}",
        )
        # Only add labels if they don't already exist (e.g., for D_arena dataset)
        if "labels" not in base_split.column_names:
            base_split = base_split.map(add_label, batched=False)

        if indices:
            return base_split.select(indices)
        return base_split

    train_indices = parse_indices(args.train_indices)
    eval_indices = parse_indices(args.eval_indices)

    train_dataset = prepare_split(args.train_split, train_indices)

    eval_dataset = None
    target_eval_split = args.eval_split or args.train_split
    if args.eval_strategy != "no":
        effective_eval_indices = eval_indices
        if (
            effective_eval_indices is None
            and target_eval_split == args.train_split
            and train_indices is not None
        ):
            effective_eval_indices = train_indices
        eval_dataset = prepare_split(target_eval_split, effective_eval_indices)

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy=args.eval_strategy
        if eval_dataset is not None
        else "no",
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    if args.lr_scheduler_type is not None:
        training_args_kwargs["lr_scheduler_type"] = args.lr_scheduler_type
    if args.report_to is not None:
        training_args_kwargs["report_to"] = args.report_to
    if args.eval_steps is not None and training_args_kwargs["eval_strategy"] == "steps":
        training_args_kwargs["eval_steps"] = args.eval_steps
    if args.max_steps is not None:
        training_args_kwargs["max_steps"] = args.max_steps
    if args.save_steps is not None:
        training_args_kwargs["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = args.save_total_limit
    if args.warmup_steps is not None:
        training_args_kwargs["warmup_steps"] = args.warmup_steps
    if args.warmup_ratio is not None:
        training_args_kwargs["warmup_ratio"] = args.warmup_ratio
    if args.load_best_model_at_end:
        training_args_kwargs["load_best_model_at_end"] = True
        if args.metric_for_best_model:
            training_args_kwargs["metric_for_best_model"] = args.metric_for_best_model
        training_args_kwargs["greater_is_better"] = args.greater_is_better

    training_args = TrainingArguments(**training_args_kwargs)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    # Freeze encoder parameters if requested
    if args.freeze_encoder:
        # For XLM-RoBERTa, the encoder is accessed via model.roberta
        for param in model.roberta.parameters():
            param.requires_grad = False
        
        # Count trainable and total parameters for verification
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params
        
        print(f"Encoder frozen: {frozen_params:,} parameters frozen, {trainable_params:,} parameters trainable (out of {total_params:,} total)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()

    metrics = dict(train_result.metrics)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")

    if wandb is not None and args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()

