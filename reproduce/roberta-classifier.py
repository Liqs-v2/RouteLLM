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
import random
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional

from datasets import load_dataset

# %% [markdown]
# ## Data

# %%
DATASET_NAME = "routellm/mmlu_battles"


# %%
dataset = load_dataset("routellm/mmlu_battles")


# %% [markdown]
# # Overfit to a single sample

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
models_root = project_root / "reproduce" / "models"
models_root.mkdir(parents=True, exist_ok=True)


def format_indices(indices: Iterable[int]) -> str:
    return ",".join(str(idx) for idx in indices)


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
    result = subprocess.run(command, check=True)
    print(f"Training script finished with return code {result.returncode}")


# %% [markdown]
# ## Overfit to a Single Sample


# %%
run_overfit_training(
    scenario_key="single_sample",
    output_subdir="overfit_single",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=100,
    save_steps=100,
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
