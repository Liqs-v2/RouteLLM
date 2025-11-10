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
# # WandB example

# %%
import wandb

import os
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import notebook_login
import torch


def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


# %% [markdown]
# # Download model and data

# %% [markdown]
# ## Data

# %%
dataset = load_dataset("routellm/mmlu_battles")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

# %%
dataset['train'].unique('model_b')

# %% [markdown]
# Because the model columns just contain one model, and we are implementing binary classification, but we have results in (win_a, win_b, tie), we need to map to just `win_b' = win_b or tie`. This way we route to the smaller model whenever possible.
#
# Actually, despite the authors stating that they implement a binary router (ie imply that ties are routed to the small model), their training objective is on (win_a, tie, win_b). So we'll follow that.

# %%
dataset['train'][0]

# %% [markdown]
# ## Model

# %%
# download the model
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=3)

# %%
model

# %% [markdown]
# The classifier dimensions seem correct and match their model dimensions as shown in HF when inspecting the metadata.

# %% [markdown]
# # Data prep

# %%
train_dataset = dataset['train'].map(tokenize_function, batched=True)

# %%
label_columns = ["winner_model_a", "winner_tie", "winner_model_b"]

def add_label(example):
    for idx, col in enumerate(label_columns):
        if example[col]:
            example["labels"] = idx
            break
    return example

train_dataset = train_dataset.map(add_label, batched=False)

# %% [markdown]
# # Overfit to a single sample

# %%
overfit_single_dataset = train_dataset.select([2])

# %%
overfit_single_dataset

# %%
torch.cuda.set_device(0)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="my-awesome-project"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="end"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

# %%
# pass "wandb" to the 'report_to' parameter to turn on wandb logging
training_args = TrainingArguments(
    output_dir='models',
    report_to="wandb",
    logging_steps=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=10,
    max_steps = 100,
    save_steps = 100,
    learning_rate = 1e-5,
    weight_decay = 0.01
)

# define the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=overfit_one_batch,
    eval_dataset=overfit_one_batch,
    compute_metrics=compute_metrics,
)
trainer.train()

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# %%
import subprocess
import sys

script_path = "/mnt/shared-fs/lindenbauer/RouteLLM/reproduce/train_roberta.py"
output_dir = "/mnt/shared-fs/lindenbauer/RouteLLM/models"

command = [
    sys.executable,
    script_path,
    "--dataset-name",
    "routellm/mmlu_battles",
    "--train-split",
    "train",
    "--train-indices",
    "2",
    "--output-dir",
    output_dir,
    "--model-name",
    "FacebookAI/xlm-roberta-base",
    "--evaluation-strategy",
    "steps",
    "--per-device-train-batch-size",
    "1",
    "--per-device-eval-batch-size",
    "1",
    "--eval-steps",
    "10",
    "--max-steps",
    "100",
    "--save-steps",
    "100",
    "--logging-steps",
    "5",
    "--report-to",
    "wandb",
    "--cuda-device",
    "0",
    "--wandb-project",
    "my-awesome-project",
    "--wandb-log-model",
    "end",
    "--wandb-watch",
    "false",
]

result = subprocess.run(command, check=True)
print(f"Training script finished with return code {result.returncode}")


# %%
overfit_single_dataset[0]

# %%
from tqdm import tqdm
from collections import Counter

N = 1000
model.eval()
pred_ids = []
for _ in tqdm(range(N)):
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids.append(logits.argmax(dim=-1).item())

counts = Counter(pred_ids)
label_counts = {id_to_label[idx]: counts.get(idx, 0) for idx in id_to_label}
normalized_counts = {label: count / N for label, count in label_counts.items()}

print("absolute_counts:", label_counts)
print("normalized_counts:", normalized_counts)


# %% [markdown]
# Great! Overfitting onto a single sample seems to work. Let's slowly scale up the overfitting tests to a few samples, to a batch and a larger subset of the this small dataset.

# %% [markdown]
# # Overfit to a small subset < batch size (so S=4)

# %%
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base", num_labels=3)

# %%
overfit_small_subset = train_dataset.shuffle(seed=1027).take(8)

# %%
# pass "wandb" to the 'report_to' parameter to turn on wandb logging
training_args = TrainingArguments(
    output_dir='models',
    report_to="wandb",
    logging_steps=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=10,
    max_steps = 200,
    save_steps = 200,
    learning_rate = 1e-5,
    weight_decay = 0.01
)

# define the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=overfit_one_batch,
    eval_dataset=overfit_one_batch,
    compute_metrics=compute_metrics,
)
trainer.train()

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
