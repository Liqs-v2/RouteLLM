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
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# %%
PROJECT_ROOT = Path("/mnt/shared-fs/lindenbauer/RouteLLM")
GSM8K_RESPONSES_PATH = PROJECT_ROOT / "routellm/evals/gsm8k/gsm8k_responses.csv"
MMLU_RESPONSES_DIR = PROJECT_ROOT / "routellm/evals/mmlu/responses"
MT_BENCH_JUDGEMENTS_PATH = PROJECT_ROOT / "routellm/evals/mt_bench/judgements.jsonl"

STRONG_MODEL_COL = "gpt-4-1106-preview"
WEAK_MODEL_COL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

STRING_TO_BOOL: Dict[str, bool] = {"true": True, "false": False}


def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """Convert True/False-like entries to booleans."""
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(STRING_TO_BOOL)
    )
    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique()
        raise ValueError(f"Encountered non boolean values: {bad_values}")
    return normalized


def load_gsm8k_responses() -> pd.DataFrame:
    """Load the GSM8K evaluation CSV and ensure boolean correctness columns."""
    df = pd.read_csv(GSM8K_RESPONSES_PATH)
    df[STRONG_MODEL_COL] = normalize_boolean_column(df[STRONG_MODEL_COL])
    df[WEAK_MODEL_COL] = normalize_boolean_column(df[WEAK_MODEL_COL])
    return df


def load_mmlu_responses() -> pd.DataFrame:
    """Load all MMLU evaluation CSVs and concatenate into a single DataFrame."""
    csv_files = list(MMLU_RESPONSES_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {MMLU_RESPONSES_DIR}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name  # Track which file this came from
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Normalize boolean columns - MMLU files may have mixed formats
    combined_df[STRONG_MODEL_COL] = normalize_boolean_column(combined_df[STRONG_MODEL_COL])
    combined_df[WEAK_MODEL_COL] = normalize_boolean_column(combined_df[WEAK_MODEL_COL])

    return combined_df


def load_mt_bench_responses() -> pd.DataFrame:
    """Load MT-bench judgements and create router labels based on score comparisons."""
    import json

    # Group entries by (question_id, judge_tuple)
    grouped = {}
    with open(MT_BENCH_JUDGEMENTS_PATH, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            key = (entry['question_id'], tuple(entry['judge']))
            if key not in grouped:
                grouped[key] = {}
            grouped[key][entry['model']] = entry['score']

    # Build records where both models are present
    records = []
    for (question_id, judge_tuple), models in grouped.items():
        if STRONG_MODEL_COL in models and WEAK_MODEL_COL in models:
            strong_score = models[STRONG_MODEL_COL]
            weak_score = models[WEAK_MODEL_COL]

            # Create boolean-like columns based on score comparison
            # For router labels: strong "wins" if strong_score > weak_score
            records.append({
                STRONG_MODEL_COL: strong_score > weak_score,
                WEAK_MODEL_COL: weak_score > strong_score,
                'question_id': question_id,
                'judge': list(judge_tuple)
            })

    return pd.DataFrame(records)


def add_router_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate dataframe with router targets (winner_strong, tie, winner_weak)."""
    strong_correct = df[STRONG_MODEL_COL]
    weak_correct = df[WEAK_MODEL_COL]

    df["winner_strong"] = strong_correct & ~weak_correct
    df["winner_weak"] = weak_correct & ~strong_correct
    df["tie"] = ~(df["winner_strong"] | df["winner_weak"])

    label_map = {
        "winner_strong": df["winner_strong"],
        "winner_weak": df["winner_weak"],
    }
    df["router_label"] = pd.Series("tie", index=df.index)
    for label, mask in label_map.items():
        df.loc[mask, "router_label"] = label

    label_to_id = {"winner_strong": 0, "tie": 1, "winner_weak": 2}
    df["router_label_id"] = df["router_label"].map(label_to_id)
    return df


def print_label_distribution(df: pd.DataFrame, dataset_name: str) -> None:
    """Print relative distribution of router labels (percent share)."""
    distribution = df["router_label"].value_counts(normalize=True).sort_index()
    distribution = (distribution * 100).round(2)
    print(f"{dataset_name} router label distribution (percent):")
    for label, percent in distribution.items():
        print(f"  {label}: {percent:.2f}%")


# %%
gsm8k_df = load_gsm8k_responses()
gsm8k_df = add_router_labels(gsm8k_df)
print_label_distribution(gsm8k_df, "GSM8K")

# %%
mmlu_df = load_mmlu_responses()
mmlu_df = add_router_labels(mmlu_df)
print_label_distribution(mmlu_df, "MMLU")

# %%
mt_bench_df = load_mt_bench_responses()
mt_bench_df = add_router_labels(mt_bench_df)
print_label_distribution(mt_bench_df, "MT-bench")
