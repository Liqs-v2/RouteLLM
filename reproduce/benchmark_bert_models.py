#!/usr/bin/env python3
"""
Benchmark multiple BERT router checkpoints on the RouteLLM evaluation suite.

This script runs the official RouteLLM benchmark pipeline three times –
once per checkpoint – using the pre-computed GPT-4 vs Mixtral evaluation
results shipped with the repository. Each run writes its plots and metrics
to an isolated output directory so that results can be compared later on.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from transformers import AutoTokenizer

from routellm.controller import Controller, ModelPair
from routellm.evals.benchmarks import GSM8K, MMLU, MTBench
from routellm.evals.evaluate import generate_results
from routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "bert_custom": {
        "path": "/s3/lindenbauer/RouteLLM/models/d_arena-cls-balanced_data-bs128/checkpoint-1900/",
        "label": "Custom (checkpoint-1900)",
    },
    "bert_augmented": {
        "path": "routellm/bert_gpt4_augmented",
        "label": "Authors Augmented",
    },
    "bert_base": {
        "path": "routellm/bert",
        "label": "Authors Base",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple BERT router checkpoints on RouteLLM benchmarks.",
    )
    parser.add_argument(
        "--custom-checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINTS["bert_custom"]["path"],
        help="Filesystem path (or HF repo id) for the custom checkpoint.",
    )
    parser.add_argument(
        "--augmented-checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINTS["bert_augmented"]["path"],
        help="HF repo id for the authors' GPT-4 augmented checkpoint.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINTS["bert_base"]["path"],
        help="HF repo id for the authors' base checkpoint.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        default=["mmlu", "gsm8k", "mt-bench"],
        choices=["mmlu", "gsm8k", "mt-bench"],
        help="Benchmarks to evaluate.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="reproduce/benchmark_results",
        help="Directory to store per-checkpoint benchmark outputs.",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="Number of threshold splits to evaluate (same as --num-results in evaluate.py).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of CPU cores for pandarallel. Defaults to four physical cores (4). Use 0 for all cores.",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        default="gpt-4-1106-preview",
        help="Strong model name – must match the benchmark CSV column.",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Weak model name – must match the benchmark CSV column.",
    )
    parser.add_argument(
        "--bert-tokenizer",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Tokenizer identifier to use for the BERTRouter (defaults to the base model).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip evaluations when the destination directory already contains results.",
    )
    parser.add_argument(
        "--overwrite-cache",
        dest="overwrite_cache",
        action="store_true",
        help="Force regeneration of router caches for each benchmark.",
    )
    parser.add_argument(
        "--no-overwrite-cache",
        dest="overwrite_cache",
        action="store_false",
        help="Reuse cached strong win rates if present.",
    )
    parser.set_defaults(overwrite_cache=True)
    return parser.parse_args()


def resolve_checkpoints(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    checkpoints = {
        "bert_custom": {
            "path": args.custom_checkpoint,
            "label": "Custom (checkpoint-1900)",
        },
        "bert_augmented": {
            "path": args.augmented_checkpoint,
            "label": "Authors Augmented",
        },
        "bert_base": {"path": args.base_checkpoint, "label": "Authors Base"},
    }
    for name, cfg in checkpoints.items():
        ckpt_path = cfg["path"]
        if os.path.exists(ckpt_path):
            continue
        # Skip existence checks for likely remote identifiers (e.g., Hugging Face repos)
        if ckpt_path.startswith("routellm/") or "hf://" in ckpt_path:
            continue
        if ckpt_path.startswith("http://") or ckpt_path.startswith("https://"):
            continue
        # Only warn – transformers will surface an error if the checkpoint is invalid.
        print(f"[WARN] Checkpoint path '{ckpt_path}' does not exist on disk.")
    return checkpoints


def create_benchmark(
    benchmark_name: str, model_pair: ModelPair, overwrite_cache: bool
):
    overwrite_list = ["bert"] if overwrite_cache else []
    if benchmark_name == "mmlu":
        print("Loading MMLU benchmark...")
        return MMLU(ALL_MMLU_DOMAINS, model_pair, overwrite_list)
    if benchmark_name == "gsm8k":
        print("Loading GSM8K benchmark...")
        return GSM8K(model_pair, overwrite_list)
    if benchmark_name == "mt-bench":
        print("Loading MT Bench benchmark...")
        return MTBench(model_pair, overwrite_list)
    raise ValueError(f"Unsupported benchmark '{benchmark_name}'")


def pretty_print_thresholds(
    label: str,
    benchmark_name: str,
    threshold: float,
    accuracy: float,
    model_counts: Dict[str, int],
    total: int,
) -> None:
    header = (
        "=" * 15
        + f" {label} with threshold {threshold:.4f} on {benchmark_name} "
        + "=" * 15
    )
    print("\n" + header)
    print(f"Average accuracy: {accuracy:.3f}")
    print(
        "Model counts: "
        + ", ".join([f"{name}: {count}" for name, count in model_counts.items()])
    )
    print(
        "Model %: "
        + ", ".join(
            [
                f"{name}: {count / total * 100:.3f}%"
                for name, count in model_counts.items()
            ]
        )
    )
    print("=" * len(header) + "\n")


def evaluate_single_benchmark(
    controller: Controller,
    benchmark_name: str,
    benchmark_output_dir: Path,
    checkpoint_label: str,
    num_results: int,
    overwrite_cache: bool,
) -> pd.DataFrame:
    benchmark = create_benchmark(
        benchmark_name, controller.model_pair, overwrite_cache=overwrite_cache
    )

    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    results_path = benchmark_output_dir / "results.csv"
    metrics_path = benchmark_output_dir / "metrics.json"

    all_results = pd.DataFrame()
    for router_name in controller.routers:
        router_results: List[Dict[str, float]] = []
        for (
            threshold,
            accuracy,
            model_counts,
            total,
        ) in benchmark.evaluate(controller, router_name, num_results, False):
            model_counts = {
                controller.model_pair.strong: model_counts.get(
                    controller.model_pair.strong, 0
                ),
                controller.model_pair.weak: model_counts.get(
                    controller.model_pair.weak, 0
                ),
            }
            pretty_print_thresholds(
                checkpoint_label, benchmark_name, threshold, accuracy, model_counts, total
            )
            router_results.append(
                {
                    "method": checkpoint_label,
                    "router": router_name,
                    "threshold": threshold,
                    "strong_percentage": model_counts[controller.model_pair.strong]
                    / total
                    * 100,
                    "accuracy": accuracy,
                }
            )
        all_results = pd.concat(
            [all_results, pd.DataFrame(router_results)], ignore_index=True
        )

    if all_results.empty:
        raise RuntimeError(
            f"No evaluation results were produced for {checkpoint_label} on {benchmark_name}."
        )

    generate_results(
        all_results,
        benchmark,
        benchmark_name,
        controller.model_pair,
        str(benchmark_output_dir),
    )

    metrics_df = compute_metrics(all_results, benchmark, controller.model_pair)
    all_results.to_csv(results_path, index=False)
    metrics_path.write_text(metrics_df.to_json(orient="records", indent=2))

    return all_results


def compute_metrics(
    df_router_result: pd.DataFrame, benchmark, model_pair: ModelPair
) -> pd.DataFrame:
    weak_accuracy = benchmark.get_model_accuracy(model_pair.weak)
    strong_accuracy = benchmark.get_model_accuracy(model_pair.strong)

    def pct_call_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        pct_calls: List[str] = []
        for pct in [0.2, 0.5, 0.8]:
            pct_call = np.interp(
                pct * (strong_accuracy - weak_accuracy) + weak_accuracy,
                df_per_method["accuracy"],
                df_per_method["strong_percentage"],
            )
            pct_calls.append(f"{pct_call:.2f}%")
        return pd.Series(pct_calls)

    def auc_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        return float(
            np.trapz(df_per_method["accuracy"], df_per_method["strong_percentage"] / 100)
        )

    def apgr_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        weak_auc = np.zeros([len(df_per_method)], dtype=float)
        weak_auc.fill(weak_accuracy)
        weak_auc = np.trapz(weak_auc, df_per_method["strong_percentage"] / 100)
        strong_auc = np.zeros([len(df_per_method)], dtype=float)
        strong_auc.fill(strong_accuracy)
        strong_auc = np.trapz(strong_auc, df_per_method["strong_percentage"] / 100)
        return (row["AUC"] - weak_auc) / (strong_auc - weak_auc)

    metrics = pd.DataFrame({"method": df_router_result["method"].unique()})
    metrics[["20% qual", "50% qual", "80% qual"]] = metrics.apply(
        pct_call_metric, axis=1
    )
    metrics["AUC"] = metrics.apply(auc_metric, axis=1)
    metrics["APGR"] = metrics.apply(apgr_metric, axis=1)
    metrics = metrics.sort_values(by=["APGR"], ascending=False).reset_index(drop=True)
    return metrics


def build_comparison_outputs(
    checkpoints: Dict[str, Dict[str, str]],
    benchmarks: Iterable[str],
    output_base: Path,
    model_pair: ModelPair,
) -> None:
    comparison_dir = output_base / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for benchmark_name in benchmarks:
        combined_frames: List[pd.DataFrame] = []
        for checkpoint_name, cfg in checkpoints.items():
            results_path = output_base / checkpoint_name / benchmark_name / "results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                combined_frames.append(df)
        if not combined_frames:
            print(f"[WARN] Skipping comparison for {benchmark_name}; no results found.")
            continue

        combined_df = pd.concat(combined_frames, ignore_index=True)
        benchmark = create_benchmark(benchmark_name, model_pair, overwrite_cache=False)
        comparison_name = f"{benchmark_name}_comparison"
        generate_results(
            combined_df,
            benchmark,
            comparison_name,
            model_pair,
            str(comparison_dir),
        )
        metrics_df = compute_metrics(combined_df, benchmark, model_pair)
        combined_df.to_csv(
            comparison_dir / f"{comparison_name}_results.csv", index=False
        )
        metrics_df.to_csv(
            comparison_dir / f"{comparison_name}_metrics.csv", index=False
        )


def main():
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)


    if args.parallel and args.parallel > 0:
        pandarallel.initialize(progress_bar=True, nb_workers=args.parallel)
    else:
        pandarallel.initialize(progress_bar=True)

    for checkpoint_name, cfg in checkpoints.items():
        checkpoint_dir = output_base / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config = {"bert": {"checkpoint_path": cfg["path"], "tokenizer_path": args.bert_tokenizer}}
        controller = Controller(
            routers=["bert"],
            config=config,
            strong_model=args.strong_model,
            weak_model=args.weak_model,
            progress_bar=True,
        )


        print(f"\n===== Running benchmarks for {cfg['label']} ({cfg['path']}) =====")
        for benchmark_name in args.benchmarks:
            benchmark_output_dir = checkpoint_dir / benchmark_name
            results_file = benchmark_output_dir / "results.csv"
            if args.skip_existing and results_file.exists():
                print(
                    f"[INFO] Skipping {benchmark_name} for {cfg['label']} "
                    f"(results already exist)."
                )
                continue

            results_df = evaluate_single_benchmark(
                controller=controller,
                benchmark_name=benchmark_name,
                benchmark_output_dir=benchmark_output_dir,
                checkpoint_label=cfg["label"],
                num_results=args.num_results,
                overwrite_cache=args.overwrite_cache,
            )

    build_comparison_outputs(
        checkpoints=checkpoints,
        benchmarks=args.benchmarks,
        output_base=output_base,
        model_pair=ModelPair(args.strong_model, args.weak_model),
    )


if __name__ == "__main__":
    main()

