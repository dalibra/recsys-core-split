"""
Seed/uncertainty analysis for sequential model results.
"""

import argparse
import math
import os
import re
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


SEED_PATTERN = re.compile(r"_(\d+)\.csv$")


def q_tag(quantile: float) -> str:
    return "q0" + str(quantile)[2:]


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def load_metric(file_path: str, metric: str, k: int) -> float:
    df = pd.read_csv(file_path)
    suffix = f"{metric}@{k}"
    if "metric_name" in df.columns and "metric_value" in df.columns:
        subset = df[df["metric_name"].str.endswith(suffix)]
        if subset.empty:
            raise ValueError(f"Missing {suffix} in {file_path}")
        return float(subset["metric_value"].iloc[0])
    df = pd.read_csv(file_path, index_col=0)
    col = f"@{k}"
    if metric not in df.index or col not in df.columns:
        raise ValueError(f"Missing {metric} or {col} in {file_path}")
    return float(df.loc[metric, col])


def rank_models(scores: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return {model: idx + 1 for idx, (model, _) in enumerate(ordered)}


def kendall_tau(rank_a: Dict[str, int], rank_b: Dict[str, int], models: Sequence[str]) -> float:
    concordant = 0
    discordant = 0
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            m1 = models[i]
            m2 = models[j]
            diff_a = rank_a[m1] - rank_a[m2]
            diff_b = rank_b[m1] - rank_b[m2]
            if diff_a == 0 or diff_b == 0:
                continue
            if diff_a * diff_b > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return (concordant - discordant) / float(total)


def flip_rate(rank_a: Dict[str, int], rank_b: Dict[str, int], models: Sequence[str]) -> float:
    flips = 0
    total = 0
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            m1 = models[i]
            m2 = models[j]
            diff_a = rank_a[m1] - rank_a[m2]
            diff_b = rank_b[m1] - rank_b[m2]
            if diff_a == 0 or diff_b == 0:
                continue
            total += 1
            if diff_a * diff_b < 0:
                flips += 1
    if total == 0:
        return float("nan")
    return flips / float(total)


def collect_rows(
    results_root: str,
    datasets: Sequence[str],
    models: Sequence[str],
    metric: str,
    k: int,
    variant_prefix: str,
) -> pd.DataFrame:
    rows = []
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Missing results root: {results_root}")

    for variant in sorted(os.listdir(results_root)):
        if not os.path.isdir(os.path.join(results_root, variant)):
            continue
        dataset = variant.split("__")[0]
        if datasets and dataset not in datasets:
            continue
        if variant_prefix and not variant.startswith(variant_prefix):
            continue
        for model in models:
            model_dir = os.path.join(results_root, variant, model, "test_last")
            if not os.path.isdir(model_dir):
                continue
            for filename in sorted(os.listdir(model_dir)):
                if not filename.endswith(".csv"):
                    continue
                match = SEED_PATTERN.search(filename)
                if not match:
                    continue
                seed = int(match.group(1))
                file_path = os.path.join(model_dir, filename)
                try:
                    value = load_metric(file_path, metric, k)
                except Exception:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "model": model,
                        "seed": seed,
                        f"{metric}@{k}": value,
                        "file": file_path,
                    }
                )
    return pd.DataFrame(rows)


def summarize_seed_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = []
    grouped = df.groupby(["dataset", "variant", "model"], sort=True)
    for (dataset, variant, model), sub in grouped:
        vals = sub[score_col].to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals)) if n > 0 else float("nan")
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        se = std / math.sqrt(n) if n > 1 else float("nan")
        ci_delta = 1.96 * se if n > 1 else float("nan")
        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "model": model,
                "n_seeds": n,
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci_delta if n > 1 else float("nan"),
                "ci95_high": mean + ci_delta if n > 1 else float("nan"),
                "seeds": ",".join(str(x) for x in sorted(sub["seed"].unique())),
            }
        )
    return pd.DataFrame(out)


def seed_rank_stability(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = []
    for (dataset, variant), sub in df.groupby(["dataset", "variant"], sort=True):
        seed_scores: Dict[int, Dict[str, float]] = {}
        for seed, seed_df in sub.groupby("seed", sort=True):
            seed_scores[int(seed)] = {r["model"]: float(r[score_col]) for _, r in seed_df.iterrows()}
        seeds = sorted(seed_scores)
        if len(seeds) < 2:
            continue

        pair_rows = []
        for s1, s2 in combinations(seeds, 2):
            common_models = sorted(
                set(seed_scores[s1].keys()) & set(seed_scores[s2].keys())
            )
            if len(common_models) < 2:
                continue
            rank1 = rank_models(seed_scores[s1])
            rank2 = rank_models(seed_scores[s2])
            tau = kendall_tau(rank1, rank2, common_models)
            flip = flip_rate(rank1, rank2, common_models)
            top1_change = 1.0 if min(rank1, key=rank1.get) != min(rank2, key=rank2.get) else 0.0
            pair_rows.append((tau, flip, top1_change))

        if not pair_rows:
            continue
        tau_vals = np.array([x[0] for x in pair_rows], dtype=float)
        flip_vals = np.array([x[1] for x in pair_rows], dtype=float)
        top1_vals = np.array([x[2] for x in pair_rows], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed_pairs": len(pair_rows),
                "mean_kendall_tau": float(np.nanmean(tau_vals)),
                "mean_flip_rate": float(np.nanmean(flip_vals)),
                "top1_change_rate": float(np.nanmean(top1_vals)),
            }
        )
    return pd.DataFrame(out)


def effect_size_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), sub in summary.groupby(["dataset", "model"], sort=True):
        if sub.empty:
            continue
        mean_vals = sub["mean"].to_numpy(dtype=float)
        seed_std = sub["std"].dropna().to_numpy(dtype=float)
        preprocess_range = float(np.nanmax(mean_vals) - np.nanmin(mean_vals)) if len(mean_vals) else float("nan")
        median_seed_std = float(np.nanmedian(seed_std)) if len(seed_std) else float("nan")
        ratio = (
            preprocess_range / median_seed_std
            if (not np.isnan(median_seed_std) and median_seed_std > 0)
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "preprocess_range": preprocess_range,
                "median_seed_std": median_seed_std,
                "range_to_seed_std_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze seed uncertainty.")
    parser.add_argument("--data-path", default="exps/data")
    parser.add_argument("--split-subtype", default="val_last_train_item")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--models", default="SASRec,GRU4Rec,BERT4Rec")
    parser.add_argument("--metric", default="NDCG")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--variant-prefix", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    score_col = f"{args.metric}@{args.k}"
    q = q_tag(args.quantile)

    results_root = os.path.join(
        args.data_path, "results", "global_timesplit", args.split_subtype
    )
    results_root = os.path.join(results_root, "")
    q_root = {}
    for variant in os.listdir(results_root):
        q_root[variant] = os.path.join(results_root, variant, q)

    # Flatten to q-root by temporarily symlinking-like logic through explicit traversal.
    rows = []
    for variant, vroot in q_root.items():
        if not os.path.isdir(vroot):
            continue
        dataset = variant.split("__")[0]
        if datasets and dataset not in datasets:
            continue
        if args.variant_prefix and not variant.startswith(args.variant_prefix):
            continue
        for model in models:
            model_dir = os.path.join(vroot, model, "test_last")
            if not os.path.isdir(model_dir):
                continue
            for filename in sorted(os.listdir(model_dir)):
                if not filename.endswith(".csv"):
                    continue
                match = SEED_PATTERN.search(filename)
                if not match:
                    continue
                seed = int(match.group(1))
                file_path = os.path.join(model_dir, filename)
                try:
                    score = load_metric(file_path, args.metric, args.k)
                except Exception:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "model": model,
                        "seed": seed,
                        score_col: score,
                        "file": file_path,
                    }
                )

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise RuntimeError("No seed-resolved metrics found.")

    summary_df = summarize_seed_table(raw_df, score_col)
    rank_df = seed_rank_stability(raw_df, score_col)
    effect_df = effect_size_table(summary_df)

    output_dir = args.output_dir or os.path.join(
        args.data_path, "analysis", "uncertainty", q
    )
    os.makedirs(output_dir, exist_ok=True)
    raw_df.to_csv(os.path.join(output_dir, "seed_metrics_raw.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "seed_summary.csv"), index=False)
    rank_df.to_csv(os.path.join(output_dir, "seed_rank_stability.csv"), index=False)
    effect_df.to_csv(os.path.join(output_dir, "seed_effect_size.csv"), index=False)
    print(f"Saved seed uncertainty analysis to {output_dir}")


if __name__ == "__main__":
    main()
