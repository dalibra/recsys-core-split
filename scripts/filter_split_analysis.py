"""Analysis driver for the filter/split-order paper.

The script intentionally recomputes the paper-facing summaries from saved
splits/results instead of relying on earlier notebook-level aggregates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
SPLIT_ROOT = DATA / "splitted" / "global_timesplit" / "val_last_train_item"
RESULTS_ROOT = DATA / "results" / "global_timesplit" / "val_last_train_item"
ANALYSIS_ROOT = ROOT / "results"
PAPER_ROOT = ROOT / "paper"
PAPER_FIG_DIR = PAPER_ROOT / "figures"
PAPER_TABLE_DIR = PAPER_ROOT / "tables"

MODELS = [
    "SASRec",
    "GRU4Rec",
    "BERT4Rec",
    "MostPopular",
    "Markov",
    "SKNN",
    "ItemKNN",
    "EASER",
]
NEURAL_MODELS = {"SASRec", "GRU4Rec", "BERT4Rec"}
DETERMINISTIC_MODELS = {"MostPopular", "Markov", "SKNN", "ItemKNN", "EASER"}
DATASETS = ["Beauty", "BeerAdvocate", "Diginetica", "Movielens-1m", "Sports", "YooChoose"]
Q_TAG = "q09"
PRIMARY_SEED = 17
GUARDRAIL_MIN_TEST_INTERACTIONS = 500
GUARDRAIL_MIN_TEST_USERS = 100
METRIC = "NDCG"
K = 20


DISPLAY_DATASET = {
    "Movielens-1m": "MovieLens-1M",
    "BeerAdvocate": "BeerAdv.",
    "Diginetica": "Diginetica",
    "YooChoose": "YooChoose",
}


sys.path.insert(0, str(EXPS / "runs"))
sys.path.insert(0, str(EXPS))
from variant_pipeline import (  # noqa: E402
    FilterSpec,
    apply_filter,
    base_preprocess,
    compute_time_threshold,
    filter_short_sequences,
    load_dataset_config,
    load_raw_data,
    split_global_time_fixed,
)


def ensure_dirs() -> None:
    for path in [ANALYSIS_ROOT, PAPER_FIG_DIR, PAPER_TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_variant(variant: str) -> Dict[str, str]:
    parts = variant.split("__")
    if len(parts) != 3:
        raise ValueError(f"Unexpected variant name: {variant}")
    return {"dataset": parts[0], "filter_tag": parts[1], "order_tag": parts[2]}


def order_short(order_tag: str) -> str:
    if order_tag == "v1_filter_then_split":
        return "V1"
    if order_tag == "v2_split_then_filter_train":
        return "V2"
    return order_tag


def filter_sort_key(filter_tag: str) -> Tuple[int, int, int]:
    if filter_tag.startswith("pcore_p"):
        return (0, int(filter_tag.replace("pcore_p", "")), 0)
    m = re.match(r"mincount_u(\d+)_i(\d+)", filter_tag)
    if m:
        return (1, int(m.group(1)), int(m.group(2)))
    return (9, 0, 0)


def q_to_float(q_tag: str) -> float:
    return float("0." + q_tag[2:])


def metric_from_file(path: Path, metric: str = METRIC, k: int = K) -> float:
    df = pd.read_csv(path)
    suffix = f"{metric}@{k}"
    if "metric_name" in df.columns and "metric_value" in df.columns:
        hit = df[df["metric_name"].astype(str).str.endswith(suffix)]
        if hit.empty:
            raise ValueError(f"Missing {suffix} in {path}")
        return float(hit["metric_value"].iloc[0])
    df = pd.read_csv(path, index_col=0)
    col = f"@{k}"
    if metric not in df.index or col not in df.columns:
        raise ValueError(f"Missing {metric} {col} in {path}")
    return float(df.loc[metric, col])


def choose_metric_file(model_dir: Path, model: str) -> Optional[Path]:
    if not model_dir.exists():
        return None
    files = sorted(p for p in model_dir.iterdir() if p.suffix == ".csv")
    if not files:
        return None
    if model in NEURAL_MODELS:
        primary = [p for p in files if p.name.endswith(f"_{PRIMARY_SEED}.csv")]
        if primary:
            return primary[0]
    scored: List[Tuple[float, Path]] = []
    for path in files:
        try:
            scored.append((metric_from_file(path), path))
        except Exception:
            continue
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def rank_scores(scores: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {model: idx + 1 for idx, (model, _) in enumerate(ordered)}


def kendall_tau(rank_a: Dict[str, int], rank_b: Dict[str, int], models: Sequence[str]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            da = rank_a[models[i]] - rank_a[models[j]]
            db = rank_b[models[i]] - rank_b[models[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    return float("nan") if total == 0 else (concordant - discordant) / total


def pairwise_flip_rate(rank_a: Dict[str, int], rank_b: Dict[str, int], models: Sequence[str]) -> float:
    flips = 0
    total = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            da = rank_a[models[i]] - rank_a[models[j]]
            db = rank_b[models[i]] - rank_b[models[j]]
            if da == 0 or db == 0:
                continue
            total += 1
            if da * db < 0:
                flips += 1
    return float("nan") if total == 0 else flips / total


def collect_leaderboards() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for variant_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        variant = variant_dir.name
        parsed = parse_variant(variant)
        q_dir = variant_dir / Q_TAG
        if not q_dir.exists():
            continue
        scores: Dict[str, float] = {}
        files: Dict[str, str] = {}
        for model in MODELS:
            metric_file = choose_metric_file(q_dir / model / "test_last", model)
            if metric_file is None:
                continue
            try:
                scores[model] = metric_from_file(metric_file)
                files[model] = str(metric_file)
            except Exception:
                continue
        if not scores:
            continue
        ranks = rank_scores(scores)
        for model, score in scores.items():
            rows.append(
                {
                    "dataset": parsed["dataset"],
                    "variant": variant,
                    "filter_tag": parsed["filter_tag"],
                    "order_tag": parsed["order_tag"],
                    "order": order_short(parsed["order_tag"]),
                    "model": model,
                    "NDCG@20": score,
                    "rank": ranks[model],
                    "metric_file": files[model],
                    "seed_policy": "seed17" if model in NEURAL_MODELS else "deterministic",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_ROOT / "leaderboards_seed17_policy.csv", index=False)
    return df


def load_variant_stats() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for variant_dir in sorted(p for p in SPLIT_ROOT.iterdir() if p.is_dir()):
        variant = variant_dir.name
        parsed = parse_variant(variant)
        stats_path = variant_dir / Q_TAG / "variant_stats.json"
        meta_path = variant_dir / Q_TAG / "variant_meta.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        row: Dict[str, object] = {
            "variant": variant,
            **parsed,
            "order": order_short(parsed["order_tag"]),
            "test_users": stats.get("test", {}).get("n_users", 0),
            "test_items": stats.get("test", {}).get("n_items", 0),
            "test_interactions": stats.get("test", {}).get("n_interactions", 0),
            "train_users": stats.get("train", {}).get("n_users", 0),
            "train_items": stats.get("train", {}).get("n_items", 0),
            "train_interactions": stats.get("train", {}).get("n_interactions", 0),
            "time_threshold": meta.get("time_threshold", np.nan),
            "train_core_violation_users_pct": meta.get("train_core_violation_users_pct", 0.0),
            "train_core_violation_items_pct": meta.get("train_core_violation_items_pct", 0.0),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["passes_guardrail"] = (
        (df["test_interactions"] >= GUARDRAIL_MIN_TEST_INTERACTIONS)
        & (df["test_users"] >= GUARDRAIL_MIN_TEST_USERS)
    )
    df.to_csv(ANALYSIS_ROOT / "variant_stats_compact.csv", index=False)
    return df


def matched_rows(leaderboards: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    stat_by_variant = stats.set_index("variant").to_dict("index")
    rows: List[Dict[str, object]] = []
    grouped = leaderboards.groupby(["dataset", "filter_tag"], sort=True)
    for (dataset, filter_tag), sub in grouped:
        variants_by_order = {
            order: order_sub["variant"].iloc[0]
            for order, order_sub in sub.groupby("order", sort=False)
        }
        if "V1" not in variants_by_order or "V2" not in variants_by_order:
            continue
        v1 = variants_by_order["V1"]
        v2 = variants_by_order["V2"]
        scores1 = {
            r["model"]: float(r["NDCG@20"])
            for _, r in sub[sub["variant"] == v1].iterrows()
        }
        scores2 = {
            r["model"]: float(r["NDCG@20"])
            for _, r in sub[sub["variant"] == v2].iterrows()
        }
        common_models = [m for m in MODELS if m in scores1 and m in scores2]
        if len(common_models) < 2:
            continue
        s1 = {m: scores1[m] for m in common_models}
        s2 = {m: scores2[m] for m in common_models}
        ranks1 = rank_scores(s1)
        ranks2 = rank_scores(s2)
        winner1 = min(ranks1, key=ranks1.get)
        winner2 = min(ranks2, key=ranks2.get)
        runner1 = sorted(ranks1, key=ranks1.get)[1]
        runner2 = sorted(ranks2, key=ranks2.get)[1]
        st1 = stat_by_variant.get(v1, {})
        st2 = stat_by_variant.get(v2, {})
        pass_guardrail = bool(st1.get("passes_guardrail", False) and st2.get("passes_guardrail", False))
        det_pair_flips = 0
        det_pair_total = 0
        deterministic_common = [m for m in common_models if m in DETERMINISTIC_MODELS]
        for i in range(len(deterministic_common)):
            for j in range(i + 1, len(deterministic_common)):
                a = deterministic_common[i]
                b = deterministic_common[j]
                da = ranks1[a] - ranks1[b]
                db = ranks2[a] - ranks2[b]
                if da == 0 or db == 0:
                    continue
                det_pair_total += 1
                det_pair_flips += int(da * db < 0)
        rows.append(
            {
                "dataset": dataset,
                "filter_tag": filter_tag,
                "filter_family": "pcore" if filter_tag.startswith("pcore_p") else "mincount",
                "v1_variant": v1,
                "v2_variant": v2,
                "matched_models": len(common_models),
                "v1_winner": winner1,
                "v2_winner": winner2,
                "v1_runner_up": runner1,
                "v2_runner_up": runner2,
                "winner_flip": winner1 != winner2,
                "winner_flip_has_deterministic": (
                    winner1 != winner2
                    and (winner1 in DETERMINISTIC_MODELS or winner2 in DETERMINISTIC_MODELS)
                ),
                "winner_flip_both_deterministic": (
                    winner1 != winner2
                    and winner1 in DETERMINISTIC_MODELS
                    and winner2 in DETERMINISTIC_MODELS
                ),
                "kendall_tau": kendall_tau(ranks1, ranks2, common_models),
                "pairwise_flip_rate": pairwise_flip_rate(ranks1, ranks2, common_models),
                "deterministic_pair_flips": det_pair_flips,
                "deterministic_pair_total": det_pair_total,
                "passes_guardrail": pass_guardrail,
                "v1_test_users": st1.get("test_users", np.nan),
                "v2_test_users": st2.get("test_users", np.nan),
                "v1_test_interactions": st1.get("test_interactions", np.nan),
                "v2_test_interactions": st2.get("test_interactions", np.nan),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset", "filter_family", "filter_tag"], key=lambda col: col)
    out.to_csv(ANALYSIS_ROOT / "matched_v1v2_pairs.csv", index=False)
    return out


def summarize_matched(
    matched: pd.DataFrame,
    stats: pd.DataFrame,
    family: Optional[str],
    exclude_p0: bool = False,
) -> pd.DataFrame:
    df = matched.copy()
    if family:
        df = df[df["filter_family"] == family]
    if exclude_p0:
        df = df[df["filter_tag"] != "pcore_p0"]
    rows: List[Dict[str, object]] = []
    for dataset, sub in df.groupby("dataset", sort=True):
        guarded = sub[sub["passes_guardrail"]]
        pcore_stats = stats[(stats["dataset"] == dataset) & (stats["filter_tag"].str.startswith("pcore_p"))]
        if exclude_p0:
            pcore_stats = pcore_stats[pcore_stats["filter_tag"] != "pcore_p0"]
        v1_pcore = pcore_stats[pcore_stats["order"] == "V1"]
        rows.append(
            {
                "dataset": dataset,
                "matched_pairs": int(len(sub)),
                "guardrailed_pairs": int(len(guarded)),
                "winner_flips": int(sub["winner_flip"].sum()),
                "guardrailed_winner_flips": int(guarded["winner_flip"].sum()),
                "deterministic_winner_flips": int(sub["winner_flip_has_deterministic"].sum()),
                "both_deterministic_winner_flips": int(sub["winner_flip_both_deterministic"].sum()),
                "deterministic_pair_flips": int(sub["deterministic_pair_flips"].sum()),
                "deterministic_pair_total": int(sub["deterministic_pair_total"].sum()),
                "mean_kendall_tau": float(sub["kendall_tau"].mean()) if len(sub) else np.nan,
                "guardrailed_mean_kendall_tau": float(guarded["kendall_tau"].mean()) if len(guarded) else np.nan,
                "max_user_violation_pct": float(v1_pcore["train_core_violation_users_pct"].max())
                if len(v1_pcore)
                else np.nan,
                "max_item_violation_pct": float(v1_pcore["train_core_violation_items_pct"].max())
                if len(v1_pcore)
                else np.nan,
                "median_test_users": float(
                    pd.concat([sub["v1_test_users"], sub["v2_test_users"]]).dropna().median()
                )
                if len(sub)
                else np.nan,
            }
        )
    suffix = f"_{family}" if family else "_all"
    if exclude_p0:
        suffix += "_pgt0"
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_ROOT / f"matched_summary{suffix}.csv", index=False)
    return out


def spec_from_filter_tag(filter_tag: str) -> FilterSpec:
    if filter_tag.startswith("pcore_p"):
        return FilterSpec(filter_type="pcore", p=int(filter_tag.replace("pcore_p", "")))
    m = re.match(r"mincount_u(\d+)_i(\d+)", filter_tag)
    if not m:
        raise ValueError(f"Unsupported filter tag: {filter_tag}")
    return FilterSpec(filter_type="mincount", u_min=int(m.group(1)), i_min=int(m.group(2)))


def build_raw_variant(
    base: pd.DataFrame,
    spec: FilterSpec,
    order: str,
    time_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if order == "filter_then_split":
        filtered = apply_filter(base, spec)
        train, validation, test, _ = split_global_time_fixed(
            filtered,
            time_threshold=time_threshold,
            validation_type="last_train_item",
            validation_quantile=0.9,
            validation_size=1024,
            random_state=17,
        )
        test_raw_before_projection = test.copy()
    elif order == "split_then_filter_train":
        train_raw, validation_raw, test_raw, _ = split_global_time_fixed(
            base,
            time_threshold=time_threshold,
            validation_type="last_train_item",
            validation_quantile=0.9,
            validation_size=1024,
            random_state=17,
        )
        train = apply_filter(train_raw, spec)
        train_users = set(train["user_id"].unique())
        train_items = set(train["item_id"].unique())
        validation = validation_raw[
            validation_raw["user_id"].isin(train_users)
            & validation_raw["item_id"].isin(train_items)
        ]
        test_raw_before_projection = test_raw.copy()
        test = test_raw[
            test_raw["user_id"].isin(train_users) & test_raw["item_id"].isin(train_items)
        ]
        validation = filter_short_sequences(validation, min_len=2)
        test = filter_short_sequences(test, min_len=2)
    else:
        raise ValueError(order)
    train = filter_short_sequences(train, min_len=2)
    return train.copy(), validation.copy(), test.copy(), test_raw_before_projection.copy()


def event_set(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    tmp = df[["user_id", "item_id", "timestamp"]].copy()
    return set(map(tuple, tmp.itertuples(index=False, name=None)))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def last_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "timestamp"])
    ordered = df.sort_values(["user_id", "timestamp"], kind="stable")
    return ordered.groupby("user_id", as_index=False).tail(1)[["user_id", "item_id", "timestamp"]]


def raw_diagnostics() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        cfg = load_dataset_config(dataset)
        raw = load_raw_data(cfg, str(DATA))
        base = base_preprocess(raw, drop_conseq_repeats=True)
        threshold = compute_time_threshold(
            base,
            0.9,
            cache_dir=str(DATA / "variants" / "time_thresholds"),
            dataset_name=cfg.name,
        )
        for pkl in (SPLIT_ROOT / f"{dataset}__pcore_p0__v1_filter_then_split" / Q_TAG).glob("time_threshold.pkl"):
            saved = pickle.load(open(pkl, "rb"))
            audit_rows.append(
                {
                    "dataset": dataset,
                    "computed_raw_q09": threshold,
                    "saved_variant_threshold": saved,
                    "abs_diff": abs(float(threshold) - float(saved)),
                }
            )
        for p in [0, 5, 10, 20]:
            spec = FilterSpec(filter_type="pcore", p=p)
            built: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
            for order in ["filter_then_split", "split_then_filter_train"]:
                built[order] = build_raw_variant(base, spec, order, threshold)
            train1, _, test1, preproj1 = built["filter_then_split"]
            train2, _, test2, preproj2 = built["split_then_filter_train"]

            for order_name, train, test, preproj in [
                ("V1", train1, test1, preproj1),
                ("V2", train2, test2, preproj2),
            ]:
                user_counts = train["user_id"].value_counts() if not train.empty else pd.Series(dtype=int)
                item_counts = train["item_id"].value_counts() if not train.empty else pd.Series(dtype=int)
                test_targets = last_targets(test)
                train_item_pop = train["item_id"].value_counts()
                target_pops = test_targets["item_id"].map(train_item_pop).dropna()
                pre_targets = last_targets(preproj)
                train_items = set(train["item_id"].unique())
                cold_rate = (
                    float((~pre_targets["item_id"].isin(train_items)).mean())
                    if len(pre_targets)
                    else np.nan
                )
                prefix_lengths = (
                    test[test["timestamp"] <= threshold].groupby("user_id").size()
                    if not test.empty
                    else pd.Series(dtype=int)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "p": p,
                        "order": order_name,
                        "train_users": int(train["user_id"].nunique()),
                        "train_items": int(train["item_id"].nunique()),
                        "train_interactions": int(len(train)),
                        "test_users": int(test["user_id"].nunique()),
                        "test_items": int(test["item_id"].nunique()),
                        "test_interactions": int(len(test)),
                        "candidate_items": int(train["item_id"].nunique()),
                        "train_user_violation_pct": float((user_counts < p).mean() * 100.0)
                        if len(user_counts)
                        else 0.0,
                        "train_item_violation_pct": float((item_counts < p).mean() * 100.0)
                        if len(item_counts)
                        else 0.0,
                        "test_users_prefix_lt_p_pct": float((prefix_lengths < p).mean() * 100.0)
                        if len(prefix_lengths)
                        else np.nan,
                        "median_eval_user_prefix_len": float(prefix_lengths.median())
                        if len(prefix_lengths)
                        else np.nan,
                        "median_train_pop_of_test_targets": float(target_pops.median())
                        if len(target_pops)
                        else np.nan,
                        "cold_target_rate_before_projection": cold_rate,
                    }
                )
            rows.append(
                {
                    "dataset": dataset,
                    "p": p,
                    "order": "V1_vs_V2",
                    "train_edge_jaccard": jaccard(event_set(train1), event_set(train2)),
                    "test_event_jaccard": jaccard(event_set(test1), event_set(test2)),
                    "test_user_jaccard": jaccard(set(test1["user_id"].unique()), set(test2["user_id"].unique())),
                    "candidate_item_jaccard": jaccard(set(train1["item_id"].unique()), set(train2["item_id"].unique())),
                }
            )
    diag = pd.DataFrame(rows)
    diag.to_csv(ANALYSIS_ROOT / "raw_pcore_diagnostics.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(ANALYSIS_ROOT / "time_threshold_audit.csv", index=False)
    return diag


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 17) -> Tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def per_user_file(variant: str, model: str, metric_file: Optional[str] = None) -> Optional[Path]:
    base = RESULTS_ROOT / variant / Q_TAG / model / "test_last_per_user"
    if not base.exists():
        return None
    if metric_file:
        candidate = base / Path(metric_file).name
        if candidate.exists():
            return candidate
    files = sorted(p for p in base.iterdir() if p.suffix == ".csv")
    if not files:
        return None
    if model in NEURAL_MODELS:
        primary = [p for p in files if p.name.endswith(f"_{PRIMARY_SEED}.csv")]
        return primary[0] if primary else files[0]
    return files[0]


def add_bootstrap_support(matched: pd.DataFrame, leaderboards: pd.DataFrame) -> pd.DataFrame:
    metric_file_lookup = {
        (r["variant"], r["model"]): r["metric_file"] for _, r in leaderboards.iterrows()
    }
    rows = []
    for _, row in matched.iterrows():
        status = dict(row)
        significant_sides = 0
        for side in ["v1", "v2"]:
            variant = str(row[f"{side}_variant"])
            winner = str(row[f"{side}_winner"])
            runner = str(row[f"{side}_runner_up"])
            w_file = per_user_file(variant, winner, metric_file_lookup.get((variant, winner)))
            r_file = per_user_file(variant, runner, metric_file_lookup.get((variant, runner)))
            if w_file is None or r_file is None:
                status[f"{side}_gap_mean"] = np.nan
                status[f"{side}_gap_ci_low"] = np.nan
                status[f"{side}_gap_ci_high"] = np.nan
                status[f"{side}_gap_supported"] = False
                continue
            w = pd.read_csv(w_file)[["user_id", "NDCG@20"]].rename(columns={"NDCG@20": "winner"})
            r = pd.read_csv(r_file)[["user_id", "NDCG@20"]].rename(columns={"NDCG@20": "runner"})
            merged = w.merge(r, on="user_id", how="inner")
            mean, low, high = bootstrap_ci((merged["winner"] - merged["runner"]).to_numpy())
            supported = bool(low > 0.0)
            significant_sides += int(supported)
            status[f"{side}_gap_mean"] = mean
            status[f"{side}_gap_ci_low"] = low
            status[f"{side}_gap_ci_high"] = high
            status[f"{side}_gap_supported"] = supported
        status["bootstrap_supported_flip"] = bool(row["winner_flip"] and significant_sides == 2)
        rows.append(status)
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_ROOT / "matched_v1v2_pairs_with_bootstrap.csv", index=False)
    return out


def write_latex_table(
    summary: pd.DataFrame,
    name: str,
    bootstrap: Optional[pd.DataFrame] = None,
    family: Optional[str] = None,
    exclude_p0: bool = False,
) -> None:
    supported_by_dataset: Dict[str, int] = {}
    if bootstrap is not None and len(bootstrap):
        boot = bootstrap.copy()
        if family:
            boot = boot[boot["filter_family"] == family]
        if exclude_p0:
            boot = boot[boot["filter_tag"] != "pcore_p0"]
        boot = boot[boot["passes_guardrail"]]
        supported_by_dataset = (
            boot.groupby("dataset")["bootstrap_supported_flip"].sum().astype(int).to_dict()
        )
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & Pairs & Elig. & Flip & Elig. flip & Boot. & $\bar{\tau}_g$ & U viol. & I viol. \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        dataset = DISPLAY_DATASET.get(str(r["dataset"]), str(r["dataset"]))
        sig = int(supported_by_dataset.get(str(r["dataset"]), 0))
        tau = r["guardrailed_mean_kendall_tau"]
        tau_s = "--" if pd.isna(tau) else f"{float(tau):.2f}"
        lines.append(
            f"{dataset} & {int(r['matched_pairs'])} & {int(r['guardrailed_pairs'])} & "
            f"{int(r['winner_flips'])} & {int(r['guardrailed_winner_flips'])} & "
            f"{sig} & {tau_s} & "
            f"{float(r['max_user_violation_pct']):.1f}\\% & "
            f"{float(r['max_item_violation_pct']):.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER_TABLE_DIR / name).write_text("\n".join(lines) + "\n")


def plot_mechanism_scatter(summary_pcore: pd.DataFrame) -> None:
    df = summary_pcore.copy()
    df["guarded_flip_rate"] = df["guardrailed_winner_flips"] / df["guardrailed_pairs"].replace(0, np.nan)
    plt.figure(figsize=(3.35, 2.35))
    colors = ["#1b9e77" if d == "Movielens-1m" else "#4c78a8" for d in df["dataset"]]
    plt.scatter(df["max_user_violation_pct"], df["guarded_flip_rate"], s=38, c=colors)
    for _, r in df.iterrows():
        label = {
            "Movielens-1m": "ML-1M",
            "BeerAdvocate": "Beer",
            "Diginetica": "Digi",
            "YooChoose": "Yoo",
        }.get(r["dataset"], r["dataset"])
        plt.annotate(label, (r["max_user_violation_pct"], r["guarded_flip_rate"]), xytext=(3, 2), textcoords="offset points", fontsize=7)
    plt.xlabel("max train-user violation (%)", fontsize=8)
    plt.ylabel("guardrailed top-1 flip rate", fontsize=8)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout(pad=0.2)
    plt.savefig(PAPER_FIG_DIR / "mechanism_scatter.pdf")
    plt.savefig(PAPER_FIG_DIR / "mechanism_scatter.png", dpi=300)
    plt.close()


def plot_supported_flip(bootstrap: pd.DataFrame) -> None:
    beauty = bootstrap[
        (bootstrap["dataset"] == "Beauty")
        & (bootstrap["filter_tag"] == "pcore_p10")
        & (bootstrap["bootstrap_supported_flip"])
    ]
    sports_path = ANALYSIS_ROOT / "sports_det_hp_pairwise.csv"
    if beauty.empty or not sports_path.exists():
        return
    sports = pd.read_csv(sports_path)
    sports = sports[(sports["metric"] == "NDCG@20") & (sports["comparison"] == "ItemKNN_minus_EASER")]
    if sports.empty:
        return

    b = beauty.iloc[0]
    # Matched bootstrap stores each arm as winner-minus-runner. For the figure,
    # use one signed convention across arms: SASRec-minus-Markov.
    b_means = np.array([float(b["v1_gap_mean"]), -float(b["v2_gap_mean"])])
    b_lows = np.array([float(b["v1_gap_ci_low"]), -float(b["v2_gap_ci_high"])])
    b_highs = np.array([float(b["v1_gap_ci_high"]), -float(b["v2_gap_ci_low"])])

    s_ordered = sports.set_index("order").loc[["V1", "V2"]]
    s_means = s_ordered["gap"].to_numpy(dtype=float)
    s_lows = s_ordered["ci_low"].to_numpy(dtype=float)
    s_highs = s_ordered["ci_high"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.05), sharey=False)
    colors = ["#4c78a8", "#f58518"]

    x = np.arange(2)
    axes[0].bar(x, b_means, color=colors, width=0.56)
    axes[0].errorbar(
        x,
        b_means,
        yerr=np.vstack([b_means - b_lows, b_highs - b_means]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
    )
    axes[0].axhline(0, color="#333333", linewidth=0.6)
    axes[0].set_xticks(x, ["V1", "V2"], fontsize=7)
    axes[0].set_title("Beauty fixed", fontsize=8, pad=3)
    axes[0].set_ylabel("NDCG@20 gap", fontsize=8)
    b_lim = float(max(abs(b_lows).max(), abs(b_highs).max()) * 1.15)
    axes[0].set_ylim(-b_lim, b_lim)

    axes[1].bar(x, s_means, color=colors, width=0.56)
    axes[1].errorbar(
        x,
        s_means,
        yerr=np.vstack([s_means - s_lows, s_highs - s_means]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
    )
    axes[1].axhline(0, color="#333333", linewidth=0.6)
    axes[1].set_xticks(x, ["V1", "V2"], fontsize=7)
    axes[1].set_title("Sports tuned\nI-E", fontsize=8, pad=1)
    lim = float(max(abs(s_lows).max(), abs(s_highs).max()) * 1.25)
    axes[1].set_ylim(-lim, lim)

    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    plt.savefig(PAPER_FIG_DIR / "supported_flip.pdf")
    plt.savefig(PAPER_FIG_DIR / "supported_flip.png", dpi=300)
    plt.close()


def plot_operator_schematic() -> None:
    fig, ax = plt.subplots(figsize=(3.35, 1.70))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.16", fc="white", ec="#333333", lw=0.8)
    wide_box = dict(boxstyle="round,pad=0.22", fc="white", ec="#333333", lw=0.8)
    arrow = dict(arrowstyle="->", color="#333333", lw=0.8)
    ax.text(0.50, 0.94, "V1: full-core -> split", ha="center", va="center", fontsize=8, weight="bold")
    ax.text(0.50, 0.46, "V2: split -> holdout -> fitted-core", ha="center", va="center", fontsize=8, weight="bold")
    ax.text(0.05, 0.74, r"$D$", bbox=box, ha="center", va="center")
    ax.text(0.33, 0.74, r"$C_p(D)$", bbox=box, ha="center", va="center")
    ax.text(0.69, 0.74, r"$S_{t^*}(C_p(D))$", bbox=box, ha="center", va="center")
    ax.annotate("", xy=(0.25, 0.74), xytext=(0.10, 0.74), arrowprops=arrow)
    ax.annotate("", xy=(0.55, 0.74), xytext=(0.42, 0.74), arrowprops=arrow)
    ax.text(0.05, 0.25, r"$D$", bbox=box, ha="center", va="center")
    ax.text(0.33, 0.25, r"$S_{t^*}(D)$", bbox=box, ha="center", va="center")
    ax.text(
        0.74,
        0.25,
        r"$C_p(D_{\mathrm{fit}})$" + "\n" + r"$\Pi(D_{\mathrm{held}})$",
        bbox=wide_box,
        ha="center",
        va="center",
        fontsize=7.7,
    )
    ax.annotate("", xy=(0.25, 0.25), xytext=(0.10, 0.25), arrowprops=arrow)
    ax.annotate("", xy=(0.54, 0.25), xytext=(0.42, 0.25), arrowprops=arrow)
    ax.text(0.97, 0.49, r"$\neq$", fontsize=13, ha="center", va="center")
    ax.text(0.50, 0.01, "Full-log filtering can use held-out support.", ha="center", va="bottom", fontsize=7)
    plt.tight_layout(pad=0.05)
    plt.savefig(PAPER_FIG_DIR / "operator_schematic.pdf")
    plt.savefig(PAPER_FIG_DIR / "operator_schematic.png", dpi=300)
    plt.close()


def write_summary(
    summary_all: pd.DataFrame,
    summary_pcore: pd.DataFrame,
    summary_pcore_pgt0: pd.DataFrame,
    raw_diag: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    all_pairs = int(summary_all["matched_pairs"].sum())
    all_flips = int(summary_all["winner_flips"].sum())
    all_guard_pairs = int(summary_all["guardrailed_pairs"].sum())
    all_guard_flips = int(summary_all["guardrailed_winner_flips"].sum())
    p_pairs = int(summary_pcore["matched_pairs"].sum())
    p_flips = int(summary_pcore["winner_flips"].sum())
    p_guard_pairs = int(summary_pcore["guardrailed_pairs"].sum())
    p_guard_flips = int(summary_pcore["guardrailed_winner_flips"].sum())
    pgt0_pairs = int(summary_pcore_pgt0["matched_pairs"].sum())
    pgt0_flips = int(summary_pcore_pgt0["winner_flips"].sum())
    pgt0_guard_pairs = int(summary_pcore_pgt0["guardrailed_pairs"].sum())
    pgt0_guard_flips = int(summary_pcore_pgt0["guardrailed_winner_flips"].sum())
    supported = bootstrap["bootstrap_supported_flip"].sum() if "bootstrap_supported_flip" in bootstrap else 0
    guard_supported = (
        bootstrap[bootstrap["passes_guardrail"]]["bootstrap_supported_flip"].sum()
        if "bootstrap_supported_flip" in bootstrap
        else 0
    )
    pcore_supported = (
        bootstrap[bootstrap["filter_family"] == "pcore"]["bootstrap_supported_flip"].sum()
        if "bootstrap_supported_flip" in bootstrap
        else 0
    )
    pcore_guard_supported = (
        bootstrap[(bootstrap["filter_family"] == "pcore") & (bootstrap["passes_guardrail"])][
            "bootstrap_supported_flip"
        ].sum()
        if "bootstrap_supported_flip" in bootstrap
        else 0
    )
    pgt0_guard_supported = (
        bootstrap[
            (bootstrap["filter_family"] == "pcore")
            & (bootstrap["filter_tag"] != "pcore_p0")
            & (bootstrap["passes_guardrail"])
        ]["bootstrap_supported_flip"].sum()
        if "bootstrap_supported_flip" in bootstrap
        else 0
    )
    unknown_boot = int(
        bootstrap[["v1_gap_mean", "v2_gap_mean"]].isna().any(axis=1).sum()
    ) if "v1_gap_mean" in bootstrap else len(bootstrap)

    max_viol = (
        raw_diag[raw_diag["order"] == "V1"]
        .sort_values("train_user_violation_pct", ascending=False)
        .head(6)[["dataset", "p", "train_user_violation_pct", "train_item_violation_pct"]]
    )
    max_viol_lines = [
        "| dataset | p | train user violation % | train item violation % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in max_viol.iterrows():
        max_viol_lines.append(
            f"| {r['dataset']} | {int(r['p'])} | "
            f"{float(r['train_user_violation_pct']):.2f} | "
            f"{float(r['train_item_violation_pct']):.2f} |"
        )
    lines = [
        "# Final analysis summary",
        "",
        "## Protocol audit",
        "",
        f"- Cutoff policy: one raw-log q=0.9 timestamp per dataset, reused for V1 and V2.",
        f"- Tie rule: train-side interactions use `timestamp <= t*`; evaluation users have at least one interaction with `timestamp > t*`.",
        f"- Test-size guardrail: at least {GUARDRAIL_MIN_TEST_INTERACTIONS} test interactions and {GUARDRAIL_MIN_TEST_USERS} test users on both sides of a matched V1/V2 pair.",
        f"- Neural leaderboard policy: primary seed {PRIMARY_SEED}; deterministic baselines are seed-free.",
        "",
        "## Matched V1/V2 results",
        "",
        f"- All filter settings: {all_flips}/{all_pairs} top-1 flips; after guardrail {all_guard_flips}/{all_guard_pairs}.",
        f"- P-core settings including the p=0 projection control: {p_flips}/{p_pairs} top-1 flips; after guardrail {p_guard_flips}/{p_guard_pairs}.",
        f"- P-core settings with p>0: {pgt0_flips}/{pgt0_pairs} nominal top-1 flips; after guardrail {pgt0_guard_flips}/{pgt0_guard_pairs}; bootstrap-supported guardrailed p>0 flips: {int(pgt0_guard_supported)}.",
        f"- Winner flips involving at least one deterministic baseline: {int(summary_all['deterministic_winner_flips'].sum())}/{all_flips}.",
        f"- Deterministic pairwise order reversals: {int(summary_all['deterministic_pair_flips'].sum())}/{int(summary_all['deterministic_pair_total'].sum())}.",
        f"- Bootstrap-supported flips: {int(supported)} overall, {int(guard_supported)} after guardrail, {int(pcore_supported)} for p-core, {int(pcore_guard_supported)} for guardrailed p-core; missing per-user gap files for {unknown_boot} matched pairs.",
        "",
        "## Strongest fitted-train core violations",
        "",
        "\n".join(max_viol_lines),
        "",
        "## Output files",
        "",
        "- `results/matched_v1v2_pairs.csv`",
        "- `results/matched_summary_all.csv`",
        "- `results/matched_summary_pcore.csv`",
        "- `results/raw_pcore_diagnostics.csv`",
        "- `paper/tables/matched_summary_pcore.tex`",
        "- `paper/figures/operator_schematic.pdf`",
        "- `paper/figures/mechanism_scatter.pdf`",
    ]
    (ANALYSIS_ROOT / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw-diagnostics", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    leaderboards = collect_leaderboards()
    stats = load_variant_stats()
    matched = matched_rows(leaderboards, stats)
    summary_all = summarize_matched(matched, stats, family=None)
    summary_pcore = summarize_matched(matched, stats, family="pcore")
    summary_pcore_pgt0 = summarize_matched(matched, stats, family="pcore", exclude_p0=True)
    raw_diag = pd.read_csv(ANALYSIS_ROOT / "raw_pcore_diagnostics.csv") if args.skip_raw_diagnostics else raw_diagnostics()
    bootstrap = add_bootstrap_support(matched, leaderboards)
    write_latex_table(summary_all, "matched_summary_all.tex", bootstrap=bootstrap, family=None)
    write_latex_table(summary_pcore, "matched_summary_pcore.tex", bootstrap=bootstrap, family="pcore")
    write_latex_table(
        summary_pcore_pgt0,
        "matched_summary_pcore_pgt0.tex",
        bootstrap=bootstrap,
        family="pcore",
        exclude_p0=True,
    )
    plot_operator_schematic()
    plot_mechanism_scatter(summary_pcore_pgt0)
    plot_supported_flip(bootstrap)
    write_summary(summary_all, summary_pcore, summary_pcore_pgt0, raw_diag, bootstrap)
    print(f"Saved analysis outputs to {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
