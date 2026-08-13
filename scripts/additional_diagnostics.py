"""Additional diagnostics for the filter/split-order paper."""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
RESULTS_ROOT = DATA / "results" / "global_timesplit" / "val_last_train_item"
OUT = ROOT / "results"
Q_TAG = "q09"
PRIMARY_SEED = 17
DATASETS = ["Beauty", "BeerAdvocate", "Diginetica", "Movielens-1m", "Sports", "YooChoose"]
P_VALUES = [0, 5, 10, 20]

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
    split_by_time_threshold,
    split_validation_last_train,
)


def quiet_apply_filter(data: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return apply_filter(data, spec)


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 17) -> Tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_base_and_threshold(dataset: str) -> Tuple[pd.DataFrame, float]:
    cfg = load_dataset_config(dataset)
    raw = load_raw_data(cfg, str(DATA))
    base = base_preprocess(raw, drop_conseq_repeats=True)
    threshold = compute_time_threshold(
        base,
        0.9,
        cache_dir=str(DATA / "variants" / "time_thresholds"),
        dataset_name=cfg.name,
    )
    return base, threshold


def v1_preholdout_and_fit(dataset: str, p: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base, threshold = load_base_and_threshold(dataset)
    spec = FilterSpec(filter_type="pcore", p=p)
    filtered = quiet_apply_filter(base, spec)
    train_pre, _ = split_by_time_threshold(filtered, threshold)
    train_fit, _ = split_validation_last_train(train_pre)
    train_fit = filter_short_sequences(train_fit, min_len=2)
    return train_pre.copy(), train_fit.copy()


def decompose_counts(
    train_pre: pd.DataFrame,
    train_fit: pd.DataFrame,
    p: int,
    col: str,
) -> Dict[str, float]:
    fit_counts = train_fit[col].value_counts()
    pre_counts = train_pre[col].value_counts()
    viol = fit_counts[fit_counts < p]
    if viol.empty:
        return {
            "entities": int(len(fit_counts)),
            "violations": 0,
            "violation_pct": 0.0,
            "preholdout_below_p": 0,
            "holdout_only": 0,
            "preholdout_below_p_share_of_violations": 0.0,
            "holdout_only_share_of_violations": 0.0,
            "preholdout_below_p_pct_of_entities": 0.0,
            "holdout_only_pct_of_entities": 0.0,
        }
    pre_for_viol = pre_counts.reindex(viol.index).fillna(0)
    pre_below = int((pre_for_viol < p).sum())
    holdout_only = int((pre_for_viol >= p).sum())
    denom = int(len(viol))
    entities = int(len(fit_counts))
    return {
        "entities": entities,
        "violations": denom,
        "violation_pct": 100.0 * denom / entities if entities else 0.0,
        "preholdout_below_p": pre_below,
        "holdout_only": holdout_only,
        "preholdout_below_p_share_of_violations": 100.0 * pre_below / denom,
        "holdout_only_share_of_violations": 100.0 * holdout_only / denom,
        "preholdout_below_p_pct_of_entities": 100.0 * pre_below / entities if entities else 0.0,
        "holdout_only_pct_of_entities": 100.0 * holdout_only / entities if entities else 0.0,
    }


def violation_decomposition() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in ["Beauty", "Sports"]:
        train_pre, train_fit = v1_preholdout_and_fit(dataset, 5)
        for entity, col in [("users", "user_id"), ("items", "item_id")]:
            row = {
                "dataset": dataset,
                "p": 5,
                "entity": entity,
                "d_train_stage": "V1 fitted train after last-train validation holdout",
            }
            row.update(decompose_counts(train_pre, train_fit, 5, col))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "violation_decomposition_p5.csv", index=False)
    return out


def iterative_fitted_core() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for p in P_VALUES:
            train_pre, train_fit = v1_preholdout_and_fit(dataset, p)
            users = set(train_fit["user_id"].unique())
            items = set(train_fit["item_id"].unique())
            if p == 0 or train_fit.empty:
                core_fit = train_fit.copy()
            else:
                core_fit = quiet_apply_filter(train_fit, FilterSpec(filter_type="pcore", p=p))
            core_users = set(core_fit["user_id"].unique())
            core_items = set(core_fit["item_id"].unique())
            rows.append(
                {
                    "dataset": dataset,
                    "p": p,
                    "v1_train_users": len(users),
                    "v1_train_items": len(items),
                    "v1_train_interactions": int(len(train_fit)),
                    "preholdout_train_users": int(train_pre["user_id"].nunique()),
                    "preholdout_train_items": int(train_pre["item_id"].nunique()),
                    "refit_core_users_removed_pct": 100.0 * (len(users - core_users) / len(users)) if users else 0.0,
                    "refit_core_items_removed_pct": 100.0 * (len(items - core_items) / len(items)) if items else 0.0,
                    "refit_core_interactions_removed_pct": 100.0 * (1.0 - len(core_fit) / len(train_fit)) if len(train_fit) else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "iterative_fitted_core_diagnostics.csv", index=False)
    return out


def per_user_path(variant: str, model: str, metric_file: Optional[str]) -> Optional[Path]:
    base = RESULTS_ROOT / variant / Q_TAG / model / "test_last_per_user"
    if not base.exists():
        return None
    if metric_file:
        candidate = base / Path(metric_file).name
        if candidate.exists():
            return candidate
    files = sorted(base.glob("*.csv"))
    if not files:
        return None
    if model in {"SASRec", "GRU4Rec", "BERT4Rec"}:
        primary = [p for p in files if p.name.endswith(f"_{PRIMARY_SEED}.csv")]
        if primary:
            return primary[0]
    return files[0]


def beauty_secondary_metric(leaderboards: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    variant_by_order = {
        "V1": "Beauty__pcore_p10__v1_filter_then_split",
        "V2": "Beauty__pcore_p10__v2_split_then_filter_train",
    }
    metric_lookup = {
        (r["variant"], r["model"]): r.get("metric_file")
        for _, r in leaderboards.iterrows()
    }
    for order, variant in variant_by_order.items():
        sas_path = per_user_path(variant, "SASRec", metric_lookup.get((variant, "SASRec")))
        markov_path = per_user_path(variant, "Markov", metric_lookup.get((variant, "Markov")))
        if sas_path is None or markov_path is None:
            continue
        sas = pd.read_csv(sas_path)[["user_id", "NDCG@20", "MRR@20"]].rename(
            columns={"NDCG@20": "NDCG@20_sasrec", "MRR@20": "MRR@20_sasrec"}
        )
        markov = pd.read_csv(markov_path)[["user_id", "NDCG@20", "MRR@20"]].rename(
            columns={"NDCG@20": "NDCG@20_markov", "MRR@20": "MRR@20_markov"}
        )
        merged = sas.merge(markov, on="user_id")
        for metric in ["NDCG@20", "MRR@20"]:
            diff = merged[f"{metric}_sasrec"] - merged[f"{metric}_markov"]
            mean, low, high = bootstrap_ci(diff.to_numpy())
            rows.append(
                {
                    "comparison": "Beauty_p10_fixed_SASRec_minus_Markov",
                    "order": order,
                    "metric": metric,
                    "eval_users": int(len(merged)),
                    "gap": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "supported_positive": bool(low > 0.0),
                    "supported_negative": bool(high < 0.0),
                }
            )
    return rows


def sports_secondary_metric() -> List[Dict[str, object]]:
    path = OUT / "sports_det_hp_selected_per_user.csv"
    if not path.exists():
        return []
    selected = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    for order in ["V1", "V2"]:
        item = selected[(selected["order"] == order) & (selected["model"] == "ItemKNN")]
        easer = selected[(selected["order"] == order) & (selected["model"] == "EASER")]
        merged = item[["user_id", "NDCG@20", "MRR@20"]].merge(
            easer[["user_id", "NDCG@20", "MRR@20"]],
            on="user_id",
            suffixes=("_itemknn", "_easer"),
        )
        for metric in ["NDCG@20", "MRR@20"]:
            diff = merged[f"{metric}_itemknn"] - merged[f"{metric}_easer"]
            mean, low, high = bootstrap_ci(diff.to_numpy())
            rows.append(
                {
                    "comparison": "Sports_p5_tuned_ItemKNN_minus_EASER",
                    "order": order,
                    "metric": metric,
                    "eval_users": int(len(merged)),
                    "gap": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "supported_positive": bool(low > 0.0),
                    "supported_negative": bool(high < 0.0),
                }
            )
    return rows


def headline_secondary_metrics() -> pd.DataFrame:
    leaderboards = pd.read_csv(OUT / "leaderboards_seed17_policy.csv")
    rows = beauty_secondary_metric(leaderboards) + sports_secondary_metric()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "headline_secondary_metrics.csv", index=False)
    return out


def pcore_p_value(filter_tag: str) -> Optional[int]:
    match = re.fullmatch(r"pcore_p(\d+)", str(filter_tag))
    return int(match.group(1)) if match else None


def mechanism_table() -> pd.DataFrame:
    raw = pd.read_csv(OUT / "raw_pcore_diagnostics.csv")
    matched = pd.read_csv(OUT / "matched_v1v2_pairs_with_bootstrap.csv")
    common_path = OUT / "common_test_probe_summary.csv"
    common = pd.read_csv(common_path) if common_path.exists() else pd.DataFrame()
    rows: List[Dict[str, object]] = []

    pcore = matched[matched["filter_tag"].str.startswith("pcore_p")].copy()
    pcore["p"] = pcore["filter_tag"].map(pcore_p_value)
    for _, m in pcore.sort_values(["dataset", "p"]).iterrows():
        dataset = m["dataset"]
        p = int(m["p"])
        v1 = raw[(raw["dataset"] == dataset) & (raw["p"] == p) & (raw["order"] == "V1")]
        v2 = raw[(raw["dataset"] == dataset) & (raw["p"] == p) & (raw["order"] == "V2")]
        jj = raw[(raw["dataset"] == dataset) & (raw["p"] == p) & (raw["order"] == "V1_vs_V2")]
        c = common[
            (common.get("dataset", pd.Series(dtype=object)) == dataset)
            & (common.get("p", pd.Series(dtype=float)) == p)
            & (common.get("setup", pd.Series(dtype=object)) == "common_test_common_candidate")
        ]
        row: Dict[str, object] = {
            "dataset": dataset,
            "p": p,
            "v1_train_users": int(v1["train_users"].iloc[0]) if len(v1) else np.nan,
            "v2_train_users": int(v2["train_users"].iloc[0]) if len(v2) else np.nan,
            "v1_train_items": int(v1["train_items"].iloc[0]) if len(v1) else np.nan,
            "v2_train_items": int(v2["train_items"].iloc[0]) if len(v2) else np.nan,
            "v1_candidate_items": int(v1["candidate_items"].iloc[0]) if len(v1) else np.nan,
            "v2_candidate_items": int(v2["candidate_items"].iloc[0]) if len(v2) else np.nan,
            "v1_test_users": int(v1["test_users"].iloc[0]) if len(v1) else np.nan,
            "v2_test_users": int(v2["test_users"].iloc[0]) if len(v2) else np.nan,
            "v1_cold_target_rate": float(v1["cold_target_rate_before_projection"].iloc[0]) if len(v1) else np.nan,
            "v2_cold_target_rate": float(v2["cold_target_rate_before_projection"].iloc[0]) if len(v2) else np.nan,
            "train_edge_jaccard": float(jj["train_edge_jaccard"].iloc[0]) if len(jj) else np.nan,
            "test_user_jaccard": float(jj["test_user_jaccard"].iloc[0]) if len(jj) else np.nan,
            "native_v1_top": m["v1_winner"],
            "native_v2_top": m["v2_winner"],
            "common_v1_top": c["v1_winner"].iloc[0] if len(c) else "",
            "common_v2_top": c["v2_winner"].iloc[0] if len(c) else "",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mechanism_by_dataset_p.csv", index=False)
    return out


def p0_projection_control() -> pd.DataFrame:
    matched = pd.read_csv(OUT / "matched_v1v2_pairs_with_bootstrap.csv")
    p0 = matched[matched["filter_tag"] == "pcore_p0"]
    guarded = p0[p0["passes_guardrail"]]
    out = pd.DataFrame(
        [
            {
                "matched_pairs": int(len(p0)),
                "winner_flips": int(p0["winner_flip"].sum()),
                "guardrailed_pairs": int(len(guarded)),
                "guardrailed_winner_flips": int(guarded["winner_flip"].sum()),
                "bootstrap_supported_flips": int(p0["bootstrap_supported_flip"].sum()),
                "guardrailed_bootstrap_supported_flips": int(guarded["bootstrap_supported_flip"].sum()),
            }
        ]
    )
    out.to_csv(OUT / "p0_projection_control_summary.csv", index=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {
        "violation_decomposition": violation_decomposition(),
        "iterative_fitted_core": iterative_fitted_core(),
        "headline_secondary_metrics": headline_secondary_metrics(),
        "mechanism_by_dataset_p": mechanism_table(),
        "p0_projection_control": p0_projection_control(),
    }
    for name, df in parts.items():
        print(f"\n## {name}")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
