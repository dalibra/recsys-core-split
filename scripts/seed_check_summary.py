"""Summarize matched-seed checks for selected p-core V1/V2 variants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "exps" / "data" / "results" / "global_timesplit" / "val_last_train_item"
OUT = ROOT / "results"
LEADERBOARD_PATH = OUT / "leaderboards_seed17_policy.csv"
MODELS = ["SASRec", "GRU4Rec", "BERT4Rec"]
SEED_SPECS = [
    ("Sports", "pcore_p5", MODELS),
    ("Movielens-1m", "pcore_p5", MODELS),
    ("Beauty", "pcore_p10", ["SASRec"]),
    ("Beauty", "pcore_p5", ["SASRec"]),
]
SEEDS = [17, 23, 42, 101, 202]
Q_TAG = "q09"


def metric_from_file(path: Path, metric: str = "NDCG", k: int = 20) -> float:
    df = pd.read_csv(path)
    suffix = f"{metric}@{k}"
    if "metric_name" in df.columns and "metric_value" in df.columns:
        hit = df[df["metric_name"].astype(str).str.endswith(suffix)]
        if hit.empty:
            raise ValueError(f"Missing {suffix} in {path}")
        return float(hit["metric_value"].iloc[0])
    df = pd.read_csv(path, index_col=0)
    return float(df.loc[metric, f"@{k}"])


def result_file(variant: str, model: str, seed: int) -> Optional[Path]:
    path = RESULTS / variant / Q_TAG / model / "test_last"
    if not path.exists():
        return None
    matches = sorted(p for p in path.glob("*.csv") if p.name.endswith(f"_{seed}.csv"))
    return matches[0] if matches else None


def collect_scores() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, filter_tag, models in SEED_SPECS:
        for order, tag in [("V1", "v1_filter_then_split"), ("V2", "v2_split_then_filter_train")]:
            variant = f"{dataset}__{filter_tag}__{tag}"
            for model in models:
                for seed in SEEDS:
                    path = result_file(variant, model, seed)
                    if path is None:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "filter_tag": filter_tag,
                            "order": order,
                            "variant": variant,
                            "model": model,
                            "seed": seed,
                            "NDCG@20": metric_from_file(path),
                            "metric_file": str(path),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "seed_check_scores.csv", index=False)
    return out


def summarize_model_effects(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, filter_tag, model), sub in scores.groupby(["dataset", "filter_tag", "model"], sort=True):
        v1 = sub[sub["order"] == "V1"].set_index("seed")["NDCG@20"]
        v2 = sub[sub["order"] == "V2"].set_index("seed")["NDCG@20"]
        common = sorted(set(v1.index) & set(v2.index))
        v1_common = v1.loc[common]
        v2_common = v2.loc[common]
        v1_mean = float(v1_common.mean()) if len(v1_common) else np.nan
        v2_mean = float(v2_common.mean()) if len(v2_common) else np.nan
        v1_std = float(v1_common.std(ddof=1)) if len(v1_common) > 1 else 0.0
        v2_std = float(v2_common.std(ddof=1)) if len(v2_common) > 1 else 0.0
        effect = abs(v1_mean - v2_mean) if len(v1_common) else np.nan
        denom = float(np.sqrt(v1_std**2 + v2_std**2))
        ratio = float(effect / denom) if denom > 0 else np.inf
        diffs = v1_common - v2_common
        mean_sign = np.sign(v1_mean - v2_mean)
        rows.append(
            {
                "dataset": dataset,
                "filter_tag": filter_tag,
                "model": model,
                "paired_seeds": int(len(common)),
                "v1_mean": v1_mean,
                "v1_std": v1_std,
                "v2_mean": v2_mean,
                "v2_std": v2_std,
                "abs_mean_order_effect": effect,
                "combined_seed_std": denom,
                "effect_to_seed_noise": ratio,
                "v1_minus_v2_mean": v1_mean - v2_mean,
                "same_direction_seeds": int((np.sign(diffs) == mean_sign).sum()) if mean_sign != 0 else 0,
                "v1_better_seeds": int((diffs > 0).sum()),
                "v2_better_seeds": int((diffs < 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "seed_check_model_summary.csv", index=False)
    return out


def summarize_rank_winners(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = {(dataset, filter_tag): set(models) for dataset, filter_tag, models in SEED_SPECS}
    for (dataset, filter_tag, order, seed), sub in scores.groupby(["dataset", "filter_tag", "order", "seed"], sort=True):
        expected_models = expected.get((dataset, filter_tag), set())
        if len(expected_models) < 2 or set(sub["model"]) != expected_models:
            continue
        best = sub.sort_values(["NDCG@20", "model"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "filter_tag": filter_tag,
                "order": order,
                "seed": int(seed),
                "winner": str(best["model"]),
                "winner_ndcg": float(best["NDCG@20"]),
            }
        )
    winners = pd.DataFrame(rows)
    winners.to_csv(OUT / "seed_check_rank_winners.csv", index=False)

    summary_rows = []
    for (dataset, filter_tag), sub in winners.groupby(["dataset", "filter_tag"], sort=True):
        common = sorted(set(sub[sub["order"] == "V1"]["seed"]) & set(sub[sub["order"] == "V2"]["seed"]))
        flips = 0
        for seed in common:
            w1 = sub[(sub["order"] == "V1") & (sub["seed"] == seed)]["winner"].iloc[0]
            w2 = sub[(sub["order"] == "V2") & (sub["seed"] == seed)]["winner"].iloc[0]
            flips += int(w1 != w2)
        mean_winners = {}
        for order in ["V1", "V2"]:
            mean_scores = scores[
                (scores["dataset"] == dataset)
                & (scores["filter_tag"] == filter_tag)
                & (scores["order"] == order)
            ].groupby("model")["NDCG@20"].mean()
            mean_winners[order] = str(mean_scores.sort_values(ascending=False).index[0])
        summary_rows.append(
            {
                "dataset": dataset,
                "filter_tag": filter_tag,
                "paired_seed_rankings": int(len(common)),
                "seed_level_winner_flips": int(flips),
                "v1_mean_winner": mean_winners.get("V1"),
                "v2_mean_winner": mean_winners.get("V2"),
                "mean_winner_flip": mean_winners.get("V1") != mean_winners.get("V2"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "seed_check_rank_summary.csv", index=False)
    return summary


def summarize_beauty_sasrec_markov_persistence(scores: pd.DataFrame) -> pd.DataFrame:
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame()
    leader = pd.read_csv(LEADERBOARD_PATH)
    rows = []
    for filter_tag in ["pcore_p10", "pcore_p5"]:
        for order, tag in [("V1", "v1_filter_then_split"), ("V2", "v2_split_then_filter_train")]:
            variant = f"Beauty__{filter_tag}__{tag}"
            markov = leader[(leader["variant"] == variant) & (leader["model"] == "Markov")]
            sas = scores[
                (scores["dataset"] == "Beauty")
                & (scores["filter_tag"] == filter_tag)
                & (scores["order"] == order)
                & (scores["model"] == "SASRec")
            ]
            if markov.empty or sas.empty:
                continue
            markov_score = float(markov["NDCG@20"].iloc[0])
            rows.append(
                {
                    "dataset": "Beauty",
                    "filter_tag": filter_tag,
                    "order": order,
                    "markov_ndcg": markov_score,
                    "sasrec_seeds": int(len(sas)),
                    "sasrec_gt_markov": int((sas["NDCG@20"] > markov_score).sum()),
                    "markov_gt_sasrec": int((markov_score > sas["NDCG@20"]).sum()),
                    "sasrec_min": float(sas["NDCG@20"].min()),
                    "sasrec_max": float(sas["NDCG@20"].max()),
                    "sasrec_mean": float(sas["NDCG@20"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "beauty_sasrec_markov_seed_persistence.csv", index=False)
    return out


def write_markdown(
    model_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    lines = [
        "# Clean seed check",
        "",
        "Scope: Sports and Movielens-1m p-core p=5 with SASRec/GRU4Rec/BERT4Rec; Beauty p-core p=5 and p=10 with SASRec; seeds 17/23/42/101/202.",
        "",
        "## Model-level order effect",
        "",
        "| dataset | filter | model | paired seeds | V1 mean | V2 mean | effect/noise | V1>V2 seeds | V2>V1 seeds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in model_summary.iterrows():
        ratio = r["effect_to_seed_noise"]
        ratio_s = "inf" if np.isinf(ratio) else f"{float(ratio):.2f}"
        lines.append(
            f"| {r['dataset']} | {r['filter_tag']} | {r['model']} | {int(r['paired_seeds'])} | "
            f"{float(r['v1_mean']):.4f} | {float(r['v2_mean']):.4f} | {ratio_s} | "
            f"{int(r['v1_better_seeds'])} | {int(r['v2_better_seeds'])} |"
        )
    lines.extend(
        [
            "",
            "## Winner stability",
            "",
            "| dataset | filter | paired seed rankings | seed-level winner flips | V1 mean winner | V2 mean winner | mean winner flip |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for _, r in rank_summary.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['filter_tag']} | {int(r['paired_seed_rankings'])} | "
            f"{int(r['seed_level_winner_flips'])} | {r['v1_mean_winner']} | "
            f"{r['v2_mean_winner']} | {bool(r['mean_winner_flip'])} |"
        )
    if len(persistence):
        lines.extend(
            [
                "",
                "## Beauty SASRec vs Markov seed persistence",
                "",
                "| filter | order | Markov NDCG@20 | SASRec seeds | SASRec>Markov | Markov>SASRec | SASRec range |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, r in persistence.iterrows():
            lines.append(
                f"| {r['filter_tag']} | {r['order']} | {float(r['markov_ndcg']):.4f} | "
                f"{int(r['sasrec_seeds'])} | {int(r['sasrec_gt_markov'])} | "
                f"{int(r['markov_gt_sasrec'])} | "
                f"{float(r['sasrec_min']):.4f}--{float(r['sasrec_max']):.4f} |"
            )
    (OUT / "seed_check_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = collect_scores()
    model_summary = summarize_model_effects(scores)
    rank_summary = summarize_rank_winners(scores)
    persistence = summarize_beauty_sasrec_markov_persistence(scores)
    write_markdown(model_summary, rank_summary, persistence)
    print(f"Saved seed summaries to {OUT}")


if __name__ == "__main__":
    main()
