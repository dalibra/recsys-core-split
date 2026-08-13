"""Summarize the Beauty p=10 SASRec hyperparameter spot check."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from run_sasrec_hp_check import FILTER_TAG, HP_GRID, Q_TAG, RESULTS, SEEDS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
ORDERS = {
    "V1": "Beauty__pcore_p10__v1_filter_then_split",
    "V2": "Beauty__pcore_p10__v2_split_then_filter_train",
}


def metric_from_file(path: Path, prefix: str, metric: str = "NDCG", k: int = 20) -> float:
    df = pd.read_csv(path)
    suffix = f"{prefix}_{metric}@{k}"
    if "metric_name" in df.columns and "metric_value" in df.columns:
        hit = df[df["metric_name"].astype(str).str.endswith(suffix)]
        if hit.empty:
            raise ValueError(f"Missing {suffix} in {path}")
        return float(hit["metric_value"].iloc[0])
    df = pd.read_csv(path, index_col=0)
    return float(df.loc[metric, f"@{k}"])


def result_path(variant: str, stem: str, prefix: str) -> Path:
    return RESULTS / variant / Q_TAG / "SASRec" / prefix / f"{stem}.csv"


def markov_score(variant: str) -> float:
    path = RESULTS / variant / Q_TAG / "Markov" / "test_last" / "baseline_markov.csv"
    return metric_from_file(path, "test_last")


def fixed_sasrec_score(variant: str, seed: int = 17) -> float:
    path = RESULTS / variant / Q_TAG / "SASRec" / "test_last" / f"0_32_1_1_0.1_128_256_{seed}.csv"
    return metric_from_file(path, "test_last")


def collect_grid() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for order, variant in ORDERS.items():
        for spec in HP_GRID:
            stem = spec.result_stem(17)
            val_path = result_path(variant, stem, "val_last")
            test_path = result_path(variant, stem, "test_last")
            rows.append(
                {
                    "dataset": "Beauty",
                    "filter_tag": FILTER_TAG,
                    "order": order,
                    "variant": variant,
                    "tag": spec.tag,
                    "grid_idx": spec.grid_idx,
                    "hidden_units": spec.hidden_units,
                    "dropout": spec.dropout,
                    "lr": spec.lr,
                    "seed": 17,
                    "val_ndcg20": metric_from_file(val_path, "val_last") if val_path.exists() else np.nan,
                    "test_ndcg20": metric_from_file(test_path, "test_last") if test_path.exists() else np.nan,
                    "val_file": str(val_path) if val_path.exists() else "",
                    "test_file": str(test_path) if test_path.exists() else "",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "beauty_sasrec_hp_grid.csv", index=False)
    return out


def select_best(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order, sub in grid.groupby("order", sort=True):
        complete = sub.dropna(subset=["val_ndcg20", "test_ndcg20"]).copy()
        if complete.empty:
            continue
        best = complete.sort_values(
            ["val_ndcg20", "test_ndcg20", "hidden_units", "dropout", "lr"],
            ascending=[False, False, True, True, False],
        ).iloc[0]
        variant = str(best["variant"])
        rows.append(
            {
                **best.to_dict(),
                "selection_metric": "val_ndcg20",
                "markov_test_ndcg20": markov_score(variant),
                "fixed_sasrec_seed17_test_ndcg20": fixed_sasrec_score(variant, 17),
                "best_sasrec_gt_markov": bool(float(best["test_ndcg20"]) > markov_score(variant)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "beauty_sasrec_hp_best.csv", index=False)
    return out


def collect_best_seeds(best: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if best.empty:
        return pd.DataFrame()
    by_tag = {spec.tag: spec for spec in HP_GRID}
    for _, selected in best.iterrows():
        spec = by_tag[str(selected["tag"])]
        variant = str(selected["variant"])
        for seed in SEEDS:
            path = result_path(variant, spec.result_stem(seed), "test_last")
            rows.append(
                {
                    "dataset": "Beauty",
                    "filter_tag": FILTER_TAG,
                    "order": str(selected["order"]),
                    "variant": variant,
                    "tag": spec.tag,
                    "grid_idx": spec.grid_idx,
                    "hidden_units": spec.hidden_units,
                    "dropout": spec.dropout,
                    "lr": spec.lr,
                    "seed": seed,
                    "test_ndcg20": metric_from_file(path, "test_last") if path.exists() else np.nan,
                    "markov_test_ndcg20": markov_score(variant),
                    "test_file": str(path) if path.exists() else "",
                }
            )
    out = pd.DataFrame(rows)
    out["sasrec_gt_markov"] = out["test_ndcg20"] > out["markov_test_ndcg20"]
    out["markov_gt_sasrec"] = out["markov_test_ndcg20"] > out["test_ndcg20"]
    out.to_csv(OUT / "beauty_sasrec_hp_seed_check.csv", index=False)
    return out


def write_markdown(grid: pd.DataFrame, best: pd.DataFrame, seeds: pd.DataFrame) -> None:
    lines = [
        "# Beauty p=10 SASRec HP Spot Check",
        "",
        "Grid: 12 SASRec settings per arm, selected by validation NDCG@20; Markov is parameter-free.",
        "",
        "| order | complete grid rows | best tag | val NDCG@20 | test NDCG@20 | Markov test | fixed SASRec test | best SASRec>Markov |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    complete_counts = grid.dropna(subset=["val_ndcg20", "test_ndcg20"]).groupby("order").size()
    for _, row in best.iterrows():
        lines.append(
            f"| {row['order']} | {int(complete_counts.get(row['order'], 0))} | {row['tag']} | "
            f"{float(row['val_ndcg20']):.4f} | {float(row['test_ndcg20']):.4f} | "
            f"{float(row['markov_test_ndcg20']):.4f} | "
            f"{float(row['fixed_sasrec_seed17_test_ndcg20']):.4f} | "
            f"{bool(row['best_sasrec_gt_markov'])} |"
        )
    if not seeds.empty:
        lines.extend(
            [
                "",
                "## Best-config seed follow-up",
                "",
                "| order | tag | seeds complete | SASRec mean | SASRec range | Markov test | SASRec>Markov | Markov>SASRec |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for (order, tag), sub in seeds.groupby(["order", "tag"], sort=True):
            complete = sub.dropna(subset=["test_ndcg20"])
            if complete.empty:
                continue
            lines.append(
                f"| {order} | {tag} | {len(complete)} | "
                f"{float(complete['test_ndcg20'].mean()):.4f} | "
                f"{float(complete['test_ndcg20'].min()):.4f}--{float(complete['test_ndcg20'].max()):.4f} | "
                f"{float(complete['markov_test_ndcg20'].iloc[0]):.4f} | "
                f"{int(complete['sasrec_gt_markov'].sum())} | "
                f"{int(complete['markov_gt_sasrec'].sum())} |"
            )
    (OUT / "beauty_sasrec_hp_check.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = collect_grid()
    best = select_best(grid)
    seeds = collect_best_seeds(best)
    write_markdown(grid, best, seeds)
    print(f"Saved Beauty SASRec HP summaries to {OUT}")


if __name__ == "__main__":
    main()
