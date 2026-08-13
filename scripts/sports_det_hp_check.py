"""Sports p=5 deterministic ItemKNN/EASE^R tuning check.

This script keeps the deterministic hyperparameter grid out of the fixed-grid
leaderboard directories. It selects ItemKNN and EASE^R per arm by validation
NDCG@20, evaluates the selected settings on test, and bootstraps the selected
ItemKNN-minus-EASE^R test gap.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
SPLIT_ROOT = DATA / "splitted" / "global_timesplit" / "val_last_train_item"
RESULTS = ROOT / "results"
Q_TAG = "q09"
K_VALUES = [20, 50, 100, 200, 400]
REG_VALUES = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
VARIANTS = {
    "V1": "Sports__pcore_p5__v1_filter_then_split",
    "V2": "Sports__pcore_p5__v2_split_then_filter_train",
}

sys.path.insert(0, str(EXPS / "runs"))
import train_baselines as tb  # noqa: E402


def max_item_id(*frames: pd.DataFrame) -> int:
    vals = [int(frame["item_id"].max()) for frame in frames if not frame.empty]
    return max(vals) if vals else 0


def per_user_from_recs(
    eval_users: np.ndarray,
    targets: np.ndarray,
    recs: np.ndarray,
    ks: Sequence[int],
) -> pd.DataFrame:
    rows = []
    max_k = max(ks)
    for idx, user_id in enumerate(eval_users.tolist()):
        target = int(targets[idx])
        ranked = recs[idx, :max_k]
        hits = np.flatnonzero(ranked == target)
        rank = int(hits[0] + 1) if hits.size else 0
        row = {"user_id": int(user_id), "target_item_id": target, "rank": rank}
        for k in ks:
            if rank and rank <= k:
                row[f"HitRate@{k}"] = 1.0
                row[f"MRR@{k}"] = 1.0 / rank
                row[f"NDCG@{k}"] = 1.0 / np.log2(rank + 1)
            else:
                row[f"HitRate@{k}"] = 0.0
                row[f"MRR@{k}"] = 0.0
                row[f"NDCG@{k}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_recs(
    recs: np.ndarray,
    eval_users: np.ndarray,
    targets: np.ndarray,
    ks: Sequence[int],
    n_items: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    metrics = tb.compute_metrics_from_topk(recs=recs, targets=targets, ks=ks, n_items=n_items)
    per_user = per_user_from_recs(eval_users=eval_users, targets=targets, recs=recs, ks=ks)
    return metrics, per_user


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 17) -> Tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def train_easer_multi(
    train: pd.DataFrame,
    n_items: int,
    candidates: int,
    regs: Iterable[float],
    progress_every: int = 4000,
) -> Dict[float, sparse.csr_matrix]:
    x_bin = tb.build_binary_user_item(train, n_items)
    if x_bin.shape[0] == 0 or x_bin.shape[1] == 0:
        empty = sparse.csr_matrix((n_items, n_items), dtype=np.float32)
        return {float(reg): empty for reg in regs}

    cooc = (x_bin.T @ x_bin).tocsr().astype(np.float32)
    cooc.setdiag(0)
    cooc.eliminate_zeros()

    reg_values = [float(reg) for reg in regs]
    rows_by_reg: Dict[float, List[int]] = {reg: [] for reg in reg_values}
    cols_by_reg: Dict[float, List[int]] = {reg: [] for reg in reg_values}
    vals_by_reg: Dict[float, List[float]] = {reg: [] for reg in reg_values}
    eps = np.float32(1e-6)

    for item in range(n_items):
        start = cooc.indptr[item]
        end = cooc.indptr[item + 1]
        if end <= start:
            if progress_every > 0 and (item + 1) % progress_every == 0:
                print(f"[EASER-grid] processed {item + 1}/{n_items} items", flush=True)
            continue

        neigh = cooc.indices[start:end]
        b_full = cooc.data[start:end]
        if neigh.size > candidates:
            pick = np.argpartition(b_full, -candidates)[-candidates:]
            pick = pick[np.argsort(-b_full[pick], kind="stable")]
            neigh = neigh[pick]
            b = b_full[pick].astype(np.float32, copy=False)
        else:
            b = b_full.astype(np.float32, copy=False)

        m = neigh.size
        if m == 0:
            continue

        base_gram = cooc[neigh][:, neigh].toarray().astype(np.float32, copy=False)
        diag = np.arange(m)
        for reg in reg_values:
            gram = base_gram.copy()
            gram[diag, diag] += np.float32(reg)
            try:
                w = np.linalg.solve(gram, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(gram + np.eye(m, dtype=np.float32) * eps, b, rcond=None)[0]
            rows_by_reg[reg].extend(neigh.tolist())
            cols_by_reg[reg].extend([item] * m)
            vals_by_reg[reg].extend(w.astype(np.float32, copy=False).tolist())

        if progress_every > 0 and (item + 1) % progress_every == 0:
            print(f"[EASER-grid] processed {item + 1}/{n_items} items", flush=True)

    out: Dict[float, sparse.csr_matrix] = {}
    for reg in reg_values:
        if not rows_by_reg[reg]:
            out[reg] = sparse.csr_matrix((n_items, n_items), dtype=np.float32)
        else:
            out[reg] = sparse.csr_matrix(
                (
                    np.asarray(vals_by_reg[reg], dtype=np.float32),
                    (np.asarray(rows_by_reg[reg], dtype=np.int32), np.asarray(cols_by_reg[reg], dtype=np.int32)),
                ),
                shape=(n_items, n_items),
                dtype=np.float32,
            )
    return out


def prepare_queries(
    split_df: pd.DataFrame,
    n_items: int,
    filter_seen: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, sparse.csr_matrix, List[np.ndarray]]:
    inputs, eval_users, targets = tb.extract_eval_users_and_targets(split_df)
    q_binary, _, _, seen_lists = tb.build_query_matrices(
        inputs=inputs,
        eval_users=eval_users,
        n_items=n_items,
        session_len=10,
        filter_seen=filter_seen,
    )
    return inputs, eval_users, targets, q_binary, seen_lists


def run_variant(order: str, variant: str, ks: Sequence[int], easer_candidates: int, batch_size: int) -> Tuple[List[Dict[str, object]], Dict[str, pd.DataFrame]]:
    print(f"[variant] {order} {variant}", flush=True)
    split_dir = SPLIT_ROOT / variant / Q_TAG
    train, validation, test = tb.load_split(str(split_dir))
    n_items = max_item_id(train, validation, test)
    filter_seen = tb.safe_load_dataset_filter_seen(str(EXPS), "Sports")
    popular = tb.build_popularity(train, n_items)
    train_sorted = train.sort_values(["user_id", "timestamp"], kind="stable")
    item_freq = np.zeros(n_items, dtype=np.float32)
    counts = train["item_id"].value_counts()
    item_freq[counts.index.to_numpy(dtype=np.int64) - 1] = counts.to_numpy(dtype=np.float32)

    _, val_users, val_targets, q_val, val_seen = prepare_queries(validation, n_items, filter_seen)
    _, test_users, test_targets, q_test, test_seen = prepare_queries(test, n_items, filter_seen)
    max_k = max(ks)
    rows: List[Dict[str, object]] = []
    selected_per_user: Dict[str, pd.DataFrame] = {}

    max_neighbors = max(K_VALUES)
    covis = tb.build_covisit_matrix(train_sorted, n_items, window=5, neighbors=max_neighbors)
    cosine = tb.normalize_cosine(covis, item_freq)
    for neighbors in K_VALUES:
        itemknn = tb.prune_topk_rows(cosine, neighbors)
        for split_name, q, seen_lists, eval_users, targets in [
            ("val", q_val, val_seen, val_users, val_targets),
            ("test", q_test, test_seen, test_users, test_targets),
        ]:
            recs = tb.batch_predict_sparse(q, itemknn, seen_lists, popular, max_k, filter_seen, batch_size)
            metrics, per_user = evaluate_recs(recs, eval_users, targets, ks, n_items)
            row = {
                "order": order,
                "variant": variant,
                "model": "ItemKNN",
                "param_name": "neighbors",
                "param_value": neighbors,
                "split": split_name,
                "eval_users": int(len(eval_users)),
            }
            row.update(metrics)
            rows.append(row)
            selected_per_user[f"ItemKNN:neighbors={neighbors}:{split_name}"] = per_user
        print(f"[done] {order} ItemKNN neighbors={neighbors}", flush=True)

    easer_mats = train_easer_multi(train, n_items, candidates=easer_candidates, regs=REG_VALUES)
    for reg, easer in easer_mats.items():
        for split_name, q, seen_lists, eval_users, targets in [
            ("val", q_val, val_seen, val_users, val_targets),
            ("test", q_test, test_seen, test_users, test_targets),
        ]:
            recs = tb.batch_predict_sparse(q, easer, seen_lists, popular, max_k, filter_seen, batch_size)
            metrics, per_user = evaluate_recs(recs, eval_users, targets, ks, n_items)
            row = {
                "order": order,
                "variant": variant,
                "model": "EASER",
                "param_name": "reg",
                "param_value": reg,
                "split": split_name,
                "eval_users": int(len(eval_users)),
            }
            row.update(metrics)
            rows.append(row)
            selected_per_user[f"EASER:reg={reg:g}:{split_name}"] = per_user
        print(f"[done] {order} EASER reg={reg:g}", flush=True)

    return rows, selected_per_user


def pick_best(grid: pd.DataFrame) -> pd.DataFrame:
    val = grid[grid["split"] == "val"].copy()
    test = grid[grid["split"] == "test"].copy()
    rows = []
    for (order, model), group in val.groupby(["order", "model"], sort=True):
        best = group.sort_values(["NDCG@20", "param_value"], ascending=[False, True], kind="stable").iloc[0]
        match = test[
            (test["order"] == order)
            & (test["model"] == model)
            & (test["param_name"] == best["param_name"])
            & (test["param_value"] == best["param_value"])
        ]
        test_row = match.iloc[0]
        rows.append(
            {
                "order": order,
                "model": model,
                "param_name": best["param_name"],
                "param_value": best["param_value"],
                "val_ndcg20": float(best["NDCG@20"]),
                "test_ndcg20": float(test_row["NDCG@20"]),
                "test_hitrate20": float(test_row["HitRate@20"]),
                "test_mrr20": float(test_row["MRR@20"]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, float_cols: Sequence[str] = ()) -> str:
    if df.empty:
        return "_empty_"
    out = df.copy()
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    for col in out.columns:
        if col not in float_cols:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(best: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    float_cols = [
        "param_value",
        "val_ndcg20",
        "test_ndcg20",
        "test_hitrate20",
        "test_mrr20",
        "gap",
        "ci_low",
        "ci_high",
    ]
    lines = [
        "# Sports p=5 deterministic HP check",
        "",
        "Validation selects ItemKNN neighbors and EASE^R regularization per arm; test comparisons use selected settings.",
        f"Grid: ItemKNN neighbors {K_VALUES}; EASE^R regularization {[int(x) if float(x).is_integer() else x for x in REG_VALUES]}.",
        "",
        "## Best settings",
        markdown_table(best, float_cols=float_cols),
        "",
        "## Selected ItemKNN-minus-EASE^R pairwise gaps",
        markdown_table(pairwise, float_cols=float_cols),
        "",
    ]
    (RESULTS / "sports_det_hp_check.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easer-candidates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_report(
            pd.read_csv(RESULTS / "sports_det_hp_best.csv"),
            pd.read_csv(RESULTS / "sports_det_hp_pairwise.csv"),
        )
        return

    ks = [1, 5, 10, 20]
    rows: List[Dict[str, object]] = []
    per_user_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    for order, variant in VARIANTS.items():
        variant_rows, per_users = run_variant(order, variant, ks, args.easer_candidates, args.batch_size)
        rows.extend(variant_rows)
        for key, per_user in per_users.items():
            per_user_by_key[(order, key)] = per_user

    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "sports_det_hp_grid.csv", index=False)
    best = pick_best(grid)
    best.to_csv(RESULTS / "sports_det_hp_best.csv", index=False)

    selected_rows = []
    pairwise_rows = []
    for order in ["V1", "V2"]:
        order_best = best[best["order"] == order]
        selected: Dict[str, pd.DataFrame] = {}
        for _, row in order_best.iterrows():
            model = str(row["model"])
            param_name = str(row["param_name"])
            param_value = float(row["param_value"])
            value_str = f"{param_value:g}" if model == "EASER" else str(int(param_value))
            key = f"{model}:{param_name}={value_str}:test"
            per_user = per_user_by_key[(order, key)].copy()
            per_user["order"] = order
            per_user["model"] = model
            per_user["param_name"] = param_name
            per_user["param_value"] = param_value
            selected[model] = per_user
            selected_rows.append(per_user)

        merged = selected["ItemKNN"][["user_id", "NDCG@20", "HitRate@20"]].merge(
            selected["EASER"][["user_id", "NDCG@20", "HitRate@20"]],
            on="user_id",
            suffixes=("_itemknn", "_easer"),
        )
        for metric in ["NDCG@20", "HitRate@20"]:
            diff = (merged[f"{metric}_itemknn"] - merged[f"{metric}_easer"]).to_numpy()
            mean, low, high = bootstrap_ci(diff)
            pairwise_rows.append(
                {
                    "comparison": "ItemKNN_minus_EASER",
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

    selected_per_user = pd.concat(selected_rows, ignore_index=True)
    selected_per_user.to_csv(RESULTS / "sports_det_hp_selected_per_user.csv", index=False)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(RESULTS / "sports_det_hp_pairwise.csv", index=False)

    write_report(best, pairwise)


if __name__ == "__main__":
    main()
