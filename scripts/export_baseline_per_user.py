"""Export per-user NDCG/HR/MRR rows for deterministic baselines."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
SPLIT_ROOT = DATA / "splitted" / "global_timesplit" / "val_last_train_item"
RESULTS_ROOT = DATA / "results" / "global_timesplit" / "val_last_train_item"
Q_TAG = "q09"
DETERMINISTIC = {"MostPopular", "Markov", "SKNN", "ItemKNN", "EASER"}

sys.path.insert(0, str(EXPS / "runs"))
import train_baselines as tb  # noqa: E402


def needed_pairs(
    matched_path: Path,
    guarded_only: bool,
    pcore_only: bool,
    exclude_p0: bool,
    all_deterministic: bool,
) -> Dict[str, Set[str]]:
    matched = pd.read_csv(matched_path)
    if guarded_only:
        matched = matched[matched["passes_guardrail"]]
    if pcore_only:
        matched = matched[matched["filter_family"] == "pcore"]
    if exclude_p0:
        matched = matched[matched["filter_tag"] != "pcore_p0"]
    out: Dict[str, Set[str]] = defaultdict(set)
    for _, row in matched.iterrows():
        for side in ["v1", "v2"]:
            variant = str(row[f"{side}_variant"])
            if all_deterministic:
                out[variant].update(DETERMINISTIC)
                continue
            for role in ["winner", "runner_up"]:
                model = str(row[f"{side}_{role}"])
                if model in DETERMINISTIC:
                    out[variant].add(model)
    return out


def output_path(variant: str, model: str) -> Path:
    return RESULTS_ROOT / variant / Q_TAG / model / "test_last_per_user" / f"baseline_{model.lower()}.csv"


def max_item_id(*frames: pd.DataFrame) -> int:
    vals = []
    for frame in frames:
        if not frame.empty:
            vals.append(int(frame["item_id"].max()))
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


def save_per_user(variant: str, model: str, per_user: pd.DataFrame) -> None:
    path = output_path(variant, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(path, index=False)


def export_variant(variant: str, models: Set[str], ks: Sequence[int], skip_existing: bool) -> None:
    missing_models = [m for m in sorted(models) if not (skip_existing and output_path(variant, m).exists())]
    if not missing_models:
        print(f"[skip] {variant}: per-user files already exist")
        return

    split_dir = SPLIT_ROOT / variant / Q_TAG
    train, validation, test = tb.load_split(str(split_dir))
    if train.empty or test.empty:
        print(f"[skip] {variant}: empty train/test")
        return

    n_items = max_item_id(train, validation, test)
    dataset = variant.split("__")[0]
    filter_seen = tb.safe_load_dataset_filter_seen(str(EXPS), dataset)
    popular = tb.build_popularity(train, n_items)
    train_sorted = train.sort_values(["user_id", "timestamp"], kind="stable")
    item_freq = np.zeros(n_items, dtype=np.float32)
    counts = train["item_id"].value_counts()
    item_freq[counts.index.to_numpy(dtype=np.int64) - 1] = counts.to_numpy(dtype=np.float32)

    inputs, eval_users, targets = tb.extract_eval_users_and_targets(test)
    if eval_users.size == 0:
        print(f"[skip] {variant}: no eval users")
        return
    q_binary, q_session, q_last, seen_lists = tb.build_query_matrices(
        inputs=inputs,
        eval_users=eval_users,
        n_items=n_items,
        session_len=10,
        filter_seen=filter_seen,
    )

    max_k = max(ks)
    markov_mat = None
    covis_mat = None
    itemknn_mat = None
    easer_mat = None
    if "Markov" in missing_models:
        markov_mat = tb.build_transition_matrix(train_sorted, n_items, neighbors=200)
    if "SKNN" in missing_models or "ItemKNN" in missing_models:
        covis_mat = tb.build_covisit_matrix(train_sorted, n_items, window=5, neighbors=200)
    if "ItemKNN" in missing_models:
        itemknn_mat = tb.prune_topk_rows(tb.normalize_cosine(covis_mat, item_freq), 200)
    if "EASER" in missing_models:
        easer_mat = tb.train_easer(train=train, n_items=n_items, candidates=50, reg=300.0)

    for model in missing_models:
        if model == "MostPopular":
            recs = tb.topk_from_popularity(len(eval_users), seen_lists, popular, max_k, filter_seen)
        elif model == "Markov":
            recs = tb.batch_predict_sparse(q_last, markov_mat, seen_lists, popular, max_k, filter_seen, 5000)
        elif model == "SKNN":
            recs = tb.batch_predict_sparse(q_session, covis_mat, seen_lists, popular, max_k, filter_seen, 5000)
        elif model == "ItemKNN":
            recs = tb.batch_predict_sparse(q_binary, itemknn_mat, seen_lists, popular, max_k, filter_seen, 5000)
        elif model == "EASER":
            recs = tb.batch_predict_sparse(q_binary, easer_mat, seen_lists, popular, max_k, filter_seen, 5000)
        else:
            raise ValueError(model)
        per_user = per_user_from_recs(eval_users=eval_users, targets=targets, recs=recs, ks=ks)
        save_per_user(variant, model, per_user)
        print(f"[done] {variant} {model} rows={len(per_user)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-path", default=str(ROOT / "results" / "matched_v1v2_pairs.csv"))
    parser.add_argument("--guarded-only", action="store_true")
    parser.add_argument("--pcore-only", action="store_true")
    parser.add_argument("--exclude-p0", action="store_true")
    parser.add_argument("--all-deterministic", action="store_true")
    parser.add_argument("--top-k", default="1,5,10,20")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    ks = [int(x) for x in args.top_k.split(",") if x]
    pairs = needed_pairs(
        Path(args.matched_path),
        args.guarded_only,
        args.pcore_only,
        args.exclude_p0,
        args.all_deterministic,
    )
    print(f"Exporting deterministic per-user metrics for {sum(len(v) for v in pairs.values())} variant/model pairs")
    for variant, models in sorted(pairs.items()):
        export_variant(variant, models, ks, args.skip_existing)


if __name__ == "__main__":
    main()
