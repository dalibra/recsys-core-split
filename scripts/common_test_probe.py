"""Common-test deterministic probe for filter/split order.

This is a mechanism check rather than a full leaderboard rerun. It rebuilds
raw-ID p-core variants, evaluates deterministic baselines on their native
tests, then evaluates the same train graphs on the intersection of V1/V2
last-target events with the candidate universe fixed to train-item
intersection.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
OUT = ROOT / "results"
MODELS = ["MostPopular", "Markov", "SKNN", "ItemKNN"]
SELECTED_DATASET_SPECS = [("Sports", 5), ("Movielens-1m", 5), ("Beauty", 10)]
ALL_DATASET_SPECS = [
    ("Beauty", 5),
    ("Beauty", 10),
    ("Beauty", 20),
    ("BeerAdvocate", 5),
    ("BeerAdvocate", 10),
    ("BeerAdvocate", 20),
    ("Diginetica", 5),
    ("Diginetica", 10),
    ("Movielens-1m", 5),
    ("Movielens-1m", 10),
    ("Movielens-1m", 20),
    ("Sports", 5),
    ("YooChoose", 5),
    ("YooChoose", 10),
]
K = 20

sys.path.insert(0, str(EXPS / "runs"))
sys.path.insert(0, str(EXPS))
import train_baselines as tb  # noqa: E402
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


def build_raw_variant(
    base: pd.DataFrame,
    spec: FilterSpec,
    order: str,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if order == "V1":
        filtered = apply_filter(base, spec)
        train, _, test, _ = split_global_time_fixed(
            filtered,
            time_threshold=threshold,
            validation_type="last_train_item",
            validation_quantile=0.9,
            validation_size=1024,
            random_state=17,
        )
    elif order == "V2":
        train_raw, _, test_raw, _ = split_global_time_fixed(
            base,
            time_threshold=threshold,
            validation_type="last_train_item",
            validation_quantile=0.9,
            validation_size=1024,
            random_state=17,
        )
        train = apply_filter(train_raw, spec)
        train_users = set(train["user_id"].unique())
        train_items = set(train["item_id"].unique())
        test = test_raw[test_raw["user_id"].isin(train_users) & test_raw["item_id"].isin(train_items)]
        test = filter_short_sequences(test, min_len=2)
    else:
        raise ValueError(order)
    return filter_short_sequences(train, min_len=2).copy(), test.copy()


def last_inputs_targets(test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = test.sort_values(["user_id", "timestamp"], kind="stable")
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    eligible = ordered[sizes >= 2]
    gt_idx = eligible.groupby("user_id", sort=False).tail(1).index
    gt = eligible.loc[gt_idx, ["user_id", "item_id", "timestamp"]].copy()
    inputs = eligible.drop(index=gt_idx).copy()
    return inputs, gt


def frame_keys(gt: pd.DataFrame) -> set:
    return set(map(tuple, gt[["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)))


def encode_frames(frames: Iterable[pd.DataFrame]) -> Tuple[List[pd.DataFrame], Dict[object, int], Dict[object, int]]:
    frames = list(frames)
    users = pd.Index(pd.concat([f["user_id"] for f in frames if not f.empty], ignore_index=True).unique()).sort_values()
    items = pd.Index(pd.concat([f["item_id"] for f in frames if not f.empty], ignore_index=True).unique()).sort_values()
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx + 1 for idx, i in enumerate(items)}

    out = []
    for frame in frames:
        enc = frame.copy()
        enc["user_id"] = enc["user_id"].map(user_map).astype("int64")
        enc["item_id"] = enc["item_id"].map(item_map).astype("int64")
        out.append(enc)
    return out, user_map, item_map


def candidate_popularity(train: pd.DataFrame, n_items: int, candidates: np.ndarray) -> np.ndarray:
    popular = tb.build_popularity(train, n_items)
    cand = set(candidates.tolist())
    filtered = [int(item) for item in popular.tolist() if int(item) in cand]
    if len(filtered) < len(cand):
        filtered.extend(sorted(cand - set(filtered)))
    return np.asarray(filtered, dtype=np.int32)


def fill_candidate_popular(
    selected: List[int],
    selected_set: set,
    seen: np.ndarray,
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> None:
    if len(selected) >= max_k:
        return
    for item in popularity:
        item_int = int(item)
        if item_int in selected_set:
            continue
        if filter_seen and tb.contains_sorted(seen, item_int):
            continue
        selected.append(item_int)
        selected_set.add(item_int)
        if len(selected) >= max_k:
            return


def topk_pop_candidate(
    n_users: int,
    seen_lists: List[np.ndarray],
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> np.ndarray:
    recs = np.zeros((n_users, max_k), dtype=np.int32)
    for row in range(n_users):
        selected: List[int] = []
        fill_candidate_popular(selected, set(), seen_lists[row], popularity, max_k, filter_seen)
        if selected:
            recs[row, : len(selected)] = np.asarray(selected, dtype=np.int32)
    return recs


def topk_sparse_candidate(
    scores: sparse.csr_matrix,
    seen_lists: List[np.ndarray],
    candidate_mask: np.ndarray,
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> np.ndarray:
    scores = scores.tocsr()
    recs = np.zeros((scores.shape[0], max_k), dtype=np.int32)
    for row in range(scores.shape[0]):
        start, end = scores.indptr[row], scores.indptr[row + 1]
        idx = scores.indices[start:end]
        vals = scores.data[start:end]
        if idx.size:
            keep = candidate_mask[idx + 1]
            if filter_seen and seen_lists[row].size:
                keep &= ~np.isin(idx + 1, seen_lists[row], assume_unique=False)
            idx = idx[keep]
            vals = vals[keep]

        selected: List[int] = []
        selected_set: set = set()
        if idx.size:
            take = min(max_k, idx.size)
            top_pos = np.argpartition(vals, -take)[-take:]
            top_pos = top_pos[np.argsort(-vals[top_pos], kind="stable")]
            for pos in top_pos.tolist():
                item_id = int(idx[pos] + 1)
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) >= max_k:
                    break
        fill_candidate_popular(selected, selected_set, seen_lists[row], popularity, max_k, filter_seen)
        if selected:
            recs[row, : len(selected)] = np.asarray(selected, dtype=np.int32)
    return recs


def batch_sparse_candidate(
    query: sparse.csr_matrix,
    similarity: sparse.csr_matrix,
    seen_lists: List[np.ndarray],
    candidate_mask: np.ndarray,
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
    batch_size: int = 5000,
) -> np.ndarray:
    out = np.zeros((query.shape[0], max_k), dtype=np.int32)
    for start in range(0, query.shape[0], batch_size):
        end = min(start + batch_size, query.shape[0])
        out[start:end] = topk_sparse_candidate(
            query[start:end] @ similarity,
            seen_lists[start:end],
            candidate_mask,
            popularity,
            max_k,
            filter_seen,
        )
    return out


def evaluate_variant(
    dataset: str,
    p: int,
    order: str,
    setup: str,
    train: pd.DataFrame,
    inputs: pd.DataFrame,
    gt: pd.DataFrame,
    candidates: np.ndarray,
) -> List[Dict[str, object]]:
    if gt.empty:
        return []
    n_items = int(max(train["item_id"].max(), inputs["item_id"].max(), gt["item_id"].max()))
    filter_seen = tb.safe_load_dataset_filter_seen(str(EXPS), dataset)
    eval_gt = gt.sort_values("user_id", kind="stable").reset_index(drop=True)
    eval_users = eval_gt["user_id"].to_numpy(dtype=np.int64)
    targets = eval_gt["item_id"].to_numpy(dtype=np.int32)
    inputs = inputs[inputs["user_id"].isin(eval_users)].copy()

    popularity = candidate_popularity(train, n_items, candidates)
    candidate_mask = np.zeros(n_items + 1, dtype=bool)
    candidate_mask[candidates] = True
    train_sorted = train.sort_values(["user_id", "timestamp"], kind="stable")
    item_freq = np.zeros(n_items, dtype=np.float32)
    counts = train["item_id"].value_counts()
    item_freq[counts.index.to_numpy(dtype=np.int64) - 1] = counts.to_numpy(dtype=np.float32)

    q_binary, q_session, q_last, seen_lists = tb.build_query_matrices(
        inputs=inputs,
        eval_users=eval_users,
        n_items=n_items,
        session_len=10,
        filter_seen=filter_seen,
    )
    markov = tb.build_transition_matrix(train_sorted, n_items, neighbors=200)
    covis = tb.build_covisit_matrix(train_sorted, n_items, window=5, neighbors=200)
    itemknn = tb.prune_topk_rows(tb.normalize_cosine(covis, item_freq), 200)

    recs_by_model = {
        "MostPopular": topk_pop_candidate(len(eval_users), seen_lists, popularity, K, filter_seen),
        "Markov": batch_sparse_candidate(q_last, markov, seen_lists, candidate_mask, popularity, K, filter_seen),
        "SKNN": batch_sparse_candidate(q_session, covis, seen_lists, candidate_mask, popularity, K, filter_seen),
        "ItemKNN": batch_sparse_candidate(q_binary, itemknn, seen_lists, candidate_mask, popularity, K, filter_seen),
    }

    rows = []
    for model, recs in recs_by_model.items():
        metrics = tb.compute_metrics_from_topk(recs, targets, [K], n_items=len(candidates))
        rows.append(
            {
                "dataset": dataset,
                "p": p,
                "order": order,
                "setup": setup,
                "model": model,
                "eval_users": int(len(eval_users)),
                "candidate_items": int(len(candidates)),
                "NDCG@20": metrics["NDCG@20"],
                "HitRate@20": metrics["HitRate@20"],
                "MRR@20": metrics["MRR@20"],
            }
        )
    return rows


def run_dataset(dataset: str, p: int) -> List[Dict[str, object]]:
    cfg = load_dataset_config(dataset)
    raw = load_raw_data(cfg, str(DATA))
    base = base_preprocess(raw, drop_conseq_repeats=True)
    threshold = compute_time_threshold(
        base,
        0.9,
        cache_dir=str(DATA / "variants" / "time_thresholds"),
        dataset_name=cfg.name,
    )
    spec = FilterSpec(filter_type="pcore", p=p)
    raw_variants = {
        order: build_raw_variant(base, spec, order, threshold) for order in ["V1", "V2"]
    }
    split_parts = {}
    for order, (train, test) in raw_variants.items():
        inputs, gt = last_inputs_targets(test)
        split_parts[order] = {"train": train, "test": test, "inputs": inputs, "gt": gt}

    common_keys = frame_keys(split_parts["V1"]["gt"]) & frame_keys(split_parts["V2"]["gt"])
    raw_candidate_intersection = set(split_parts["V1"]["train"]["item_id"].unique()) & set(
        split_parts["V2"]["train"]["item_id"].unique()
    )
    common_keys = {key for key in common_keys if key[1] in raw_candidate_intersection}
    key_cols = ["user_id", "item_id", "timestamp"]

    frames_to_encode = []
    for order in ["V1", "V2"]:
        part = split_parts[order]
        native_gt = part["gt"][part["gt"]["item_id"].isin(part["train"]["item_id"].unique())].copy()
        native_inputs = part["inputs"][part["inputs"]["user_id"].isin(native_gt["user_id"])].copy()
        common_gt = part["gt"][
            part["gt"][key_cols].apply(tuple, axis=1).isin(common_keys)
        ].copy()
        common_inputs = part["inputs"][part["inputs"]["user_id"].isin(common_gt["user_id"])].copy()
        part["native_gt"] = native_gt
        part["native_inputs"] = native_inputs
        part["common_gt"] = common_gt
        part["common_inputs"] = common_inputs
        frames_to_encode.extend([part["train"], native_inputs, native_gt, common_inputs, common_gt])

    encoded, _, item_map = encode_frames(frames_to_encode)
    idx = 0
    for order in ["V1", "V2"]:
        part = split_parts[order]
        part["train_enc"] = encoded[idx]
        part["native_inputs_enc"] = encoded[idx + 1]
        part["native_gt_enc"] = encoded[idx + 2]
        part["common_inputs_enc"] = encoded[idx + 3]
        part["common_gt_enc"] = encoded[idx + 4]
        idx += 5

    rows: List[Dict[str, object]] = []
    for order in ["V1", "V2"]:
        part = split_parts[order]
        native_candidates = np.asarray(
            sorted(item_map[item] for item in part["train"]["item_id"].unique()),
            dtype=np.int32,
        )
        common_candidates = np.asarray(
            sorted(item_map[item] for item in raw_candidate_intersection),
            dtype=np.int32,
        )
        rows.extend(
            evaluate_variant(
                dataset,
                p,
                order,
                "native",
                part["train_enc"],
                part["native_inputs_enc"],
                part["native_gt_enc"],
                native_candidates,
            )
        )
        rows.extend(
            evaluate_variant(
                dataset,
                p,
                order,
                "common_test_common_candidate",
                part["train_enc"],
                part["common_inputs_enc"],
                part["common_gt_enc"],
                common_candidates,
            )
        )
    return rows


def add_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, p, setup), sub in results.groupby(["dataset", "p", "setup"], sort=True):
        winners = {}
        for order, order_sub in sub.groupby("order", sort=True):
            best = order_sub.sort_values(["NDCG@20", "model"], ascending=[False, True]).iloc[0]
            winners[order] = str(best["model"])
        rows.append(
            {
                "dataset": dataset,
                "p": int(p),
                "setup": setup,
                "v1_winner": winners.get("V1"),
                "v2_winner": winners.get("V2"),
                "winner_flip": winners.get("V1") != winners.get("V2"),
                "eval_users_v1": int(sub[sub["order"] == "V1"]["eval_users"].max()),
                "eval_users_v2": int(sub[sub["order"] == "V2"]["eval_users"].max()),
                "candidate_items_v1": int(sub[sub["order"] == "V1"]["candidate_items"].max()),
                "candidate_items_v2": int(sub[sub["order"] == "V2"]["candidate_items"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["selected", "all"],
        default="selected",
        help="Run only the paper spotlight probes or all matched p-core p>0 settings.",
    )
    args = parser.parse_args()
    dataset_specs = ALL_DATASET_SPECS if args.scope == "all" else SELECTED_DATASET_SPECS

    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for dataset, p in dataset_specs:
        print(f"[dataset] {dataset} p={p}", flush=True)
        rows.extend(run_dataset(dataset, p))
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "common_test_probe.csv", index=False)
    summary = add_summary(results)
    summary.to_csv(OUT / "common_test_probe_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
