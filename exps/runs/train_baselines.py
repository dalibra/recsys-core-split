"""
Run lightweight classical baselines on prepared split variants.

Outputs are written in the same metrics format as `runs/train.py`:
`metric_name`, `metric_value` CSV files under:
data/results/global_timesplit/<split_subtype>/<variant>/<q>/<Model>/test_last/
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.prepr import last_item_split


def q_tag(quantile: float) -> str:
    return "q0" + str(quantile)[2:]


def parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x) for x in parse_csv_list(value)]


def safe_load_dataset_filter_seen(repo_root: str, dataset: str) -> bool:
    cfg_path = os.path.join(repo_root, "runs", "configs", "dataset", f"{dataset}.yaml")
    if not os.path.exists(cfg_path):
        return True
    cfg = OmegaConf.load(cfg_path)
    return bool(cfg.get("filter_seen", True))


def find_variants(split_root: str, datasets: Sequence[str], variant_prefix: str) -> List[str]:
    if not os.path.exists(split_root):
        raise FileNotFoundError(f"Split root not found: {split_root}")
    found = []
    for variant in sorted(os.listdir(split_root)):
        v_path = os.path.join(split_root, variant)
        if not os.path.isdir(v_path):
            continue
        if datasets and not any(variant.startswith(d + "__") for d in datasets):
            continue
        if variant_prefix and not variant.startswith(variant_prefix):
            continue
        found.append(variant)
    return found


def model_result_exists(
    data_path: str,
    split_subtype: str,
    variant: str,
    q: str,
    model: str,
    prefix: str,
) -> bool:
    out_dir = os.path.join(
        data_path, "results", "global_timesplit", split_subtype, variant, q, model, prefix
    )
    if not os.path.exists(out_dir):
        return False
    return any(name.endswith(".csv") for name in os.listdir(out_dir))


def variant_has_test_data(split_dir: str) -> bool:
    stats_path = os.path.join(split_dir, "variant_stats.json")
    if not os.path.exists(stats_path):
        return True
    try:
        stats = pd.read_json(stats_path, typ="series")
        train_n = int(stats.get("train", {}).get("n_interactions", 0))
        test_n = int(stats.get("test", {}).get("n_interactions", 0))
        return train_n > 0 and test_n > 0
    except Exception:
        return True


def load_split(split_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    validation = pd.read_csv(os.path.join(split_dir, "validation.csv"))
    test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    return train, validation, test


def build_popularity(train: pd.DataFrame, n_items: int) -> np.ndarray:
    counts = (
        train.groupby("item_id", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "item_id"], ascending=[False, True], kind="stable")
    )
    popular = counts["item_id"].to_numpy(dtype=np.int32)
    if len(popular) < n_items:
        all_items = np.arange(1, n_items + 1, dtype=np.int32)
        missing = np.setdiff1d(all_items, popular, assume_unique=False)
        popular = np.concatenate([popular, missing.astype(np.int32, copy=False)])
    return popular


def build_binary_user_item(train: pd.DataFrame, n_items: int) -> sparse.csr_matrix:
    uniq = train[["user_id", "item_id"]].drop_duplicates(ignore_index=True)
    if uniq.empty:
        return sparse.csr_matrix((0, n_items), dtype=np.float32)
    user_codes, _ = pd.factorize(uniq["user_id"], sort=True)
    item_cols = uniq["item_id"].to_numpy(dtype=np.int64) - 1
    vals = np.ones(len(uniq), dtype=np.float32)
    n_users = int(user_codes.max()) + 1
    return sparse.csr_matrix((vals, (user_codes, item_cols)), shape=(n_users, n_items), dtype=np.float32)


def train_easer(
    train: pd.DataFrame,
    n_items: int,
    candidates: int,
    reg: float,
    progress_every: int = 4000,
) -> sparse.csr_matrix:
    x_bin = build_binary_user_item(train, n_items)
    if x_bin.shape[0] == 0 or x_bin.shape[1] == 0:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)

    cooc = (x_bin.T @ x_bin).tocsr().astype(np.float32)
    cooc.setdiag(0)
    cooc.eliminate_zeros()

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    eps = np.float32(1e-6)

    for item in range(n_items):
        start = cooc.indptr[item]
        end = cooc.indptr[item + 1]
        if end <= start:
            if progress_every > 0 and (item + 1) % progress_every == 0:
                print(f"[EASER] processed {item + 1}/{n_items} items")
            continue

        neigh = cooc.indices[start:end]
        b_full = cooc.data[start:end]
        if neigh.size > candidates:
            pick = np.argpartition(b_full, -candidates)[-candidates:]
            pick = pick[np.argsort(-b_full[pick], kind="stable")]
            neigh = neigh[pick]
            b = b_full[pick]
        else:
            b = b_full

        m = neigh.size
        if m == 0:
            if progress_every > 0 and (item + 1) % progress_every == 0:
                print(f"[EASER] processed {item + 1}/{n_items} items")
            continue

        gram = cooc[neigh][:, neigh].toarray().astype(np.float32, copy=False)
        gram[np.arange(m), np.arange(m)] += np.float32(reg)
        try:
            w = np.linalg.solve(gram, b.astype(np.float32, copy=False))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(gram + np.eye(m, dtype=np.float32) * eps, b, rcond=None)[0]

        rows.extend(neigh.tolist())
        cols.extend([item] * m)
        vals.extend(w.astype(np.float32, copy=False).tolist())

        if progress_every > 0 and (item + 1) % progress_every == 0:
            print(f"[EASER] processed {item + 1}/{n_items} items")

    if not rows:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


def prune_topk_rows(matrix: sparse.csr_matrix, topk: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    if topk <= 0:
        return matrix

    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    n_rows = matrix.shape[0]

    out_indptr = np.zeros(n_rows + 1, dtype=np.int64)
    out_indices: List[np.ndarray] = []
    out_data: List[np.ndarray] = []

    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        nnz = end - start
        if nnz <= 0:
            out_indptr[row + 1] = out_indptr[row]
            continue

        row_idx = indices[start:end]
        row_data = data[start:end]
        if nnz > topk:
            pick = np.argpartition(row_data, -topk)[-topk:]
            pick = pick[np.argsort(-row_data[pick], kind="stable")]
            row_idx = row_idx[pick]
            row_data = row_data[pick]
        out_indices.append(row_idx.astype(np.int32, copy=False))
        out_data.append(row_data.astype(np.float32, copy=False))
        out_indptr[row + 1] = out_indptr[row] + len(row_idx)

    if out_indices:
        idx = np.concatenate(out_indices)
        vals = np.concatenate(out_data)
    else:
        idx = np.array([], dtype=np.int32)
        vals = np.array([], dtype=np.float32)

    return sparse.csr_matrix((vals, idx, out_indptr), shape=matrix.shape, dtype=np.float32)


def build_transition_matrix(
    train_sorted: pd.DataFrame,
    n_items: int,
    neighbors: int,
) -> sparse.csr_matrix:
    prev = train_sorted.groupby("user_id", sort=False)["item_id"].shift(1)
    curr = train_sorted["item_id"]
    mask = prev.notna()
    if not mask.any():
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    rows = prev.loc[mask].to_numpy(dtype=np.int64) - 1
    cols = curr.loc[mask].to_numpy(dtype=np.int64) - 1
    vals = np.ones(rows.shape[0], dtype=np.float32)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    return prune_topk_rows(mat, neighbors)


def build_covisit_matrix(
    train_sorted: pd.DataFrame,
    n_items: int,
    window: int,
    neighbors: int,
) -> sparse.csr_matrix:
    if window <= 0:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)

    grouped = train_sorted.groupby("user_id", sort=False)["item_id"]
    curr = train_sorted["item_id"]
    covis = sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    for lag in range(1, window + 1):
        prev = grouped.shift(lag)
        mask = prev.notna()
        if not mask.any():
            continue
        rows = prev.loc[mask].to_numpy(dtype=np.int64) - 1
        cols = curr.loc[mask].to_numpy(dtype=np.int64) - 1
        vals = np.full(rows.shape[0], 1.0 / float(lag), dtype=np.float32)
        part = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        covis = covis + part + part.T
    return prune_topk_rows(covis, neighbors)


def normalize_cosine(matrix: sparse.csr_matrix, item_freq: np.ndarray) -> sparse.csr_matrix:
    coo = matrix.tocoo(copy=True)
    denom = np.sqrt(item_freq[coo.row] * item_freq[coo.col]).astype(np.float32)
    denom = np.maximum(denom, np.float32(1e-12))
    vals = coo.data.astype(np.float32, copy=False) / denom
    out = sparse.csr_matrix((vals, (coo.row, coo.col)), shape=matrix.shape, dtype=np.float32)
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def extract_eval_users_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    inputs, gt = last_item_split(df)
    if gt.empty:
        return inputs, np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    gt = gt.sort_values("user_id", kind="stable").reset_index(drop=True)
    users = gt["user_id"].to_numpy(dtype=np.int64)
    targets = gt["item_id"].to_numpy(dtype=np.int32)
    return inputs, users, targets


def build_query_matrices(
    inputs: pd.DataFrame,
    eval_users: np.ndarray,
    n_items: int,
    session_len: int,
    filter_seen: bool,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, List[np.ndarray]]:
    inputs_sorted = inputs.sort_values(["user_id", "timestamp"], kind="stable")
    user_hist: Dict[int, List[int]] = (
        inputs_sorted.groupby("user_id", sort=False)["item_id"].agg(list).to_dict()
    )
    n_users = len(eval_users)

    rows_binary: List[int] = []
    cols_binary: List[int] = []
    data_binary: List[float] = []

    rows_session: List[int] = []
    cols_session: List[int] = []
    data_session: List[float] = []

    rows_last: List[int] = []
    cols_last: List[int] = []
    data_last: List[float] = []

    seen_lists: List[np.ndarray] = []

    for row, user_id in enumerate(eval_users.tolist()):
        hist = user_hist.get(int(user_id), [])
        if not hist:
            seen_lists.append(np.array([], dtype=np.int32))
            continue

        hist_arr = np.asarray(hist, dtype=np.int32)
        uniq_hist = np.unique(hist_arr)
        seen_lists.append(uniq_hist if filter_seen else np.array([], dtype=np.int32))

        rows_binary.extend([row] * len(uniq_hist))
        cols_binary.extend((uniq_hist - 1).tolist())
        data_binary.extend([1.0] * len(uniq_hist))

        session = hist_arr[-session_len:] if session_len > 0 else hist_arr
        if session.size > 0:
            weights = np.linspace(
                1.0 / float(session.size),
                1.0,
                num=session.size,
                endpoint=True,
                dtype=np.float32,
            )
            rows_session.extend([row] * session.size)
            cols_session.extend((session - 1).tolist())
            data_session.extend(weights.tolist())
            rows_last.append(row)
            cols_last.append(int(session[-1] - 1))
            data_last.append(1.0)

    shape = (n_users, n_items)
    q_binary = sparse.csr_matrix(
        (np.asarray(data_binary, dtype=np.float32), (rows_binary, cols_binary)),
        shape=shape,
        dtype=np.float32,
    )
    q_session = sparse.csr_matrix(
        (np.asarray(data_session, dtype=np.float32), (rows_session, cols_session)),
        shape=shape,
        dtype=np.float32,
    )
    q_last = sparse.csr_matrix(
        (np.asarray(data_last, dtype=np.float32), (rows_last, cols_last)),
        shape=shape,
        dtype=np.float32,
    )
    return q_binary, q_session, q_last, seen_lists


def contains_sorted(sorted_arr: np.ndarray, value: int) -> bool:
    if sorted_arr.size == 0:
        return False
    pos = int(np.searchsorted(sorted_arr, value))
    return pos < sorted_arr.size and int(sorted_arr[pos]) == value


def fill_from_popularity(
    selected: List[int],
    selected_set: set,
    seen: np.ndarray,
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> None:
    if len(selected) >= max_k:
        return

    scan_limits = [1000, 5000, len(popularity)]
    for scan_limit in scan_limits:
        for item in popularity[:scan_limit]:
            item_int = int(item)
            if item_int in selected_set:
                continue
            if filter_seen and contains_sorted(seen, item_int):
                continue
            selected.append(item_int)
            selected_set.add(item_int)
            if len(selected) >= max_k:
                return


def topk_from_popularity(
    n_users: int,
    seen_lists: List[np.ndarray],
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> np.ndarray:
    recs = np.zeros((n_users, max_k), dtype=np.int32)
    if not filter_seen:
        recs[:] = popularity[:max_k]
        return recs

    for row in range(n_users):
        selected: List[int] = []
        selected_set: set = set()
        fill_from_popularity(
            selected=selected,
            selected_set=selected_set,
            seen=seen_lists[row],
            popularity=popularity,
            max_k=max_k,
            filter_seen=filter_seen,
        )
        if selected:
            recs[row, : len(selected)] = np.asarray(selected, dtype=np.int32)
    return recs


def topk_from_sparse_rows(
    scores: sparse.csr_matrix,
    seen_lists: List[np.ndarray],
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
) -> np.ndarray:
    scores = scores.tocsr()
    n_rows = scores.shape[0]
    recs = np.zeros((n_rows, max_k), dtype=np.int32)

    for row in range(n_rows):
        start = scores.indptr[row]
        end = scores.indptr[row + 1]
        idx = scores.indices[start:end]
        vals = scores.data[start:end]
        seen = seen_lists[row]

        if idx.size > 0 and filter_seen and seen.size > 0:
            keep = ~np.isin(idx + 1, seen, assume_unique=False)
            idx = idx[keep]
            vals = vals[keep]

        selected: List[int] = []
        selected_set: set = set()
        if idx.size > 0:
            take = min(max_k, idx.size)
            top_pos = np.argpartition(vals, -take)[-take:]
            top_pos = top_pos[np.argsort(-vals[top_pos], kind="stable")]
            for pos in top_pos.tolist():
                item_id = int(idx[pos] + 1)
                if item_id in selected_set:
                    continue
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) >= max_k:
                    break

        fill_from_popularity(
            selected=selected,
            selected_set=selected_set,
            seen=seen,
            popularity=popularity,
            max_k=max_k,
            filter_seen=filter_seen,
        )

        if selected:
            recs[row, : len(selected)] = np.asarray(selected, dtype=np.int32)
    return recs


def batch_predict_sparse(
    query: sparse.csr_matrix,
    similarity: sparse.csr_matrix,
    seen_lists: List[np.ndarray],
    popularity: np.ndarray,
    max_k: int,
    filter_seen: bool,
    batch_size: int,
) -> np.ndarray:
    n_users = query.shape[0]
    out = np.zeros((n_users, max_k), dtype=np.int32)
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        scores = query[start:end] @ similarity
        batch_seen = seen_lists[start:end]
        out[start:end] = topk_from_sparse_rows(
            scores=scores,
            seen_lists=batch_seen,
            popularity=popularity,
            max_k=max_k,
            filter_seen=filter_seen,
        )
    return out


def compute_metrics_from_topk(
    recs: np.ndarray,
    targets: np.ndarray,
    ks: Sequence[int],
    n_items: int,
) -> Dict[str, float]:
    if recs.shape[0] != targets.shape[0]:
        raise ValueError("Recommendation and target sizes do not match.")

    n_users = recs.shape[0]
    metrics: Dict[str, float] = {}
    if n_users == 0:
        for k in ks:
            metrics[f"NDCG@{k}"] = float("nan")
            metrics[f"HitRate@{k}"] = float("nan")
            metrics[f"MRR@{k}"] = float("nan")
            metrics[f"Coverage@{k}"] = float("nan")
        return metrics

    target_col = targets.reshape(-1, 1)
    for k in ks:
        topk = recs[:, :k]
        hits = topk == target_col
        has_hit = hits.any(axis=1)
        ranks = np.where(has_hit, np.argmax(hits, axis=1) + 1, 0)

        hr = float(has_hit.mean())
        ndcg_vec = np.zeros(n_users, dtype=np.float64)
        mrr_vec = np.zeros(n_users, dtype=np.float64)
        hit_idx = ranks > 0
        if np.any(hit_idx):
            ndcg_vec[hit_idx] = 1.0 / np.log2(ranks[hit_idx] + 1)
            mrr_vec[hit_idx] = 1.0 / ranks[hit_idx]
        ndcg = float(ndcg_vec.mean())
        mrr = float(mrr_vec.mean())
        unique_items = np.unique(topk[topk > 0])
        cov = float(unique_items.size) / float(n_items) if n_items > 0 else float("nan")

        metrics[f"NDCG@{k}"] = ndcg
        metrics[f"HitRate@{k}"] = hr
        metrics[f"MRR@{k}"] = mrr
        metrics[f"Coverage@{k}"] = cov
    return metrics


def save_metrics(
    data_path: str,
    split_subtype: str,
    variant: str,
    q: str,
    model: str,
    prefix: str,
    metrics: Dict[str, float],
    filename_tag: str,
) -> None:
    out_dir = os.path.join(
        data_path, "results", "global_timesplit", split_subtype, variant, q, model, prefix
    )
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"metric_name": k, "metric_value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{filename_tag}.csv"), index=False)


def run_for_variant(
    variant: str,
    split_dir: str,
    data_path: str,
    split_subtype: str,
    q: str,
    models: Sequence[str],
    ks: Sequence[int],
    session_len: int,
    cooc_window: int,
    neighbors: int,
    markov_neighbors: int,
    easer_candidates: int,
    easer_reg: float,
    batch_size: int,
    skip_existing: bool,
    include_validation: bool,
    filter_seen: bool,
) -> None:
    if not variant_has_test_data(split_dir):
        print(f"[skip:no_data] {variant}")
        return

    wanted = []
    eval_prefixes = ["test_last"] + (["val_last"] if include_validation else [])
    for model in models:
        for prefix in eval_prefixes:
            if skip_existing and model_result_exists(
                data_path, split_subtype, variant, q, model, prefix
            ):
                continue
            wanted.append((model, prefix))
    if not wanted:
        print(f"[skip:existing] {variant}")
        return

    t0 = time.time()
    train, validation, test = load_split(split_dir)
    n_items = int(max(train["item_id"].max(), validation["item_id"].max(), test["item_id"].max()))
    popular = build_popularity(train, n_items)
    train_sorted = train.sort_values(["user_id", "timestamp"], kind="stable")
    item_freq = np.zeros(n_items, dtype=np.float32)
    counts = train["item_id"].value_counts()
    item_freq[counts.index.to_numpy(dtype=np.int64) - 1] = counts.to_numpy(dtype=np.float32)

    need_markov = any(model == "Markov" for model, _ in wanted)
    need_knn = any(model in ("SKNN", "ItemKNN") for model, _ in wanted)
    need_easer = any(model == "EASER" for model, _ in wanted)

    markov_mat = None
    covis_mat = None
    itemknn_mat = None
    easer_mat = None
    if need_markov:
        markov_mat = build_transition_matrix(train_sorted, n_items, markov_neighbors)
    if need_knn:
        covis_mat = build_covisit_matrix(train_sorted, n_items, cooc_window, neighbors)
        itemknn_mat = prune_topk_rows(normalize_cosine(covis_mat, item_freq), neighbors)
    if need_easer:
        easer_mat = train_easer(
            train=train,
            n_items=n_items,
            candidates=easer_candidates,
            reg=easer_reg,
        )

    split_frames = {"test_last": test, "val_last": validation}
    for prefix in eval_prefixes:
        split_df = split_frames[prefix]
        inputs, eval_users, targets = extract_eval_users_and_targets(split_df)
        if eval_users.size == 0:
            print(f"[skip:{prefix}:empty_gt] {variant}")
            continue

        max_k = int(max(ks))
        q_binary, q_session, q_last, seen_lists = build_query_matrices(
            inputs=inputs,
            eval_users=eval_users,
            n_items=n_items,
            session_len=session_len,
            filter_seen=filter_seen,
        )

        for model in models:
            if skip_existing and model_result_exists(
                data_path, split_subtype, variant, q, model, prefix
            ):
                continue

            if model == "MostPopular":
                recs = topk_from_popularity(
                    n_users=len(eval_users),
                    seen_lists=seen_lists,
                    popularity=popular,
                    max_k=max_k,
                    filter_seen=filter_seen,
                )
            elif model == "Markov":
                if markov_mat is None:
                    raise RuntimeError("Markov matrix is not initialized.")
                recs = batch_predict_sparse(
                    query=q_last,
                    similarity=markov_mat,
                    seen_lists=seen_lists,
                    popularity=popular,
                    max_k=max_k,
                    filter_seen=filter_seen,
                    batch_size=batch_size,
                )
            elif model == "SKNN":
                if covis_mat is None:
                    raise RuntimeError("Co-visitation matrix is not initialized.")
                recs = batch_predict_sparse(
                    query=q_session,
                    similarity=covis_mat,
                    seen_lists=seen_lists,
                    popularity=popular,
                    max_k=max_k,
                    filter_seen=filter_seen,
                    batch_size=batch_size,
                )
            elif model == "ItemKNN":
                if itemknn_mat is None:
                    raise RuntimeError("ItemKNN matrix is not initialized.")
                recs = batch_predict_sparse(
                    query=q_binary,
                    similarity=itemknn_mat,
                    seen_lists=seen_lists,
                    popularity=popular,
                    max_k=max_k,
                    filter_seen=filter_seen,
                    batch_size=batch_size,
                )
            elif model == "EASER":
                if easer_mat is None:
                    raise RuntimeError("EASER matrix is not initialized.")
                recs = batch_predict_sparse(
                    query=q_binary,
                    similarity=easer_mat,
                    seen_lists=seen_lists,
                    popularity=popular,
                    max_k=max_k,
                    filter_seen=filter_seen,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unknown baseline model: {model}")

            m = compute_metrics_from_topk(recs=recs, targets=targets, ks=ks, n_items=n_items)
            prefixed = {f"{prefix}_{k}": v for k, v in m.items()}
            save_metrics(
                data_path=data_path,
                split_subtype=split_subtype,
                variant=variant,
                q=q,
                model=model,
                prefix=prefix,
                metrics=prefixed,
                filename_tag=f"baseline_{model.lower()}",
            )
            print(
                f"[done] {variant} {prefix} {model} "
                f"NDCG@20={m.get('NDCG@20', float('nan')):.6f}"
            )

    print(f"[variant_done] {variant} elapsed={time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical baselines on split variants.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split-subtype", default="val_last_train_item")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--variant-prefix", default="")
    parser.add_argument("--models", default="MostPopular,Markov,SKNN,ItemKNN,EASER")
    parser.add_argument("--top-k", default="1,5,10,20")
    parser.add_argument("--session-len", type=int, default=10)
    parser.add_argument("--cooc-window", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--markov-neighbors", type=int, default=200)
    parser.add_argument("--easer-candidates", type=int, default=50)
    parser.add_argument("--easer-reg", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--include-validation", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--max-variants", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = args.data_path or os.environ.get(
        "SEQ_SPLITS_DATA_PATH", os.path.join(repo_root, "data")
    )
    os.environ["SEQ_SPLITS_DATA_PATH"] = data_path

    q = q_tag(args.quantile)
    split_root = os.path.join(
        data_path, "splitted", "global_timesplit", args.split_subtype
    )
    datasets = parse_csv_list(args.datasets)
    variants = find_variants(split_root, datasets, args.variant_prefix)
    if not variants:
        raise RuntimeError("No variants found.")

    models = parse_csv_list(args.models)
    ks = parse_int_list(args.top_k)
    if not ks:
        raise ValueError("top-k list is empty.")

    print(
        f"Running baselines for {len(variants)} variants | "
        f"models={models} | k={ks} | q={q}"
    )
    processed = 0
    for variant in variants:
        split_dir = os.path.join(split_root, variant, q)
        if not os.path.exists(split_dir):
            continue
        dataset = variant.split("__")[0]
        filter_seen = safe_load_dataset_filter_seen(repo_root, dataset)
        run_for_variant(
            variant=variant,
            split_dir=split_dir,
            data_path=data_path,
            split_subtype=args.split_subtype,
            q=q,
            models=models,
            ks=ks,
            session_len=args.session_len,
            cooc_window=args.cooc_window,
            neighbors=args.neighbors,
            markov_neighbors=args.markov_neighbors,
            easer_candidates=args.easer_candidates,
            easer_reg=args.easer_reg,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            include_validation=args.include_validation,
            filter_seen=filter_seen,
        )
        processed += 1
        if args.max_variants is not None and processed >= args.max_variants:
            break


if __name__ == "__main__":
    main()
