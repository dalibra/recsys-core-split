"""Beauty p=10 SASRec/Markov common-task probe.

This evaluation keeps the raw target events, raw input events, and train-item
candidate universe identical across V1/V2, then maps the same raw task into
each arm's native encoded item space before scoring existing checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
DATA = EXPS / "data"
OUT = ROOT / "results"
SPLIT_ROOT = DATA / "splitted" / "global_timesplit" / "val_last_train_item"
RESULTS_ROOT = DATA / "results" / "global_timesplit" / "val_last_train_item"
MODEL_ROOT = EXPS / "models" / "global_timesplit" / "val_last_train_item"
Q_TAG = "q09"
DATASET = "Beauty"
P = 10
K = 20
SEEDS = [17, 23, 42, 101, 202]


sys.path.insert(0, str(EXPS / "runs"))
sys.path.insert(0, str(EXPS))
import train_baselines as tb  # noqa: E402
from src.datasets import CausalLMPredictionDataset, PaddingCollateFn  # noqa: E402
from src.models import SASRec  # noqa: E402
from src.modules import SeqRec  # noqa: E402
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if order == "V1":
        filtered = apply_filter(base, spec)
        train, validation, test, _ = split_global_time_fixed(
            filtered,
            time_threshold=threshold,
            validation_type="last_train_item",
            validation_quantile=0.9,
            validation_size=1024,
            random_state=17,
        )
    elif order == "V2":
        train_raw, validation_raw, test_raw, _ = split_global_time_fixed(
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
        validation = validation_raw[
            validation_raw["user_id"].isin(train_users)
            & validation_raw["item_id"].isin(train_items)
        ]
        test = test_raw[
            test_raw["user_id"].isin(train_users)
            & test_raw["item_id"].isin(train_items)
        ]
        validation = filter_short_sequences(validation, min_len=2)
        test = filter_short_sequences(test, min_len=2)
    else:
        raise ValueError(order)
    train = filter_short_sequences(train, min_len=2)
    return train.copy(), validation.copy(), test.copy()


def encode_with_maps(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[Dict[object, int], Dict[object, int], int]:
    users = pd.Index(
        pd.concat([train["user_id"], validation["user_id"], test["user_id"]], ignore_index=True)
        .unique()
    ).sort_values()
    items = pd.Index(
        pd.concat([train["item_id"], validation["item_id"], test["item_id"]], ignore_index=True)
        .unique()
    ).sort_values()
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx + 1 for idx, i in enumerate(items)}
    return user_map, item_map, len(items)


def encode_frame(frame: pd.DataFrame, user_map: Dict[object, int], item_map: Dict[object, int]) -> pd.DataFrame:
    enc = frame.copy()
    enc["raw_user_id"] = enc["user_id"]
    enc["raw_item_id"] = enc["item_id"]
    enc["user_id"] = enc["user_id"].map(user_map)
    enc["item_id"] = enc["item_id"].map(item_map)
    if enc["user_id"].isna().any() or enc["item_id"].isna().any():
        raise ValueError("Common task contains a user/item missing from a native map")
    enc["user_id"] = enc["user_id"].astype("int64")
    enc["item_id"] = enc["item_id"].astype("int64")
    return enc


def last_inputs_targets(test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = test.sort_values(["user_id", "timestamp"], kind="stable")
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    eligible = ordered[sizes >= 2]
    gt_idx = eligible.groupby("user_id", sort=False).tail(1).index
    gt = eligible.loc[gt_idx, ["user_id", "item_id", "timestamp"]].copy()
    inputs = eligible.drop(index=gt_idx).copy()
    return inputs, gt


def frame_keys(frame: pd.DataFrame) -> set:
    return set(map(tuple, frame[["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)))


def select_common_task(parts: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    raw_candidate_intersection = set(parts["V1"]["train"]["item_id"].unique()) & set(
        parts["V2"]["train"]["item_id"].unique()
    )
    key_cols = ["user_id", "item_id", "timestamp"]
    gt_keys = frame_keys(parts["V1"]["gt"]) & frame_keys(parts["V2"]["gt"])
    gt_keys = {key for key in gt_keys if key[1] in raw_candidate_intersection}
    input_keys = frame_keys(parts["V1"]["inputs"]) & frame_keys(parts["V2"]["inputs"])
    input_keys = {key for key in input_keys if key[1] in raw_candidate_intersection}

    gt = parts["V1"]["gt"][parts["V1"]["gt"][key_cols].apply(tuple, axis=1).isin(gt_keys)].copy()
    inputs = parts["V1"]["inputs"][
        parts["V1"]["inputs"][key_cols].apply(tuple, axis=1).isin(input_keys)
    ].copy()

    target_by_user = gt.set_index("user_id")["timestamp"].to_dict()
    inputs = inputs[inputs["user_id"].map(target_by_user).notna()]
    inputs = inputs[inputs["timestamp"] <= inputs["user_id"].map(target_by_user)]
    users_with_input = set(inputs["user_id"].unique())
    gt = gt[gt["user_id"].isin(users_with_input)].copy()
    inputs = inputs[inputs["user_id"].isin(gt["user_id"].unique())].copy()
    return inputs, gt, raw_candidate_intersection


def per_user_from_recs(eval_users: np.ndarray, raw_users: np.ndarray, targets: np.ndarray, recs: np.ndarray) -> pd.DataFrame:
    rows = []
    for row_idx, user_id in enumerate(eval_users.tolist()):
        target = int(targets[row_idx])
        ranked = recs[row_idx, :K]
        hits = np.flatnonzero(ranked == target)
        rank = int(hits[0] + 1) if hits.size else 0
        rows.append(
            {
                "user_id": int(user_id),
                "raw_user_id": raw_users[row_idx],
                "target_item_id": target,
                "rank": rank,
                "HitRate@20": 1.0 if rank and rank <= K else 0.0,
                "MRR@20": 1.0 / rank if rank and rank <= K else 0.0,
                "NDCG@20": 1.0 / np.log2(rank + 1) if rank and rank <= K else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_per_user(per_user: pd.DataFrame) -> Dict[str, float]:
    return {
        "NDCG@20": float(per_user["NDCG@20"].mean()) if len(per_user) else np.nan,
        "HitRate@20": float(per_user["HitRate@20"].mean()) if len(per_user) else np.nan,
        "MRR@20": float(per_user["MRR@20"].mean()) if len(per_user) else np.nan,
    }


def candidate_popularity(train: pd.DataFrame, n_items: int, candidates: np.ndarray) -> np.ndarray:
    popular = tb.build_popularity(train, n_items)
    cand = set(candidates.tolist())
    filtered = [int(item) for item in popular.tolist() if int(item) in cand]
    if len(filtered) < len(cand):
        filtered.extend(sorted(cand - set(filtered)))
    return np.asarray(filtered, dtype=np.int32)


def topk_markov(
    train: pd.DataFrame,
    inputs: pd.DataFrame,
    eval_users: np.ndarray,
    candidates: np.ndarray,
    n_items: int,
    filter_seen: bool,
) -> np.ndarray:
    popularity = candidate_popularity(train, n_items, candidates)
    candidate_mask = np.zeros(n_items + 1, dtype=bool)
    candidate_mask[candidates] = True
    train_sorted = train.sort_values(["user_id", "timestamp"], kind="stable")
    _, _, q_last, seen_lists = tb.build_query_matrices(
        inputs=inputs,
        eval_users=eval_users,
        n_items=n_items,
        session_len=10,
        filter_seen=filter_seen,
    )
    markov = tb.build_transition_matrix(train_sorted, n_items, neighbors=200)
    out = np.zeros((len(eval_users), K), dtype=np.int32)
    scores = (q_last @ markov).tocsr()
    for row in range(scores.shape[0]):
        start, end = scores.indptr[row], scores.indptr[row + 1]
        idx = scores.indices[start:end]
        vals = scores.data[start:end]
        seen = seen_lists[row]
        if idx.size:
            keep = candidate_mask[idx + 1]
            if filter_seen and seen.size:
                keep &= ~np.isin(idx + 1, seen, assume_unique=False)
            idx = idx[keep]
            vals = vals[keep]
        selected: List[int] = []
        selected_set: set = set()
        if idx.size:
            take = min(K, idx.size)
            top_pos = np.argpartition(vals, -take)[-take:]
            top_pos = top_pos[np.argsort(-vals[top_pos], kind="stable")]
            for pos in top_pos.tolist():
                item = int(idx[pos] + 1)
                selected.append(item)
                selected_set.add(item)
        for item in popularity:
            item_int = int(item)
            if len(selected) >= K:
                break
            if item_int in selected_set:
                continue
            if filter_seen and tb.contains_sorted(seen, item_int):
                continue
            selected.append(item_int)
            selected_set.add(item_int)
        if selected:
            out[row, : len(selected)] = np.asarray(selected[:K], dtype=np.int32)
    return out


def checkpoint_path(variant: str, seed: int) -> Path:
    return MODEL_ROOT / variant / Q_TAG / "SASRec" / f"0_32_1_1_0.1_128_256_{seed}.ckpt"


def topk_sasrec(
    variant: str,
    seed: int,
    inputs: pd.DataFrame,
    eval_users: np.ndarray,
    candidates: np.ndarray,
    n_items: int,
    filter_seen: bool,
    device: torch.device,
) -> np.ndarray:
    model = SASRec(
        item_num=n_items,
        maxlen=128,
        hidden_units=32,
        num_blocks=1,
        num_heads=1,
        dropout_rate=0.1,
    )
    module = SeqRec(model, lr=0.001, predict_top_k=K, filter_seen=filter_seen)
    ckpt = torch.load(checkpoint_path(variant, seed), map_location=device)
    module.load_state_dict(ckpt["state_dict"])
    module.to(device)
    module.eval()

    dataset = CausalLMPredictionDataset(inputs, max_length=128)
    loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=1024,
        num_workers=0,
    )
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.long, device=device)
    recs_by_user: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = module.model(batch["input_ids"], batch["attention_mask"])
            input_ids = batch["input_ids"]
            rows = torch.arange(input_ids.shape[0], dtype=torch.long, device=device)
            last_idx = (input_ids != 0).sum(axis=1) - 1
            scores = outputs[rows, last_idx, :]
            cand_scores = scores[:, candidate_tensor].clone()
            if filter_seen:
                seen_ids = batch["seen_ids"]
                for row_idx in range(seen_ids.shape[0]):
                    seen = seen_ids[row_idx][seen_ids[row_idx] != 0]
                    if seen.numel():
                        seen_mask = torch.isin(candidate_tensor, seen)
                        cand_scores[row_idx, seen_mask] = -torch.inf
            take = min(K, candidate_tensor.numel())
            _, top_pos = torch.topk(cand_scores, k=take, dim=1)
            top_items = candidate_tensor[top_pos].detach().cpu().numpy()
            users = batch["user_id"].detach().cpu().numpy()
            for user_id, items in zip(users.tolist(), top_items):
                row = np.zeros(K, dtype=np.int32)
                row[: len(items)] = items.astype(np.int32, copy=False)
                recs_by_user[int(user_id)] = row
    return np.vstack([recs_by_user[int(user)] for user in eval_users.tolist()])


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 17) -> Tuple[float, float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_probe(device_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_dataset_config(DATASET)
    raw = load_raw_data(cfg, str(DATA))
    base = base_preprocess(raw, drop_conseq_repeats=True)
    threshold = compute_time_threshold(
        base,
        0.9,
        cache_dir=str(DATA / "variants" / "time_thresholds"),
        dataset_name=cfg.name,
    )
    spec = FilterSpec(filter_type="pcore", p=P)
    parts: Dict[str, Dict[str, pd.DataFrame]] = {}
    for order in ["V1", "V2"]:
        train, validation, test = build_raw_variant(base, spec, order, threshold)
        inputs, gt = last_inputs_targets(test)
        parts[order] = {"train": train, "validation": validation, "test": test, "inputs": inputs, "gt": gt}

    common_inputs_raw, common_gt_raw, raw_candidates = select_common_task(parts)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    filter_seen = tb.safe_load_dataset_filter_seen(str(EXPS), DATASET)

    summary_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    per_user_frames: List[pd.DataFrame] = []
    per_user_lookup: Dict[Tuple[str, str, int], pd.DataFrame] = {}

    for order in ["V1", "V2"]:
        variant = f"Beauty__pcore_p10__{'v1_filter_then_split' if order == 'V1' else 'v2_split_then_filter_train'}"
        user_map, item_map, n_items = encode_with_maps(
            parts[order]["train"],
            parts[order]["validation"],
            parts[order]["test"],
        )
        train_enc = encode_frame(parts[order]["train"], user_map, item_map)
        inputs_enc = encode_frame(common_inputs_raw, user_map, item_map)
        gt_enc = encode_frame(common_gt_raw, user_map, item_map).sort_values("user_id", kind="stable")
        eval_users = gt_enc["user_id"].to_numpy(dtype=np.int64)
        raw_users = gt_enc["raw_user_id"].to_numpy()
        targets = gt_enc["item_id"].to_numpy(dtype=np.int32)
        inputs_enc = inputs_enc[inputs_enc["user_id"].isin(eval_users)].copy()
        candidates = np.asarray(sorted(item_map[item] for item in raw_candidates), dtype=np.int32)

        markov_recs = topk_markov(
            train=train_enc,
            inputs=inputs_enc,
            eval_users=eval_users,
            candidates=candidates,
            n_items=n_items,
            filter_seen=filter_seen,
        )
        markov_per_user = per_user_from_recs(eval_users, raw_users, targets, markov_recs)
        markov_per_user["order"] = order
        markov_per_user["model"] = "Markov"
        markov_per_user["seed"] = -1
        per_user_lookup[(order, "Markov", -1)] = markov_per_user
        per_user_frames.append(markov_per_user)
        summary_rows.append(
            {
                "dataset": DATASET,
                "p": P,
                "setup": "common_target_input_candidate",
                "order": order,
                "variant": variant,
                "model": "Markov",
                "seed": "",
                "eval_users": int(len(eval_users)),
                "candidate_items": int(len(candidates)),
                **summarize_per_user(markov_per_user),
            }
        )

        for seed in SEEDS:
            sas_recs = topk_sasrec(
                variant=variant,
                seed=seed,
                inputs=inputs_enc,
                eval_users=eval_users,
                candidates=candidates,
                n_items=n_items,
                filter_seen=filter_seen,
                device=device,
            )
            sas_per_user = per_user_from_recs(eval_users, raw_users, targets, sas_recs)
            sas_per_user["order"] = order
            sas_per_user["model"] = "SASRec"
            sas_per_user["seed"] = seed
            per_user_lookup[(order, "SASRec", seed)] = sas_per_user
            per_user_frames.append(sas_per_user)
            summary_rows.append(
                {
                    "dataset": DATASET,
                    "p": P,
                    "setup": "common_target_input_candidate",
                    "order": order,
                    "variant": variant,
                    "model": "SASRec",
                    "seed": seed,
                    "eval_users": int(len(eval_users)),
                    "candidate_items": int(len(candidates)),
                    **summarize_per_user(sas_per_user),
                }
            )
            merged = sas_per_user.merge(
                markov_per_user[["raw_user_id", "NDCG@20"]],
                on="raw_user_id",
                suffixes=("_sasrec", "_markov"),
            )
            mean, low, high = bootstrap_ci(
                merged["NDCG@20_sasrec"].to_numpy() - merged["NDCG@20_markov"].to_numpy()
            )
            pair_rows.append(
                {
                    "comparison": "SASRec_minus_Markov",
                    "order": order,
                    "seed": seed,
                    "eval_users": int(len(merged)),
                    "gap": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "supported_sasrec_gt_markov": bool(low > 0),
                    "supported_markov_gt_sasrec": bool(high < 0),
                }
            )

    for seed in SEEDS:
        v1 = per_user_lookup[("V1", "SASRec", seed)]
        v2 = per_user_lookup[("V2", "SASRec", seed)]
        merged = v1.merge(
            v2[["raw_user_id", "NDCG@20"]],
            on="raw_user_id",
            suffixes=("_v1", "_v2"),
        )
        mean, low, high = bootstrap_ci(
            merged["NDCG@20_v1"].to_numpy() - merged["NDCG@20_v2"].to_numpy()
        )
        pair_rows.append(
            {
                "comparison": "SASRec_V1_minus_SASRec_V2",
                "order": "common",
                "seed": seed,
                "eval_users": int(len(merged)),
                "gap": mean,
                "ci_low": low,
                "ci_high": high,
                "supported_sasrec_v1_gt_v2": bool(low > 0),
                "supported_sasrec_v2_gt_v1": bool(high < 0),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "beauty_sasrec_common_probe.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "beauty_sasrec_common_probe_pairwise.csv", index=False)
    pd.concat(per_user_frames, ignore_index=True).to_csv(
        OUT / "beauty_sasrec_common_probe_per_user.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(pair_rows).to_string(index=False))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_probe(args.device)


if __name__ == "__main__":
    main()
