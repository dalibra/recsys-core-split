"""
Pipeline to generate preprocessing + split variants with fixed global time threshold.
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.filters import core_filter, min_count_filter, drop_consecutive_repeats
from src.preprocess.utils import dataset_stats, rename_cols


@dataclass
class FilterSpec:
    filter_type: str  # "pcore" or "mincount"
    p: Optional[int] = None
    u_min: Optional[int] = None
    i_min: Optional[int] = None


def _q_tag(quantile: float) -> str:
    return "q0" + str(quantile)[2:]


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def load_dataset_config(dataset_key: str) -> OmegaConf:
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "dataset", f"{dataset_key}.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    return OmegaConf.load(config_path)


def load_raw_data(cfg: OmegaConf, data_path: str) -> pd.DataFrame:
    raw_path = os.path.join(data_path, "raw", f"{cfg.name}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    data = pd.read_csv(raw_path)

    user_col = cfg.column_name.user_id
    item_col = cfg.column_name.item_id
    time_col = cfg.column_name.timestamp
    relevance_col = cfg.column_name.relevance

    columns = [c for c in [user_col, item_col, time_col, relevance_col] if c is not None]
    data = data[columns].copy()
    data = rename_cols(data, user_col, item_col, time_col)

    return data


def base_preprocess(
    data: pd.DataFrame, drop_conseq_repeats: bool
) -> pd.DataFrame:
    if drop_conseq_repeats:
        data = drop_consecutive_repeats(data)
    return data


def compute_time_threshold(
    data: pd.DataFrame,
    quantile: float,
    cache_dir: str,
    dataset_name: str,
) -> float:
    _ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{_q_tag(quantile)}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    time_threshold = data["timestamp"].quantile(quantile)
    with open(cache_path, "wb") as f:
        pickle.dump(time_threshold, f)
    return time_threshold


def apply_filter(data: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    if spec.filter_type == "pcore":
        if spec.p is None:
            raise ValueError("FilterSpec.p is required for pcore filtering.")
        if spec.p == 0:
            return data.copy()
        return core_filter(
            data=data,
            item_min_count=spec.p,
            seq_min_len=spec.p,
            drop_conseq_repeats=False,
            user_id="user_id",
            item_id="item_id",
            timestamp="timestamp",
        )
    if spec.filter_type == "mincount":
        if spec.u_min is None or spec.i_min is None:
            raise ValueError("FilterSpec.u_min and FilterSpec.i_min are required for mincount.")
        filtered = min_count_filter(data, min_count=spec.u_min, col_name="user_id")
        filtered = min_count_filter(filtered, min_count=spec.i_min, col_name="item_id")
        return filtered
    raise ValueError(f"Unknown filter_type: {spec.filter_type}")


def split_validation_last_train(
    train: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.sort_values([user_col, time_col], kind="stable")
    train["time_idx_reversed"] = train.groupby(user_col).cumcount(ascending=False)
    validation = train[
        train.groupby(user_col)["time_idx_reversed"].transform("max") > 0
    ].drop(columns=["time_idx_reversed"])
    train = train[train.time_idx_reversed >= 1]
    train = train[
        train.groupby(user_col)["time_idx_reversed"].transform("max") > 1
    ].drop(columns=["time_idx_reversed"])
    return train, validation


def split_validation_by_user(
    train: pd.DataFrame,
    validation_size: int,
    random_state: int,
    user_col: str = "user_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(random_state)
    users = train[user_col].unique()
    if validation_size >= len(users):
        raise ValueError("validation_size must be smaller than number of users.")
    val_users = rng.choice(users, size=validation_size, replace=False)
    validation = train[train[user_col].isin(val_users)]
    train = train[~train[user_col].isin(val_users)]
    return train, validation


def split_by_time_threshold(
    data: pd.DataFrame,
    time_threshold: float,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = data.sort_values([user_col, time_col], kind="stable")
    user_second = data.groupby(user_col)[time_col].apply(
        lambda s: s.iloc[1] if len(s) > 1 else np.nan
    ).dropna()
    train_users = user_second[user_second <= time_threshold].index
    train = data[
        (data[user_col].isin(train_users)) & (data[time_col] <= time_threshold)
    ]
    user_last = data.groupby(user_col)[time_col].last()
    test_users = user_last[user_last > time_threshold].index
    test = data[data[user_col].isin(test_users)]
    return train, test


def split_global_time_fixed(
    data: pd.DataFrame,
    time_threshold: float,
    validation_type: str,
    validation_quantile: float,
    validation_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float]]:
    train, test = split_by_time_threshold(data, time_threshold)

    val_time_threshold = None
    if validation_type == "last_train_item":
        train, validation = split_validation_last_train(train)
    elif validation_type == "by_user":
        train, validation = split_validation_by_user(train, validation_size, random_state)
    elif validation_type == "by_time":
        val_time_threshold = train["timestamp"].quantile(validation_quantile)
        train, validation = split_by_time_threshold(train, val_time_threshold)
    else:
        raise ValueError(f"Unknown validation_type: {validation_type}")

    return train, validation, test, val_time_threshold


def filter_short_sequences(
    data: pd.DataFrame,
    min_len: int = 2,
    user_col: str = "user_id",
) -> pd.DataFrame:
    counts = data[user_col].value_counts()
    keep_users = counts[counts >= min_len].index
    return data[data[user_col].isin(keep_users)]


def encode_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    users = pd.Index(
        pd.concat(
            [train["user_id"], validation["user_id"], test["user_id"]],
            ignore_index=True,
        ).unique()
    ).sort_values()
    items = pd.Index(
        pd.concat(
            [train["item_id"], validation["item_id"], test["item_id"]],
            ignore_index=True,
        ).unique()
    ).sort_values()

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: idx + 1 for idx, i in enumerate(items)}

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["user_id"] = df["user_id"].map(user_map).astype("int64")
        df["item_id"] = df["item_id"].map(item_map).astype("int64")
        return df

    return _apply(train), _apply(validation), _apply(test), {
        "n_users": len(users),
        "n_items": len(items),
    }


def gini_coef(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    x = np.sort(x)
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def top_share(x: np.ndarray, frac: float) -> float:
    if len(x) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(x) * frac)))
    top = np.sort(x)[::-1][:k].sum()
    return float(top) / float(x.sum())


def compute_extended_stats(df: pd.DataFrame) -> Dict:
    stats = dataset_stats(df, extended=True)
    item_counts = df["item_id"].value_counts().values
    stats["item_gini"] = gini_coef(item_counts)
    stats["item_top1pct_share"] = top_share(item_counts, 0.01)
    stats["item_top5pct_share"] = top_share(item_counts, 0.05)
    stats["item_top10pct_share"] = top_share(item_counts, 0.10)
    user_counts = df["user_id"].value_counts().values
    stats["user_gini"] = gini_coef(user_counts)
    return stats


def make_variant_name(base: str, spec: FilterSpec, order: str) -> str:
    if spec.filter_type == "pcore":
        tag = f"pcore_p{spec.p}"
    else:
        tag = f"mincount_u{spec.u_min}_i{spec.i_min}"
    order_tag = {
        "filter_then_split": "v1_filter_then_split",
        "split_then_filter_train": "v2_split_then_filter_train",
        "split_then_filter_each": "v3_split_then_filter_each",
    }.get(order, order)
    return f"{base}__{tag}__{order_tag}"


def build_variant(
    base_data: pd.DataFrame,
    spec: FilterSpec,
    order: str,
    time_threshold: float,
    validation_type: str,
    validation_quantile: float,
    validation_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict, Optional[float]]:
    meta = {
        "filter_spec": asdict(spec),
        "order": order,
        "validation_type": validation_type,
        "validation_quantile": validation_quantile,
        "validation_size": validation_size,
        "random_state": random_state,
        "time_threshold": time_threshold,
    }

    if order == "filter_then_split":
        filtered = apply_filter(base_data, spec)
        train, validation, test, val_time_threshold = split_global_time_fixed(
            filtered,
            time_threshold=time_threshold,
            validation_type=validation_type,
            validation_quantile=validation_quantile,
            validation_size=validation_size,
            random_state=random_state,
        )
        meta["filtered_interactions"] = len(filtered)

    elif order == "split_then_filter_train":
        train_raw, validation_raw, test_raw, val_time_threshold = split_global_time_fixed(
            base_data,
            time_threshold=time_threshold,
            validation_type=validation_type,
            validation_quantile=validation_quantile,
            validation_size=validation_size,
            random_state=random_state,
        )
        train = apply_filter(train_raw, spec)
        train_users = set(train["user_id"].unique())
        train_items = set(train["item_id"].unique())
        validation = validation_raw[
            validation_raw["user_id"].isin(train_users)
            & validation_raw["item_id"].isin(train_items)
        ]
        test = test_raw[
            test_raw["user_id"].isin(train_users) & test_raw["item_id"].isin(train_items)
        ]
        meta.update(
            {
                "filtered_interactions": len(train),
                "removed_validation_interactions": int(len(validation_raw) - len(validation)),
                "removed_test_interactions": int(len(test_raw) - len(test)),
                "removed_validation_users": int(
                    validation_raw["user_id"].nunique() - validation["user_id"].nunique()
                ),
                "removed_test_users": int(
                    test_raw["user_id"].nunique() - test["user_id"].nunique()
                ),
                "removed_validation_items": int(
                    validation_raw["item_id"].nunique() - validation["item_id"].nunique()
                ),
                "removed_test_items": int(
                    test_raw["item_id"].nunique() - test["item_id"].nunique()
                ),
            }
        )
        validation = filter_short_sequences(validation, min_len=2)
        test = filter_short_sequences(test, min_len=2)

    elif order == "split_then_filter_each":
        train_raw, validation_raw, test_raw, val_time_threshold = split_global_time_fixed(
            base_data,
            time_threshold=time_threshold,
            validation_type=validation_type,
            validation_quantile=validation_quantile,
            validation_size=validation_size,
            random_state=random_state,
        )
        train = apply_filter(train_raw, spec)
        validation = apply_filter(validation_raw, spec)
        test = apply_filter(test_raw, spec)
        meta["filtered_interactions"] = len(train) + len(validation) + len(test)
        validation = filter_short_sequences(validation, min_len=2)
        test = filter_short_sequences(test, min_len=2)
    else:
        raise ValueError(f"Unknown order: {order}")

    train = filter_short_sequences(train, min_len=2)

    if spec.filter_type == "pcore" and spec.p is not None and len(train) > 0:
        user_counts = train["user_id"].value_counts()
        item_counts = train["item_id"].value_counts()
        meta["train_core_violation_users_pct"] = float(
            (user_counts < spec.p).mean() * 100.0
        )
        meta["train_core_violation_items_pct"] = float(
            (item_counts < spec.p).mean() * 100.0
        )

    train_enc, validation_enc, test_enc, enc_meta = encode_splits(
        train, validation, test
    )
    meta.update(enc_meta)

    stats = {
        "train": compute_extended_stats(train_enc),
        "validation": compute_extended_stats(validation_enc),
        "test": compute_extended_stats(test_enc),
        "full": compute_extended_stats(
            pd.concat([train_enc, validation_enc, test_enc], ignore_index=True)
        ),
    }

    return train_enc, validation_enc, test_enc, meta, stats, val_time_threshold


def save_variant(
    data_path: str,
    dataset_name: str,
    quantile: float,
    validation_type: str,
    variant_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    time_threshold: float,
    val_time_threshold: Optional[float],
    meta: Dict,
    stats: Dict,
) -> str:
    split_subtype = f"val_{validation_type}"
    q = _q_tag(quantile)
    out_dir = os.path.join(
        data_path, "splitted", "global_timesplit", split_subtype, variant_name, q
    )
    _ensure_dir(out_dir)

    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    validation.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    with open(os.path.join(out_dir, "time_threshold.pkl"), "wb") as f:
        pickle.dump(time_threshold, f)
    if validation_type == "by_time" and val_time_threshold is not None:
        with open(os.path.join(out_dir, "val_time_threshold.pkl"), "wb") as f:
            pickle.dump(val_time_threshold, f)

    with open(os.path.join(out_dir, "variant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "variant_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=_json_default)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate preprocessing/split variants.")
    parser.add_argument("--dataset", required=True, help="Dataset config key, e.g., Beauty")
    parser.add_argument("--filter-type", required=True, choices=["pcore", "mincount"])
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--u-min", type=int, default=None)
    parser.add_argument("--i-min", type=int, default=None)
    parser.add_argument(
        "--order",
        required=True,
        choices=["filter_then_split", "split_then_filter_train", "split_then_filter_each"],
    )
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument(
        "--validation-type",
        choices=["last_train_item", "by_user", "by_time"],
        default="last_train_item",
    )
    parser.add_argument("--validation-quantile", type=float, default=0.9)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--drop-conseq-repeats", action="store_true", default=False)
    parser.add_argument("--variant-name", default=None)
    args = parser.parse_args()

    data_path = os.environ.get("SEQ_SPLITS_DATA_PATH")
    if not data_path:
        raise EnvironmentError("SEQ_SPLITS_DATA_PATH is not set.")

    cfg = load_dataset_config(args.dataset)
    raw = load_raw_data(cfg, data_path)
    base = base_preprocess(raw, drop_conseq_repeats=args.drop_conseq_repeats)

    cache_dir = os.path.join(data_path, "variants", "time_thresholds")
    time_threshold = compute_time_threshold(base, args.quantile, cache_dir, cfg.name)

    spec = FilterSpec(
        filter_type=args.filter_type,
        p=args.p,
        u_min=args.u_min,
        i_min=args.i_min,
    )
    order = args.order
    variant_name = args.variant_name or make_variant_name(cfg.name, spec, order)

    train, validation, test, meta, stats, val_time_threshold = build_variant(
        base,
        spec,
        order,
        time_threshold=time_threshold,
        validation_type=args.validation_type,
        validation_quantile=args.validation_quantile,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    out_dir = save_variant(
        data_path=data_path,
        dataset_name=cfg.name,
        quantile=args.quantile,
        validation_type=args.validation_type,
        variant_name=variant_name,
        train=train,
        validation=validation,
        test=test,
        time_threshold=time_threshold,
        val_time_threshold=val_time_threshold,
        meta=meta,
        stats=stats,
    )
    print(f"Saved variant to {out_dir}")


if __name__ == "__main__":
    main()
