"""
Orchestrate preprocessing variants and model training for filter/split order experiments.
"""

import argparse
import os
import subprocess
import sys
from typing import List

from variant_pipeline import (
    FilterSpec,
    _q_tag,
    base_preprocess,
    build_variant,
    compute_time_threshold,
    load_dataset_config,
    load_raw_data,
    make_variant_name,
    save_variant,
)


def parse_list(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def parse_int_list(arg: str) -> List[int]:
    return [int(x) for x in parse_list(arg)]


def parse_mincount_pairs(arg: str) -> List[FilterSpec]:
    specs = []
    for part in parse_list(arg):
        if ":" not in part:
            raise ValueError(f"Bad mincount spec: {part}")
        u, i = part.split(":")
        specs.append(FilterSpec(filter_type="mincount", u_min=int(u), i_min=int(i)))
    return specs


def results_exist(data_path: str, split_subtype: str, variant: str, q: str, model: str) -> bool:
    metrics_dir = os.path.join(
        data_path, "results", "global_timesplit", split_subtype, variant, q, model, "test_last"
    )
    if not os.path.exists(metrics_dir):
        return False
    return any(f.endswith(".csv") for f in os.listdir(metrics_dir))


def variant_has_data(out_dir: str) -> bool:
    stats_path = os.path.join(out_dir, "variant_stats.json")
    if not os.path.exists(stats_path):
        return True
    try:
        import json
        with open(stats_path, "r") as f:
            stats = json.load(f)
        train_ok = stats.get("train", {}).get("n_interactions", 0) > 0
        test_ok = stats.get("test", {}).get("n_interactions", 0) > 0
        return train_ok and test_ok
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser(description="Run filter order experiments.")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--validation-type", default="last_train_item")
    parser.add_argument("--validation-quantile", type=float, default=0.9)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--drop-conseq-repeats", action="store_true", default=True)
    parser.add_argument(
        "--orders",
        default="filter_then_split,split_then_filter_train",
    )
    parser.add_argument("--pcore", default="0,5,10,20")
    parser.add_argument("--mincount", default="5:5,5:10,10:5,10:10,20:20")
    parser.add_argument("--models", default="SASRec,GRU4Rec,BERT4Rec")
    parser.add_argument("--grid-point", default="0")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true")
    parser.add_argument("--no-force-cpu", dest="force_cpu", action="store_false")
    parser.set_defaults(force_cpu=False)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--load-if-possible", dest="load_if_possible", action="store_true")
    parser.add_argument("--no-load-if-possible", dest="load_if_possible", action="store_false")
    parser.set_defaults(load_if_possible=True)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--stage", default="all", choices=["prepare", "train", "analyze", "all"])
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--max-variants", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.environ.get("SEQ_SPLITS_DATA_PATH", os.path.join(repo_root, "data"))
    os.environ["SEQ_SPLITS_DATA_PATH"] = data_path
    os.environ.setdefault("PYTHONPATH", repo_root)

    datasets = parse_list(args.datasets)
    if not datasets:
        raw_dir = os.path.join(data_path, "raw")
        datasets = [os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith(".csv")]
        datasets = sorted(datasets)

    orders = parse_list(args.orders)
    pcore_specs = [FilterSpec(filter_type="pcore", p=p) for p in parse_int_list(args.pcore)]
    mincount_specs = parse_mincount_pairs(args.mincount)
    filter_specs = pcore_specs + mincount_specs

    models = parse_list(args.models)
    seeds = parse_int_list(args.seeds)
    grid_point = args.grid_point if args.grid_point.lower() != "none" else None

    q_tag = _q_tag(args.quantile)
    split_subtype = f"val_{args.validation_type}"

    variants_run = 0

    for dataset_key in datasets:
        cfg = load_dataset_config(dataset_key)
        raw = load_raw_data(cfg, data_path)
        base = base_preprocess(raw, drop_conseq_repeats=args.drop_conseq_repeats)
        time_threshold = compute_time_threshold(
            base,
            args.quantile,
            cache_dir=os.path.join(data_path, "variants", "time_thresholds"),
            dataset_name=cfg.name,
        )

        for spec in filter_specs:
            for order in orders:
                variant_name = make_variant_name(cfg.name, spec, order)
                out_dir = os.path.join(
                    data_path,
                    "splitted",
                    "global_timesplit",
                    split_subtype,
                    variant_name,
                    q_tag,
                )
                train_path = os.path.join(out_dir, "train.csv")

                if args.stage in ("prepare", "all"):
                    if args.skip_existing and os.path.exists(train_path):
                        pass
                    else:
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
                        save_variant(
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

                if args.stage in ("train", "all"):
                    if not variant_has_data(out_dir):
                        continue
                    for model in models:
                        for seed in seeds:
                            if args.skip_existing and results_exist(
                                data_path, split_subtype, variant_name, q_tag, model
                            ):
                                continue
                            cmd = [
                                sys.executable,
                                "runs/train.py",
                                f"dataset={dataset_key}",
                                f"dataset.name={variant_name}",
                                "split_type=global_timesplit",
                                f"split_subtype={split_subtype}",
                                f"quantile={args.quantile}",
                                f"validation_quantile={args.validation_quantile}",
                                f"model={model}",
                                f"random_state={seed}",
                                f"cuda_visible_devices={-1 if args.force_cpu else args.gpu}",
                                f"dataloader.num_workers={args.num_workers}",
                            ]
                            if grid_point is not None:
                                cmd.append(f"model.grid_point_number={grid_point}")
                            if args.max_epochs is not None:
                                cmd.append(f"trainer_params.max_epochs={args.max_epochs}")
                            if args.force_cpu:
                                cmd.append("+trainer_params.accelerator=cpu")
                                cmd.append("+trainer_params.devices=1")
                            if not args.load_if_possible:
                                cmd.append("load_if_possible=False")
                            if args.dry_run:
                                print("DRY RUN:", " ".join(cmd))
                            else:
                                subprocess.run(cmd, cwd=repo_root, check=True)

                variants_run += 1
                if args.max_variants is not None and variants_run >= args.max_variants:
                    break
            if args.max_variants is not None and variants_run >= args.max_variants:
                break
        if args.max_variants is not None and variants_run >= args.max_variants:
            break

        if args.stage in ("analyze", "all"):
            analysis_cmd = [
                sys.executable,
                "runs/leaderboard_stability.py",
                "--data-path",
                data_path,
                "--dataset",
                cfg.name,
                "--split-subtype",
                split_subtype,
                "--quantile",
                str(args.quantile),
                "--variant-prefix",
                f"{cfg.name}__",
            ]
            if args.dry_run:
                print("DRY RUN:", " ".join(analysis_cmd))
            else:
                subprocess.run(analysis_cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
