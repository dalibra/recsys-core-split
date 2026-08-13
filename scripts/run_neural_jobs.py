"""Launch neural evaluation/training jobs for the filter/split-order paper.

The script is intentionally a thin subprocess launcher around the existing
Hydra training entrypoint. It keeps all model logic in ``exps/runs/train.py``
and only decides which variant/model/seed combinations are still missing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
RESULTS = EXPS / "data" / "results" / "global_timesplit" / "val_last_train_item"
ANALYSIS = ROOT / "results"
Q_TAG = "q09"
ENV_NAME = os.environ.get("RECSYS_ENV", "recsys-repro")
NEURAL_MODELS = {"SASRec", "GRU4Rec", "BERT4Rec"}
SEED_MODELS = ["SASRec", "GRU4Rec", "BERT4Rec"]
SEED_SPECS = [
    ("Sports", "pcore_p5", SEED_MODELS),
    ("Movielens-1m", "pcore_p5", SEED_MODELS),
    ("Beauty", "pcore_p10", ["SASRec"]),
    ("Beauty", "pcore_p5", ["SASRec"]),
]
SEEDS = [17, 23, 42, 101, 202]


@dataclass(frozen=True)
class Job:
    kind: str
    variant: str
    model: str
    seed: int

    @property
    def dataset(self) -> str:
        return self.variant.split("__", 1)[0]


def per_user_exists(variant: str, model: str, seed: int) -> bool:
    path = RESULTS / variant / Q_TAG / model / "test_last_per_user"
    return path.exists() and any(p.name.endswith(f"_{seed}.csv") for p in path.glob("*.csv"))


def result_exists(variant: str, model: str, seed: int) -> bool:
    path = RESULTS / variant / Q_TAG / model / "test_last"
    return path.exists() and any(p.name.endswith(f"_{seed}.csv") for p in path.glob("*.csv"))


def bootstrap_jobs(scope: str) -> List[Job]:
    matched = pd.read_csv(ANALYSIS / "matched_v1v2_pairs.csv")
    if scope == "guarded":
        matched = matched[matched["passes_guardrail"]]
    elif scope == "pcore":
        matched = matched[matched["filter_family"] == "pcore"]
    elif scope == "pcore_guarded":
        matched = matched[(matched["filter_family"] == "pcore") & (matched["passes_guardrail"])]
    elif scope != "all":
        raise ValueError(f"Unknown scope: {scope}")

    pairs = set()
    for _, row in matched.iterrows():
        for side in ["v1", "v2"]:
            variant = str(row[f"{side}_variant"])
            for role in ["winner", "runner_up"]:
                model = str(row[f"{side}_{role}"])
                if model in NEURAL_MODELS:
                    pairs.add((variant, model))
    return [Job("per_user", variant, model, 17) for variant, model in sorted(pairs)]


def seed_jobs() -> List[Job]:
    jobs: List[Job] = []
    for dataset, filter_tag, models in SEED_SPECS:
        for order in ["v1_filter_then_split", "v2_split_then_filter_train"]:
            variant = f"{dataset}__{filter_tag}__{order}"
            for model in models:
                for seed in SEEDS:
                    jobs.append(Job("seed", variant, model, seed))
    return jobs


def shard(jobs: Sequence[Job], index: int, count: int) -> List[Job]:
    if count <= 0:
        raise ValueError("shard count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard index must be in [0, count)")
    return [job for pos, job in enumerate(jobs) if pos % count == index]


def build_command(job: Job, gpu: int, save_per_user: bool) -> List[str]:
    overrides = [
        f"dataset={job.dataset}",
        f"dataset.name={job.variant}",
        "split_type=global_timesplit",
        "split_subtype=val_last_train_item",
        "quantile=0.9",
        "validation_quantile=0.9",
        f"model={job.model}",
        "model.grid_point_number=0",
        f"random_state={job.seed}",
        f"cuda_visible_devices={gpu}",
        "dataloader.num_workers=4",
        "load_if_possible=True",
        "+trainer_params.accelerator=gpu",
        "+trainer_params.devices=1",
        "+trainer_predict_params.accelerator=gpu",
        "+trainer_predict_params.devices=1",
    ]
    if save_per_user:
        overrides.append("+save_local_per_user_metrics=True")

    return [
        "conda",
        "run",
        "-n",
        ENV_NAME,
        "python",
        "-m",
        "runs.train",
        *overrides,
    ]


def should_skip(job: Job, mode: str) -> bool:
    if job.kind == "per_user":
        return per_user_exists(job.variant, job.model, job.seed)
    if job.kind == "seed":
        if mode == "seed-results":
            return result_exists(job.variant, job.model, job.seed)
        return result_exists(job.variant, job.model, job.seed) and (
            not (job.seed == 17 and mode == "seed-per-user")
            or per_user_exists(job.variant, job.model, job.seed)
        )
    raise ValueError(job.kind)


def run_jobs(jobs: Iterable[Job], gpu: int, dry_run: bool, mode: str) -> None:
    env = os.environ.copy()
    env["SEQ_SPLITS_DATA_PATH"] = str(EXPS / "data")
    env["PYTHONPATH"] = str(EXPS)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    planned = list(jobs)
    print(f"Planned jobs on GPU {gpu}: {len(planned)}", flush=True)
    for idx, job in enumerate(planned, start=1):
        if should_skip(job, mode):
            print(f"[{idx}/{len(planned)}] skip {job.kind} {job.variant} {job.model} seed={job.seed}", flush=True)
            continue
        save_per_user = job.kind == "per_user"
        cmd = build_command(job, gpu=gpu, save_per_user=save_per_user)
        print(f"[{idx}/{len(planned)}] run {job.kind} {job.variant} {job.model} seed={job.seed}", flush=True)
        print(" ".join(cmd), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(cmd, cwd=EXPS, env=env)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bootstrap", "seed-results"], required=True)
    parser.add_argument("--bootstrap-scope", choices=["all", "guarded", "pcore", "pcore_guarded"], default="all")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "bootstrap":
        jobs = bootstrap_jobs(args.bootstrap_scope)
    else:
        jobs = seed_jobs()
    jobs = shard(jobs, args.shard_index, args.shard_count)
    run_jobs(jobs, gpu=args.gpu, dry_run=args.dry_run, mode=args.mode)


if __name__ == "__main__":
    main()
