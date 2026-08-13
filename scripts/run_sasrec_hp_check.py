"""Run the Beauty p=10 SASRec hyperparameter spot check.

The jobs deliberately write tagged result files so they do not change the
fixed grid-0 leaderboards used by the main paper analysis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
RESULTS = EXPS / "data" / "results" / "global_timesplit" / "val_last_train_item"
ANALYSIS = ROOT / "results"
ENV_NAME = os.environ.get("RECSYS_ENV", "recsys-repro")
Q_TAG = "q09"
DATASET = "Beauty"
FILTER_TAG = "pcore_p10"
SEEDS = [17, 23, 42, 101, 202]


@dataclass(frozen=True)
class HpSpec:
    grid_idx: int
    hidden_units: int
    dropout: float
    lr: float

    @property
    def tag(self) -> str:
        drop = str(self.dropout).replace(".", "")
        lr = {0.001: "001", 0.0003: "0003"}[self.lr]
        return f"hp_h{self.hidden_units}_d{drop}_lr{lr}"

    def result_stem(self, seed: int) -> str:
        return f"{self.grid_idx}_{self.hidden_units}_1_1_{self.dropout}_128_256_{seed}__{self.tag}"


HP_GRID: List[HpSpec] = [
    HpSpec(0, 32, 0.1, 0.001),
    HpSpec(0, 32, 0.1, 0.0003),
    HpSpec(1, 32, 0.3, 0.001),
    HpSpec(1, 32, 0.3, 0.0003),
    HpSpec(27, 64, 0.1, 0.001),
    HpSpec(27, 64, 0.1, 0.0003),
    HpSpec(28, 64, 0.3, 0.001),
    HpSpec(28, 64, 0.3, 0.0003),
    HpSpec(54, 128, 0.1, 0.001),
    HpSpec(54, 128, 0.1, 0.0003),
    HpSpec(55, 128, 0.3, 0.001),
    HpSpec(55, 128, 0.3, 0.0003),
]


@dataclass(frozen=True)
class Job:
    order: str
    spec: HpSpec
    seed: int

    @property
    def variant(self) -> str:
        order_tag = {
            "V1": "v1_filter_then_split",
            "V2": "v2_split_then_filter_train",
        }[self.order]
        return f"{DATASET}__{FILTER_TAG}__{order_tag}"


def metric_path(job: Job, prefix: str) -> Path:
    return (
        RESULTS
        / job.variant
        / Q_TAG
        / "SASRec"
        / prefix
        / f"{job.spec.result_stem(job.seed)}.csv"
    )


def job_done(job: Job) -> bool:
    return metric_path(job, "val_last").exists() and metric_path(job, "test_last").exists()


def grid_jobs() -> List[Job]:
    return [Job(order, spec, 17) for order in ["V1", "V2"] for spec in HP_GRID]


def best_seed_jobs() -> List[Job]:
    path = ANALYSIS / "beauty_sasrec_hp_best.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run summarize_sasrec_hp_check.py first: {path}")
    best = pd.read_csv(path)
    jobs: List[Job] = []
    by_tag = {spec.tag: spec for spec in HP_GRID}
    for _, row in best.iterrows():
        spec = by_tag[str(row["tag"])]
        for seed in SEEDS:
            jobs.append(Job(str(row["order"]), spec, seed))
    return jobs


def shard(jobs: Sequence[Job], index: int, count: int) -> List[Job]:
    if count <= 0:
        raise ValueError("shard count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard index must be in [0, count)")
    return [job for pos, job in enumerate(jobs) if pos % count == index]


def build_command(job: Job, gpu: int) -> List[str]:
    return [
        "conda",
        "run",
        "-n",
        ENV_NAME,
        "python",
        "-m",
        "runs.train",
        f"dataset={DATASET}",
        f"dataset.name={job.variant}",
        "split_type=global_timesplit",
        "split_subtype=val_last_train_item",
        "quantile=0.9",
        "validation_quantile=0.9",
        "model=SASRec",
        f"model.grid_point_number={job.spec.grid_idx}",
        f"seqrec_module.lr={job.spec.lr}",
        f"random_state={job.seed}",
        f"cuda_visible_devices={gpu}",
        "dataloader.num_workers=4",
        "load_if_possible=True",
        f"+result_tag={job.spec.tag}",
        "+trainer_params.accelerator=gpu",
        "+trainer_params.devices=1",
        "+trainer_predict_params.accelerator=gpu",
        "+trainer_predict_params.devices=1",
    ]


def run_jobs(jobs: Iterable[Job], gpu: int, dry_run: bool) -> None:
    env = os.environ.copy()
    env["SEQ_SPLITS_DATA_PATH"] = str(EXPS / "data")
    env["PYTHONPATH"] = str(EXPS)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    planned = list(jobs)
    print(f"Planned Beauty p=10 SASRec HP jobs on GPU {gpu}: {len(planned)}", flush=True)
    for idx, job in enumerate(planned, start=1):
        if job_done(job):
            print(f"[{idx}/{len(planned)}] skip {job.order} {job.spec.tag} seed={job.seed}", flush=True)
            continue
        cmd = build_command(job, gpu)
        print(f"[{idx}/{len(planned)}] run {job.order} {job.spec.tag} seed={job.seed}", flush=True)
        print(" ".join(cmd), flush=True)
        if dry_run:
            continue
        completed = subprocess.run(cmd, cwd=EXPS, env=env)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid", "best-seeds"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = grid_jobs() if args.mode == "grid" else best_seed_jobs()
    jobs = shard(jobs, args.shard_index, args.shard_count)
    run_jobs(jobs, gpu=args.gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
