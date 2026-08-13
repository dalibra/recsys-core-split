from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = pd.read_csv(RESULTS / "matched_summary_pcore_pgt0.csv")
    require(int(summary["matched_pairs"].sum()) == 14, "expected 14 matched p-core pairs with p>0")
    require(int(summary["winner_flips"].sum()) == 8, "expected 8 nominal p-core winner flips")
    require(int(summary["guardrailed_pairs"].sum()) == 10, "expected 10 guardrailed p-core pairs")
    require(int(summary["guardrailed_winner_flips"].sum()) == 5, "expected 5 guardrailed p-core winner flips")

    matched = pd.read_csv(RESULTS / "matched_v1v2_pairs_with_bootstrap.csv")
    pcore_pgt0 = matched[(matched["filter_family"] == "pcore") & (matched["filter_tag"] != "pcore_p0")]
    require(int(pcore_pgt0["bootstrap_supported_flip"].sum()) == 1, "expected one supported p-core top-1 flip")

    p0 = pd.read_csv(RESULTS / "p0_projection_control_summary.csv").iloc[0]
    require((int(p0["winner_flips"]), int(p0["guardrailed_winner_flips"])) == (2, 2), "unexpected p=0 flip counts")
    require(int(p0["guardrailed_bootstrap_supported_flips"]) == 0, "p=0 should have no supported guarded flip")

    v = pd.read_csv(RESULTS / "violation_decomposition_p5.csv")
    sports_users = v[(v["dataset"] == "Sports") & (v["entity"] == "users")].iloc[0]
    sports_items = v[(v["dataset"] == "Sports") & (v["entity"] == "items")].iloc[0]
    require(abs(float(sports_users["violation_pct"]) - 38.20) < 0.05, "unexpected Sports user violation rate")
    require(abs(float(sports_items["violation_pct"]) - 22.33) < 0.05, "unexpected Sports item violation rate")

    sports = pd.read_csv(RESULTS / "sports_det_hp_pairwise.csv")
    ndcg = sports[sports["metric"] == "NDCG@20"].set_index("order")
    require(bool(ndcg.loc["V1", "supported_negative"]), "Sports V1 tuned NDCG gap should be negative-supported")
    require(bool(ndcg.loc["V2", "supported_positive"]), "Sports V2 tuned NDCG gap should be positive-supported")

    print("Matched p-core pairs:", int(summary["matched_pairs"].sum()))
    print("Nominal p-core flips:", int(summary["winner_flips"].sum()))
    print("Guardrailed p-core flips:", int(summary["guardrailed_winner_flips"].sum()))
    print("Supported p-core top-1 flips:", int(pcore_pgt0["bootstrap_supported_flip"].sum()))
    print("p=0 control: 2/6 nominal, 2/5 guarded, 0 supported")
    print("Sports p=5 violations: users 38.20%, items 22.33%")
    print("Sports tuned ItemKNN/EASE^R NDCG@20 reversal: supported in both arms")


if __name__ == "__main__":
    main()
