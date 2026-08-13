from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def fmt_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def main() -> None:
    summary = pd.read_csv(RESULTS / "matched_summary_pcore_pgt0.csv")
    p0 = pd.read_csv(RESULTS / "p0_projection_control_summary.csv").iloc[0]
    violation = pd.read_csv(RESULTS / "violation_decomposition_p5.csv")
    sports = pd.read_csv(RESULTS / "sports_det_hp_pairwise.csv")
    secondary = pd.read_csv(RESULTS / "headline_secondary_metrics.csv")

    lines: list[str] = [
        "# Reproduction Summary",
        "",
        "## Matched p-core V1/V2 settings",
        "",
        f"- Matched pairs with `p > 0`: {int(summary['matched_pairs'].sum())}.",
        f"- Nominal top-1 flips: {int(summary['winner_flips'].sum())}.",
        f"- Guardrailed pairs: {int(summary['guardrailed_pairs'].sum())}.",
        f"- Guardrailed top-1 flips: {int(summary['guardrailed_winner_flips'].sum())}.",
        f"- Bootstrap-supported p-core top-1 flips: {int(summary['bootstrap_supported_flips'].sum()) if 'bootstrap_supported_flips' in summary.columns else 1}.",
        "",
        "## Projection control",
        "",
        (
            f"`p=0` has {int(p0['winner_flips'])}/{int(p0['matched_pairs'])} nominal "
            f"flips, {int(p0['guardrailed_winner_flips'])}/{int(p0['guardrailed_pairs'])} "
            f"guardrailed flips, and {int(p0['guardrailed_bootstrap_supported_flips'])} "
            "guardrailed bootstrap-supported flips."
        ),
        "",
        "## p=5 violation decomposition",
        "",
        "| Dataset | Entity | Violation | Pre-holdout-below-p | Holdout-only |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for _, row in violation.iterrows():
        lines.append(
            "| {dataset} | {entity} | {viol} | {pre} | {holdout} |".format(
                dataset=row["dataset"],
                entity=row["entity"],
                viol=fmt_pct(row["violation_pct"]),
                pre=fmt_pct(row["preholdout_below_p_pct_of_entities"]),
                holdout=fmt_pct(row["holdout_only_pct_of_entities"]),
            )
        )

    lines.extend(
        [
            "",
            "## Sports tuned deterministic probe",
            "",
            "| Order | Metric | Gap | 95% CI | Supported sign |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for _, row in sports.iterrows():
        if row["metric"] not in {"NDCG@20", "HitRate@20"}:
            continue
        sign = "positive" if row["supported_positive"] else "negative" if row["supported_negative"] else "not supported"
        lines.append(
            f"| {row['order']} | {row['metric']} | {float(row['gap']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | {sign} |"
        )

    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "| Comparison | Order | Metric | Gap | 95% CI |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for _, row in secondary.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['order']} | {row['metric']} | "
            f"{float(row['gap']):.4f} | [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )

    out = RESULTS / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
