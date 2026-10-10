#!/usr/bin/env python3
"""Build the complete failure-boundary table (plan §12 item 8).

The transparency policy of this project is that every endpoint is reported
regardless of sign: frozen internal endpoints, the two phase-2/3 external
candidates, the two historic externals, the pure-MSE ablation, the
random-gene-set null and the persistence contrast. Until now those verdicts
lived in seven different report files, so a reviewer could not see in one
place which endpoints ODE wins, ties or loses.

This tool reads only committed report artifacts, derives one row per
endpoint/contrast and writes:

* ``artifacts/reports/failure_boundary_table.json`` — machine-readable rows
  with the source file recorded per row;
* ``artifacts/reports/failure_boundary_table.md`` — a reading-order table
  that marks each row as ``win`` / ``tie`` / ``loss`` / ``null`` relative to
  the stated comparator, with the pre-committed policy note.

Classification rules (fixed here, before looking at any row):

* a row is ``win`` / ``loss`` only if the 95% interval excludes zero in the
  corresponding direction;
* an interval that spans zero is a ``tie``; exact-zero differences are ties;
* without an interval, a row with a one-sided p < 0.05 is ``win_p_only`` and
  one with p >= 0.05 is ``not_significant``; a row with neither interval nor
  p-value is ``positive_point_estimate`` / ``negative_point_estimate``;
* mechanism controls use the same interval rule, so a reversal would surface
  as ``win`` / ``loss`` against the reference and is not hidden;
* the random-prior null compares the observed value against the null
  distribution quantiles (``null`` inside, ``above_null`` / ``below_null``
  outside).

Every row is retained. The tool never edits the source reports.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"


def classify(mean: float | None, ci_low: float | None, ci_high: float | None) -> str:
    if mean is None:
        return "not_evaluable"
    if ci_low is not None and ci_high is not None:
        if ci_low > 0:
            return "win"
        if ci_high < 0:
            return "loss"
        return "tie"
    if mean > 0:
        return "positive_point_estimate"
    if mean < 0:
        return "negative_point_estimate"
    return "tie"


def classify_with_p(mean: float, p_value: float | None) -> str:
    """Point-estimate rows that carry a one-sided paired p-value."""
    if p_value is None:
        return classify(mean, None, None)
    if p_value >= 0.05:
        return "not_significant"
    return "win_p_only" if mean > 0 else "loss_p_only"


def classify_null_against_quantiles(
    observed: float, q_low: float, q_high: float
) -> str:
    """Random-null rows: the reference is a null distribution, not a mean CI."""
    if observed > q_high:
        return "above_null"
    if observed < q_low:
        return "below_null"
    return "null"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def build_rows() -> list[dict]:
    rows: list[dict] = []

    # 1. Frozen internal endpoints (committed drift report).
    drift = read_json(REPORT_DIR / "composite_drift_report.json")
    src = "artifacts/reports/composite_drift_report.json"
    ode32 = drift["endpoint_aligned_contrasts"]["ode_vs_gru"]
    comp = ode32["composite"]
    rows.append(
        {
            "layer": "internal_frozen",
            "endpoint": "27->36/48 composite",
            "comparator": "matched GRU",
            "left_label": "ODE",
            "right_label": "GRU",
            "left_arm_mean": comp["left_mean"],
            "right_arm_mean": comp["right_mean"],
            "diff_mean": comp["diff_mean"],
            "ci95": None,
            "p_value": comp["p_value"],
            "verdict": classify_with_p(comp["diff_mean"], comp["p_value"]),
            "note": "one-sided paired t-test, df=2, exploratory",
            "source": src,
        }
    )
    pers = drift["endpoint_aligned_contrasts"]["ode_vs_persistence"]["composite"]
    rows.append(
        {
            "layer": "internal_frozen",
            "endpoint": "27->36/48 composite",
            "comparator": "population persistence",
            "left_label": "ODE",
            "right_label": "persistence",
            "left_arm_mean": pers["left_mean"],
            "right_arm_mean": pers["right_mean"],
            "diff_mean": pers["diff_mean"],
            "ci95": None,
            "p_value": pers["p_value"],
            "verdict": classify_with_p(pers["diff_mean"], pers["p_value"]),
            "note": "not significant at the composite endpoint (p=0.145)",
            "source": src,
        }
    )

    # 2. External phase-3 candidates (each on its own preregistered seed
    #    panel: the two mouse datasets at five seeds, GSE305979 at three).
    phase3 = read_json(REPORT_DIR / "external_phase3_aggregate_v2.json")
    forest = read_json(REPORT_DIR / "external_phase3_forest.json")
    forest_by_label = {e["label"]: e for e in forest["endpoints"]}
    for accession, block in phase3["datasets"].items():
        for horizon, hb in block["horizons"].items():
            label = f"{accession} {hb['source_time']}->{horizon}"
            fort = forest_by_label.get(label)
            n_seeds = len(hb["ode"]["per_seed"])
            rows.append(
                {
                    "layer": "external_phase3",
                    "endpoint": f"{accession} {hb['source_time']}->{horizon}",
                    "comparator": "matched GRU",
                    "left_label": "ODE",
                    "right_label": "GRU",
                    "left_arm_mean": hb["ode"]["summary"]["mean"],
                    "right_arm_mean": hb["gru"]["summary"]["mean"],
                    "diff_mean": (fort or {}).get("mean_diff"),
                    "ci95": (fort or {}).get("ci95"),
                    "p_value": None,
                    "verdict": classify(
                        (fort or {}).get("mean_diff"),
                        *((fort or {}).get("ci95") or (None, None)),
                    ),
                    "note": (
                        f"{n_seeds} seeds; normal-approximation interval from "
                        "the seed-level paired differences"
                    ),
                    "source": "artifacts/reports/external_phase3_forest.json",
                }
            )
            persistence = hb.get("persistence") or {}
            if persistence.get("status") == "completed":
                rows.append(
                    {
                        "layer": "external_phase3",
                        "endpoint": f"{accession} {hb['source_time']}->{horizon}",
                        "comparator": "population persistence",
                        "left_label": "ODE",
                        "right_label": "persistence",
                        "left_arm_mean": hb["ode"]["summary"]["mean"],
                        "right_arm_mean": persistence["corr"],
                        "diff_mean": hb["ode"]["summary"]["mean"]
                        - persistence["corr"],
                        "ci95": None,
                        "p_value": None,
                        "verdict": classify(
                            hb["ode"]["summary"]["mean"] - persistence["corr"],
                            None,
                            None,
                        ),
                        "note": "deterministic persistence, point estimate only",
                        "source": "artifacts/reports/external_phase3_aggregate_v2.json",
                    }
                )

    # 3. Historic externals (3-seed aggregates with per-seed extrap means).
    historic = [
        (
            "GSE224518 3->5",
            "artifacts/reports/stage2_gse224518_3to5h_agg.json",
        ),
        (
            "GSE224518 3->5 (balanced)",
            "artifacts/reports/stage2_gse224518_3to5h_balanced_agg.json",
        ),
        ("GSE224518 3->8", "artifacts/reports/stage2_gse224518_agg.json"),
        ("GSE226824 24->72", "artifacts/reports/stage2_gse226824_agg.json"),
    ]
    for label, source in historic:
        payload = read_json(ROOT / source)
        agg3 = payload["agg3"]
        left_mean = agg3["ode_extrap"]["mean"]
        right_mean = agg3["gru_extrap"]["mean"]
        fort = forest_by_label.get(label)
        diff = fort["mean_diff"] if fort else left_mean - right_mean
        ci = fort.get("ci95") if fort else None
        rows.append(
            {
                "layer": "historic_external",
                "endpoint": label,
                "comparator": "matched GRU",
                "left_label": "ODE",
                "right_label": "GRU",
                "left_arm_mean": left_mean,
                "right_arm_mean": right_mean,
                "diff_mean": diff,
                "ci95": ci,
                "p_value": None,
                "verdict": classify(diff, *(ci or (None, None))),
                "note": "3 seeds; retained as boundary evidence",
                "source": source,
            }
        )

    # 4. Pure-MSE ablation (matched internal, 3 seeds).
    pure = read_json(REPORT_DIR / "pure_mse_vs_primary_comparison.json")
    paired = pure["composite"]["paired_pure_minus_primary"]
    rows.append(
        {
            "layer": "mechanism_control",
            "endpoint": "internal 27->36/48 composite",
            "comparator": "full model (pure MSE minus full)",
            "left_label": "ODE pure-MSE",
            "right_label": "ODE full",
            "left_arm_mean": pure["composite"]["pure_mse"]["mean"],
            "right_arm_mean": pure["composite"]["primary"]["mean"],
            "diff_mean": paired["diff_mean"],
            "ci95": paired["ci95_difference"],
            "p_value": paired["p_two_sided"],
            "verdict": classify(paired["diff_mean"], *paired["ci95_difference"]),
            "note": "null: removing pathway terms does not hurt the endpoint",
            "source": "artifacts/reports/pure_mse_vs_primary_comparison.json",
        }
    )

    # 5. Random gene-set negative control (20 controls x 3 seeds).
    rnd = read_json(REPORT_DIR / "random_gene_set_negative_control.json")
    rm = rnd["comparison"]["composite"]
    rows.append(
        {
            "layer": "mechanism_control",
            "endpoint": "internal 27->36/48 composite",
            "comparator": "matched-size random pathway priors",
            "left_label": "ODE primary",
            "right_label": "random-prior null mean",
            "left_arm_mean": rm["observed"],
            "right_arm_mean": rm["null_mean"],
            "diff_mean": rm["observed"] - rm["null_mean"],
            "ci95": [rm["null_q025"], rm["null_q975"]],
            "p_value": rm["empirical_p_ge"],
            "verdict": classify_null_against_quantiles(
                rm["observed"], rm["null_q025"], rm["null_q975"]
            ),
            "note": (
                "null: prior content is not distinguishable from random sets; "
                "the interval column holds the null-distribution quantiles "
                "[q025, q975], not a difference CI"
            ),
            "source": "artifacts/reports/random_gene_set_negative_control.json",
        }
    )

    # 6. External pure-MSE contrast (plan item 12.7, two accessions).
    # Optional block: a checkout that has not run the external pure-MSE arm
    # yet still gets a complete internal/historic boundary table.
    external_pure_path = REPORT_DIR / "external_pure_mse_contrast.json"
    if external_pure_path.is_file():
        external_pure = read_json(external_pure_path)
        for accession, block in external_pure["datasets"].items():
            for endpoint, eb in block["endpoints"].items():
                paired = eb["paired_pure_minus_full"]
                rows.append(
                    {
                        "layer": "mechanism_control_external",
                        "endpoint": f"{accession} {endpoint}",
                        "comparator": "full model (pure MSE minus full)",
                        "left_label": "ODE pure-MSE",
                        "right_label": "ODE full",
                        "left_arm_mean": eb["pure_mse"]["mean"],
                        "right_arm_mean": eb["full"]["mean"],
                        "diff_mean": paired["diff_mean"],
                        "ci95": paired["ci95_difference"],
                        "p_value": paired["p_two_sided"],
                        "verdict": classify(
                            paired["diff_mean"], *paired["ci95_difference"]
                        ),
                        "note": (
                            "null replicates on the external endpoint: "
                            "removing the pathway losses does not move the "
                            "population endpoint"
                        ),
                        "source": (
                            "artifacts/reports/external_pure_mse_contrast.json"
                        ),
                    }
                )

    return rows


def build_markdown(rows: list[dict], created_at: str) -> str:
    lines = [
        "# Failure-boundary table (all endpoints, all signs)",
        "",
        f"Generated: {created_at}",
        "",
        "Policy (pre-committed): every endpoint is retained and reported,",
        "including ODE ties and losses. A `win` requires a positive mean",
        "difference with a 95% interval lower bound above zero; intervals that",
        "span zero are `tie`. Rows with only a one-sided p-value are",
        "`win_p_only` / `loss_p_only` when p < 0.05 and `not_significant`",
        "otherwise; rows with no interval and no p-value stay at",
        "`positive_point_estimate` / `negative_point_estimate` and must not be",
        "upgraded to significance. Mechanism controls are `null` while",
        "indistinguishable from their reference, and the random-prior row",
        "compares the observed value against the null distribution quantiles",
        "rather than a difference interval.",
        "",
        "| layer | endpoint | comparator | left | right | diff | 95% CI | p | verdict |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        ci = row.get("ci95")
        ci_text = (
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if isinstance(ci, (list, tuple)) and ci and ci[0] is not None
            else "—"
        )
        p = row.get("p_value")
        p_text = f"{p:.4g}" if isinstance(p, (int, float)) else "—"
        lines.append(
            f"| {row['layer']} | {row['endpoint']} | {row['comparator']} | "
            f"{row['left_arm_mean']:+.4f} ({row['left_label']}) | "
            f"{row['right_arm_mean']:+.4f} ({row['right_label']}) | "
            f"{row['diff_mean']:+.4f} | {ci_text} | {p_text} | {row['verdict']} |"
        )
    lines.append("")
    lines.append("## Notes per row")
    lines.append("")
    for row in rows:
        lines.append(
            f"- **{row['endpoint']}** vs {row['comparator']}: {row['note']} "
            f"(`{row['source']}`)"
        )
    lines.append("")
    lines.append("## Counting")
    lines.append("")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    for verdict, count in sorted(counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORT_DIR / "failure_boundary_table.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORT_DIR / "failure_boundary_table.md",
    )
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat()
    rows = build_rows()
    report = {
        "record_type": "failure_boundary_table",
        "created_at": created_at,
        "policy": (
            "All endpoints retained regardless of sign. win/loss require the "
            "95% interval to exclude zero; tie covers intervals spanning zero "
            "and exact zero. Rows with only a one-sided p-value use "
            "win_p_only/loss_p_only (p<0.05) or not_significant; rows with no "
            "interval and no p-value stay at positive_point_estimate / "
            "negative_point_estimate and must not be upgraded to significance. "
            "Mechanism controls are null while indistinguishable from their "
            "reference; the random-prior row compares the observed value "
            "against the null distribution quantiles."
        ),
        "rows": rows,
        "counts": {
            verdict: sum(1 for r in rows if r["verdict"] == verdict)
            for verdict in sorted({r["verdict"] for r in rows})
        },
    }
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(build_markdown(rows, created_at), encoding="utf-8")
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
