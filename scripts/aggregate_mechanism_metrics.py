#!/usr/bin/env python3
"""Aggregate the mechanism-proxy metrics across internal and external arms.

The training script records four biology-facing proxies in every validation
artifact: ``tf_direction_consistency``, ``pathway_activation_order`` and
``marker_rank_spearman`` (plus the unaggregated ``literature`` block, which
this tool does not touch). They have existed per seed since the first run but
were never rolled up, so reviewers could only see whichever seed they opened.

Coupling statement (must stay attached to every reading of these numbers):

* the pathway gene sets (base priors plus ``configs/external_pathways.gmt``)
  enter the training objective through ``pathway_score`` and
  ``pathway_delta`` (weights 0.2 and 0.1), so pathway-order accuracy is a
  prior-consistency check on the trained model, not an independent mechanism
  validation;
* the TF sets (``configs/external_tf_targets.csv``) and marker sets are used
  only by the validation stage, but they were also merged into the priors
  object, so they are likewise consistency proxies, not held-out biology.

Aggregation rules:

* internal layer: the frozen rerun artifacts
  (``validation_results_{model}_primary_internal_freeze_seed{42,123,2026}.json``)
  that back the 0.4516 composite; the June-era recovered copies are *not*
  used;
* external layer: the tagged phase-2/3 artifacts (five seeds for the two
  mouse candidates and three seeds for GSE305979);
* metrics that are structurally not evaluable (e.g. pathway order on a
  single-holdout dataset) are recorded as not evaluable with the reason,
  never imputed;
* every seed is retained, including values that cut against ODE.

Outputs ``artifacts/reports/mechanism_metrics_aggregate.json`` and
``mechanism_metrics_aggregate.md``. Read-only with respect to all result
artifacts; never writes into the frozen namespaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "reports"

TF_KEY = "tf_direction_consistency"
PATHWAY_KEY = "pathway_activation_order"
MARKER_KEY = "marker_rank_spearman"

LAYERS = {
    "internal_primary_freeze": {
        "label": "Internal GSE151658 (frozen rerun, composite 0.4516)",
        "tag": "primary_internal_freeze",
        "seeds": [42, 123, 2026],
        "result_dir": None,  # defaults to the shared ode_benchmark dir
    },
    "gse148794": {
        "label": "External GSE148794 (DSS colitis, 9->12 / 9->15)",
        "tag": "ext_gse148794_v2",
        "seeds": [42, 123, 2026, 7, 13],
        "result_dir": None,
    },
    "gse149622": {
        "label": "External GSE149622 (IL-23 skin, 3->4)",
        "tag": "ext_gse149622_v2",
        "seeds": [42, 123, 2026, 7, 13],
        "result_dir": None,
    },
    "gse305979": {
        "label": "External GSE305979 (human CD34+ HSPC, 2->5 / 2->7)",
        "tag": "ext_gse305979_control_v2",
        "seeds": [42, 123, 2026],
        "result_dir": None,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "sd_ddof": 1,
        "min": min(finite),
        "max": max(finite),
    }


def load_artifact(result_dir: Path, tag: str, model: str, seed: int) -> dict:
    path = result_dir / f"validation_results_{model}_{tag}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def extract_block(payload: dict, key: str) -> dict | None:
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    overall = block.get("overall")
    if overall is None or not isinstance(overall, (int, float)):
        return None
    return block


def aggregate_layer(
    result_dir: Path,
    spec: dict,
) -> dict:
    tag = spec["tag"]
    arms: dict = {}
    source_files: list[dict] = []
    for model in ("ode", "gru"):
        per_seed: dict = {
            TF_KEY: {},
            PATHWAY_KEY: {},
            MARKER_KEY: {},
        }
        per_group: dict = {
            TF_KEY: {},
            PATHWAY_KEY: {},
            MARKER_KEY: {},
        }
        missing: dict = {}
        for seed in spec["seeds"]:
            payload = load_artifact(result_dir, tag, model, seed)
            source_path = result_dir / f"validation_results_{model}_{tag}_seed{seed}.json"
            try:
                shown_path = str(source_path.relative_to(ROOT))
            except ValueError:
                shown_path = str(source_path)
            source_files.append(
                {
                    "path": shown_path,
                    "sha256": sha256(source_path),
                }
            )
            for key, group_keys in (
                (TF_KEY, "per_tf"),
                (PATHWAY_KEY, "per_pathway"),
                (MARKER_KEY, "per_set"),
            ):
                block = extract_block(payload, key)
                if block is None:
                    missing.setdefault(key, []).append(seed)
                    continue
                per_seed[key][str(seed)] = block["overall"]
                for group_name, group_value in (block.get(group_keys) or {}).items():
                    if isinstance(group_value, (int, float)):
                        per_group[key].setdefault(group_name, {})[str(seed)] = float(
                            group_value
                        )
        arm = {}
        for key in (TF_KEY, PATHWAY_KEY, MARKER_KEY):
            seed_map = per_seed[key]
            summary = summarize(list(seed_map.values()))
            group_summaries = {
                name: summarize(list(values.values()))
                for name, values in per_group[key].items()
            }
            arm[key] = {
                "per_seed": seed_map,
                "summary": summary,
                "per_group_mean_across_seeds": group_summaries,
            }
            if missing.get(key):
                arm[key]["not_evaluable_seeds"] = sorted(missing[key])
        arms[model] = arm

    contrasts = {}
    for key in (TF_KEY, PATHWAY_KEY, MARKER_KEY):
        ode_sum = arms["ode"][key]["summary"]["mean"]
        gru_sum = arms["gru"][key]["summary"]["mean"]
        contrasts[key] = {
            "ode_minus_gru_mean": (
                ode_sum - gru_sum
                if ode_sum is not None and gru_sum is not None
                else None
            ),
            "note": (
                "mean-level difference; not a significance test and not "
                "exchangeable across layers"
            ),
        }

    return {
        "label": spec["label"],
        "result_tag": tag,
        "seeds": spec["seeds"],
        "arms": arms,
        "ode_minus_gru": contrasts,
        "source_files": source_files,
    }


def build_markdown(report: dict) -> str:
    lines = [
        "# Mechanism-proxy metrics across internal and external arms",
        "",
        f"Generated: {report['created_at']}",
        "",
        "These are prior-consistency proxies recorded by the training script's",
        "validation stage, aggregated across seeds for the first time. The",
        "pathway sets feed the training objective (`pathway_score` weight 0.2,",
        "`pathway_delta` weight 0.1), and the TF/marker sets were merged into",
        "the same priors object, so none of these numbers is an independent",
        "mechanism validation: they measure how far each trained arm sits from",
        "the provided biology priors, nothing more. All seeds are kept,",
        "including values that cut against ODE.",
        "",
        "| layer | arm | TF direction | pathway order | marker rank Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for layer_key, layer in report["layers"].items():
        for model in ("ode", "gru"):
            arm = layer["arms"][model]
            cells = []
            for key in (TF_KEY, PATHWAY_KEY, MARKER_KEY):
                summary = arm[key]["summary"]
                if summary["mean"] is None:
                    cells.append("not evaluable")
                else:
                    cells.append(
                        f"{summary['mean']:+.3f} ± {summary['sd']:.3f} "
                        f"(n={summary['n']})"
                    )
            lines.append(
                f"| {layer_key} | {model.upper()} | " + " | ".join(cells) + " |"
            )
    lines.append("")
    lines.append("## ODE minus GRU (mean level)")
    lines.append("")
    lines.append("| layer | TF direction | pathway order | marker rank |")
    lines.append("|---|---:|---:|---:|")
    for layer_key, layer in report["layers"].items():
        cells = []
        for key in (TF_KEY, PATHWAY_KEY, MARKER_KEY):
            diff = layer["ode_minus_gru"][key]["ode_minus_gru_mean"]
            cells.append(f"{diff:+.3f}" if diff is not None else "n/a")
        lines.append(f"| {layer_key} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Reading notes")
    lines.append("")
    lines.append(
        "- Positive TF/pathway/marker values mean the arm's extrapolated "
        "changes agree more with the direction implied by the priors."
    )
    lines.append(
        "- GSE149622 has a single holdout timepoint, so pathway order is "
        "structurally not evaluable there; that is recorded, not imputed."
    )
    lines.append(
        "- GSE305979 has pathway-order values for its three seeds, while the "
        "validation artifacts do not provide TF-direction or marker-rank "
        "values; those cells remain not evaluable."
    )
    lines.append(
        "- The internal layer uses the frozen rerun artifacts that back the "
        "0.4516 composite; the June-era recovered copies (0.4833 lineage) are "
        "deliberately not aggregated here."
    )
    lines.append(
        "- Differences are descriptive mean-level contrasts with three or "
        "five seeds; they are not significance tests and must not be used to "
        "overturn or replace the endpoint claims."
    )
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    for layer_key, layer in report["layers"].items():
        for record in layer["source_files"]:
            lines.append(f"- `{record['path']}` — `{record['sha256'][:16]}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="default: <report-dir>/mechanism_metrics_aggregate.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="default: <report-dir>/mechanism_metrics_aggregate.md",
    )
    args = parser.parse_args()

    layers = {}
    for layer_key, spec in LAYERS.items():
        result_dir = spec["result_dir"] or args.result_dir
        layers[layer_key] = aggregate_layer(Path(result_dir), spec)

    report = {
        "record_type": "mechanism_metrics_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "Prior-consistency proxies only: the pathway sets enter the "
            "training loss (pathway_score 0.2, pathway_delta 0.1) and the "
            "TF/marker sets were merged into the same priors object, so these "
            "values are not an independent mechanism validation."
        ),
        "aggregation_rules": [
            "internal layer uses the frozen rerun artifacts backing the 0.4516 composite",
            "external layers use five seeds for the two mouse candidates and three seeds for GSE305979",
            "not-evaluable metrics are recorded with the affected seeds, never imputed",
            "all seeds retained, including values cutting against ODE",
        ],
        "layers": layers,
    }

    output_json = args.output_json or (args.report_dir / "mechanism_metrics_aggregate.json")
    output_md = args.output_md or (args.report_dir / "mechanism_metrics_aggregate.md")
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"saved: {output_json}")
    print(f"saved: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
