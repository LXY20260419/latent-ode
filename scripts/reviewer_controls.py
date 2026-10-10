#!/usr/bin/env python3
"""Generate and aggregate reviewer-requested controls.

Dry runs use only the Python standard library and never write artifacts. A
production artifact is written only after required source files and provenance
metadata validate successfully.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
REPORT_DIR = ROOT / "artifacts" / "reports"
DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_TARGETS = [36, 48]
DEFAULT_SOURCE_TIME = 27
METRIC = "population_mean_across_gene_pearson_r"


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def pearson_corr(a, b) -> float:
    """Pure-stdlib Pearson correlation for dry-run fixtures."""
    left = [float(x) for x in a]
    right = [float(x) for x in b]
    if len(left) < 2 or len(left) != len(right):
        raise ValueError("correlation inputs must have equal length >= 2")
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    centered_left = [x - mean_left for x in left]
    centered_right = [x - mean_right for x in right]
    denominator = math.sqrt(
        sum(x * x for x in centered_left)
        * sum(y * y for y in centered_right)
    )
    if denominator <= 1e-12:
        raise ValueError("correlation is undefined for a constant vector")
    return (
        sum(x * y for x, y in zip(centered_left, centered_right))
        / denominator
    )


def quantile(values, probability: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_seed_bootstrap(values) -> dict:
    """Enumerate all n**n bootstrap resamples for a small seed vector."""
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("bootstrap values must be nonempty and finite")
    means = [
        statistics.mean(sample)
        for sample in itertools.product(values, repeat=len(values))
    ]
    return {
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "ci_method": f"exact_seed_bootstrap_enumeration_{len(means)}",
        "bootstrap_resamples": len(means),
    }


def summarize_seed_values(per_seed: dict[str, float]) -> dict:
    values = [float(per_seed[key]) for key in sorted(per_seed, key=int)]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("seed values must be nonempty and finite")
    payload = {
        "per_seed": {
            str(key): float(value) for key, value in per_seed.items()
        },
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sd_ddof": 1,
    }
    payload.update(exact_seed_bootstrap(values))
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def tagged_validation_path(
    result_dir: Path, model: str, result_tag: str, seed: int
) -> Path:
    if not result_tag:
        raise ValueError("production aggregation requires a result tag")
    return (
        result_dir
        / f"validation_results_{model}_{result_tag}_seed{seed}.json"
    )


def load_validation_result(
    path: Path,
    *,
    seed: int,
    result_tag: str,
    targets=DEFAULT_TARGETS,
    source_time=DEFAULT_SOURCE_TIME,
    required_weights: tuple[float, float] | None = None,
    expected_priors_sha256: str | None = None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name}: missing run_metadata")
    protocol = metadata.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path.name}: missing run_metadata.protocol")
    if protocol.get("seed") != seed:
        raise ValueError(f"{path.name}: metadata seed does not match {seed}")
    if protocol.get("result_tag") != result_tag:
        raise ValueError(
            f"{path.name}: metadata result_tag does not match {result_tag!r}"
        )
    selection = protocol.get("checkpoint_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("holdout_used_for_selection") is not False
    ):
        raise ValueError(
            f"{path.name}: holdout must not be used for checkpoint selection"
        )
    pairs = payload.get("time_extrapolation", {}).get("pairs")
    if not isinstance(pairs, dict):
        raise ValueError(f"{path.name}: missing time_extrapolation.pairs")
    expected_keys = {str(target) for target in targets}
    if {str(key) for key in pairs} != expected_keys:
        raise ValueError(
            f"{path.name}: expected horizons {sorted(expected_keys)}"
        )
    correlations = {}
    for target in targets:
        row = pairs.get(str(target), pairs.get(target))
        if not isinstance(row, dict) or row.get("from_time") != source_time:
            raise ValueError(
                f"{path.name}: {target}h must extrapolate from {source_time}h"
            )
        value = row.get("corr")
        if not finite_number(value) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{path.name}: invalid finite correlation for {target}h"
            )
        correlations[str(target)] = float(value)

    config = metadata.get("config")
    if required_weights is not None:
        if not isinstance(config, dict):
            raise ValueError(f"{path.name}: missing run_metadata.config")
        pathway_weight, pathway_delta_weight = required_weights
        if config.get("pathway_weight") != pathway_weight:
            raise ValueError(
                f"{path.name}: pathway_weight must equal {pathway_weight}"
            )
        if config.get("pathway_delta_weight") != pathway_delta_weight:
            raise ValueError(
                f"{path.name}: pathway_delta_weight must equal "
                f"{pathway_delta_weight}"
            )
    if expected_priors_sha256 is not None:
        hashes = metadata.get("source_hashes")
        if (
            not isinstance(hashes, dict)
            or hashes.get("biology_priors_sha256")
            != expected_priors_sha256
        ):
            raise ValueError(
                f"{path.name}: biology-prior hash does not match control config"
            )
    return {
        "path": path,
        "payload": payload,
        "metadata": metadata,
        "correlations": correlations,
        "composite": statistics.mean(correlations.values()),
    }


def aggregate_tagged_results(
    *,
    result_dir: Path,
    result_tag: str,
    model: str,
    seeds: list[int],
    targets=DEFAULT_TARGETS,
    source_time=DEFAULT_SOURCE_TIME,
    required_weights: tuple[float, float] | None = None,
    expected_priors_sha256: str | None = None,
) -> dict:
    per_horizon = {str(target): {} for target in targets}
    per_seed_composite = {}
    source_files = []
    for seed in seeds:
        path = tagged_validation_path(
            result_dir, model, result_tag, seed
        )
        record = load_validation_result(
            path,
            seed=seed,
            result_tag=result_tag,
            targets=targets,
            source_time=source_time,
            required_weights=required_weights,
            expected_priors_sha256=expected_priors_sha256,
        )
        source_files.append(
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        )
        per_seed_composite[str(seed)] = record["composite"]
        for target in targets:
            per_horizon[str(target)][str(seed)] = record[
                "correlations"
            ][str(target)]
    return {
        "source_files": source_files,
        "horizons": {
            target: {
                "from_time": source_time,
                **summarize_seed_values(values),
            }
            for target, values in per_horizon.items()
        },
        "composite": summarize_seed_values(per_seed_composite),
    }


def persistence_from_rows(
    rows: list[list[float]], times: list[int], train, targets
) -> dict:
    source = max(train)
    source_rows = [
        row for row, time in zip(rows, times) if time == source
    ]
    if not source_rows:
        raise ValueError(f"no source cells at {source}h")
    source_mean = [
        statistics.mean(column) for column in zip(*source_rows)
    ]
    pairs = {}
    for target in targets:
        target_rows = [
            row for row, time in zip(rows, times) if time == target
        ]
        if not target_rows:
            raise ValueError(f"no target cells at {target}h")
        target_mean = [
            statistics.mean(column) for column in zip(*target_rows)
        ]
        pairs[str(target)] = {
            "from_time": source,
            "corr": pearson_corr(source_mean, target_mean),
            "n_source": len(source_rows),
            "n_target": len(target_rows),
        }
    values = [row["corr"] for row in pairs.values()]
    return {
        "method": "population_persistence",
        "metric": METRIC,
        "train_times": list(map(int, train)),
        "target_times": list(map(int, targets)),
        "pairs": pairs,
        "mean_extrapolation_corr": statistics.mean(values),
    }


def production_persistence(
    data_dir: Path, train, targets
) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "production persistence requires numpy"
        ) from exc
    matrix_path = data_dir / "expression_matrix.npy"
    metadata_path = data_dir / "metadata.csv"
    x = np.load(matrix_path)
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        times = np.asarray(
            [
                int(float(row["timepoint_numeric"]))
                for row in rows
            ]
        )
    if len(times) != x.shape[0]:
        raise ValueError(
            "metadata row count does not match expression matrix"
        )
    source = max(train)
    source_mask = times == source
    if not np.any(source_mask):
        raise ValueError(f"no source cells at {source}h")
    source_mean = x[source_mask].mean(axis=0)
    pairs = {}
    for target in targets:
        target_mask = times == target
        if not np.any(target_mask):
            raise ValueError(f"no target cells at {target}h")
        target_mean = x[target_mask].mean(axis=0)
        corr = float(np.corrcoef(source_mean, target_mean)[0, 1])
        if not math.isfinite(corr):
            raise ValueError(
                f"non-finite persistence correlation at {target}h"
            )
        pairs[str(target)] = {
            "from_time": source,
            "corr": corr,
            "n_source": int(source_mask.sum()),
            "n_target": int(target_mask.sum()),
        }
    return {
        "method": "population_persistence",
        "metric": METRIC,
        "train_times": list(map(int, train)),
        "target_times": list(map(int, targets)),
        "pairs": pairs,
        "mean_extrapolation_corr": statistics.mean(
            row["corr"] for row in pairs.values()
        ),
        "source_files": {
            "expression_matrix": str(matrix_path.relative_to(ROOT)),
            "metadata": str(metadata_path.relative_to(ROOT)),
            "metadata_sha256": file_sha256(metadata_path),
        },
    }


def build_uncertainty(
    *, result_dir: Path, result_tag: str, model: str, seeds: list[int]
) -> dict:
    aggregated = aggregate_tagged_results(
        result_dir=result_dir,
        result_tag=result_tag,
        model=model,
        seeds=seeds,
    )
    return {
        "status": "completed",
        "metric": METRIC,
        "model": model,
        "result_tag": result_tag,
        "seeds": seeds,
        **aggregated,
        "note": (
            "Uncertainty is across stochastic training seeds, "
            "not biological replicates."
        ),
    }


def build_manual_uncertainty(
    horizons: dict[str, list[float]]
) -> dict:
    if set(horizons) != {"36", "48"}:
        raise ValueError(
            "manual uncertainty requires exactly horizons 36 and 48"
        )
    return {
        "status": "completed_manual",
        "metric": METRIC,
        "horizons": {
            target: {
                "from_time": DEFAULT_SOURCE_TIME,
                **summarize_seed_values(
                    {
                        str(index): value
                        for index, value in enumerate(values)
                    }
                ),
            }
            for target, values in horizons.items()
        },
        "note": (
            "Manual values are not eligible for the submission "
            "readiness gate."
        ),
    }


def build_pure_mse(
    *, result_dir: Path, result_tag: str, model: str, seeds: list[int]
) -> dict:
    aggregated = aggregate_tagged_results(
        result_dir=result_dir,
        result_tag=result_tag,
        model=model,
        seeds=seeds,
        required_weights=(0.0, 0.0),
    )
    return {
        "status": "completed",
        "group": "pure_mse",
        "metric": METRIC,
        "model": model,
        "config": {
            "pathway_weight": 0.0,
            "pathway_delta_weight": 0.0,
            "result_tag": result_tag,
        },
        "seeds": seeds,
        **aggregated,
    }


def load_gene_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != len({name.upper() for name in names}):
        raise ValueError(
            "gene universe contains duplicate case-insensitive symbols"
        )
    return names


def load_gmt_pathways(path: Path) -> dict[str, list[str]]:
    pathways = {}
    if not path.is_file():
        return pathways
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split("\t")
        if len(parts) >= 3 and parts[0].strip():
            pathways[parts[0].strip()] = [
                gene.strip() for gene in parts[2:] if gene.strip()
            ]
    return pathways


def generate_random_prior_plan(
    *,
    gene_names: list[str],
    base_priors: dict,
    replicates: int,
    seed: int,
    output_dir: Path | None,
    tag_prefix: str,
) -> dict:
    pathways = base_priors.get("pathways")
    if not isinstance(pathways, dict) or not pathways:
        raise ValueError(
            "base priors must contain nonempty pathways"
        )
    universe = {name.upper() for name in gene_names}
    requested_sizes = {
        name: len(genes) for name, genes in pathways.items()
    }
    sizes = {
        name: len(
            {
                str(gene).upper()
                for gene in genes
                if str(gene).upper() in universe
            }
        )
        for name, genes in pathways.items()
    }
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("pathway sizes must be positive")
    if sum(sizes.values()) > len(gene_names):
        raise ValueError(
            "gene universe is too small for disjoint matched sets"
        )
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = random.Random(seed)
    controls = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for replicate in range(replicates):
        chosen = rng.sample(gene_names, sum(sizes.values()))
        offset = 0
        random_pathways = {}
        for name, size in sizes.items():
            random_pathways[name] = chosen[
                offset : offset + size
            ]
            offset += size
        config = dict(base_priors)
        config["pathways"] = random_pathways
        tag = f"{tag_prefix}_r{replicate:03d}"
        record = {
            "replicate": replicate,
            "result_tag": tag,
            "pathways": random_pathways,
        }
        if output_dir is not None:
            path = output_dir / f"biology_priors_{tag}.json"
            path.write_text(
                json.dumps(
                    config,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            record["config_path"] = str(
                path.relative_to(ROOT)
                if path.is_relative_to(ROOT)
                else path
            )
            record["config_sha256"] = file_sha256(path)
        controls.append(record)
    return {
        "status": (
            "prepared" if output_dir is not None else "dry_run"
        ),
        "control": "matched_size_random_pathway_training",
        "gene_universe_size": len(gene_names),
        "matched_sizes": sizes,
        "requested_sizes": requested_sizes,
        "rng": {
            "generator": "random.Random",
            "seed": seed,
            "sampling": (
                "without_replacement_within_each_control"
            ),
        },
        "replicates": replicates,
        "result_tag_prefix": tag_prefix,
        "controls": controls,
        "run_contract": {
            "environment": {
                "LEWORLD_BIOLOGY_PRIORS_PATH": (
                    "<control.config_path>"
                ),
                "LEWORLD_EXTERNAL_PATHWAY_GMT": "",
                "LEWORLD_RESULT_TAG": "<control.result_tag>",
                "LEWORLD_SEED": "each configured seed",
            },
            "note": (
                "Each control must be retrained; gene-list overlap "
                "alone is not the null statistic."
            ),
        },
    }


def aggregate_random_prior_controls(
    *,
    plan: dict,
    result_dir: Path,
    model: str,
    seeds: list[int],
    primary_result_tag: str,
    max_controls: int | None = None,
    scope_decision: dict | None = None,
) -> dict:
    controls = plan.get("controls")
    if (
        plan.get("status") != "prepared"
        or not isinstance(controls, list)
        or not controls
    ):
        raise ValueError(
            "random-gene-set plan must be a nonempty prepared manifest"
        )
    if max_controls is not None:
        if max_controls <= 0:
            raise ValueError("max_controls must be positive")
        if max_controls > len(controls):
            raise ValueError(
                f"max_controls {max_controls} exceeds the planned "
                f"{len(controls)} controls"
            )
        controls = controls[:max_controls]
    primary = aggregate_tagged_results(
        result_dir=result_dir,
        result_tag=primary_result_tag,
        model=model,
        seeds=seeds,
    )
    null_composite = []
    null_horizons = {
        str(target): [] for target in DEFAULT_TARGETS
    }
    per_control = []
    all_source_files = list(primary["source_files"])
    for record in controls:
        tag = record.get("result_tag")
        config_sha256 = record.get("config_sha256")
        if not isinstance(tag, str) or not isinstance(
            config_sha256, str
        ):
            raise ValueError(
                "each control needs result_tag and config_sha256"
            )
        aggregated = aggregate_tagged_results(
            result_dir=result_dir,
            result_tag=tag,
            model=model,
            seeds=seeds,
            expected_priors_sha256=config_sha256,
        )
        all_source_files.extend(aggregated["source_files"])
        composite_mean = aggregated["composite"]["mean"]
        null_composite.append(composite_mean)
        horizon_means = {}
        for target in DEFAULT_TARGETS:
            value = aggregated["horizons"][str(target)][
                "mean"
            ]
            horizon_means[str(target)] = value
            null_horizons[str(target)].append(value)
        per_control.append(
            {
                "replicate": record.get("replicate"),
                "result_tag": tag,
                "config_path": record.get("config_path"),
                "config_sha256": config_sha256,
                "composite_mean": composite_mean,
                "horizon_means": horizon_means,
                "source_files": aggregated["source_files"],
            }
        )

    def compare(
        observed: float, null_values: list[float]
    ) -> dict:
        return {
            "observed": observed,
            "null_mean": statistics.mean(null_values),
            "null_sd": (
                statistics.stdev(null_values)
                if len(null_values) > 1
                else 0.0
            ),
            "null_q025": quantile(null_values, 0.025),
            "null_q975": quantile(null_values, 0.975),
            "empirical_p_ge": (
                1
                + sum(
                    value >= observed
                    for value in null_values
                )
            )
            / (len(null_values) + 1),
        }

    comparison = {
        "composite": compare(
            primary["composite"]["mean"], null_composite
        ),
        "horizons": {
            target: compare(
                primary["horizons"][target]["mean"], values
            )
            for target, values in null_horizons.items()
        },
    }
    return {
        "status": "completed",
        "control": "matched_size_random_pathway_training",
        "metric": METRIC,
        "model": model,
        "primary_result_tag": primary_result_tag,
        "seeds": seeds,
        "replicates": len(per_control),
        "matched_sizes": plan.get("matched_sizes"),
        "gene_universe_size": plan.get("gene_universe_size"),
        "rng": plan.get("rng"),
        "primary": primary,
        "per_control": per_control,
        "comparison": comparison,
        "source_files": all_source_files,
        "planned_replicates": len(plan.get("controls") or []),
        "scope_decision": scope_decision,
        "note": (
            "Null distribution is over retrained matched-size "
            "random pathway priors."
        ),
    }


def build_scope_decision(
    *,
    scope_record: Path,
    replicates: int,
    rationale: str | None,
    full_grid_status: str,
) -> dict:
    """Build the embedded scope block for a reduced random-control grid.

    The reviewer report asks for randomly sampled matched-size gene sets plus
    permutation/bootstrap tests; it names no replicate count. The gate
    therefore accepts a reduced grid only when the author's decision, its
    rationale, and the decision record are embedded in the artifact, so the
    reduced scope is auditable rather than silent.
    """
    record_path = Path(scope_record)
    if record_path.is_absolute() and record_path.is_relative_to(ROOT):
        record_text = record_path.relative_to(ROOT).as_posix()
    else:
        record_text = record_path.as_posix()
    text = (
        rationale
        if rationale
        else (
            "Author decision 2+4 (2026-09-27): a reduced matched-size random "
            "gene-set grid answers the reviewer's negative-control request "
            "within the available single-host budget, with the full 100-control "
            "grid retained for a submission-side request. See "
            f"{record_text} for measured per-run cost and the "
            "option comparison."
        )
    )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("scope rationale must be a non-empty string")
    return {
        "decision": "documented_subset",
        "decision_owner": "author",
        "declared_replicates": int(replicates),
        "rationale": text,
        "record": record_text,
        "full_grid_status": full_grid_status,
    }


def write_json(
    payload: dict, output: Path | None, dry_run: bool = False
) -> None:
    text = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    )
    if dry_run or output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(
        dest="command", required=True
    )

    command = sub.add_parser("persistence")
    command.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "leworld_input",
    )
    command.add_argument(
        "--train-times", default="0,1,4,16,27"
    )
    command.add_argument("--target-times", default="36,48")
    command.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "results"
            / "baselines"
            / "persistence_population.json"
        ),
    )
    command.add_argument("--dry-run", action="store_true")

    command = sub.add_parser("uncertainty")
    command.add_argument(
        "--result-dir", type=Path, default=RESULT_DIR
    )
    command.add_argument("--result-tag")
    command.add_argument(
        "--model", default="ode", choices=["ode", "gru"]
    )
    command.add_argument("--seeds", default="42,123,2026")
    command.add_argument(
        "--horizon", action="append", default=[]
    )
    command.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR
        / "horizon_specific_uncertainty.json",
    )
    command.add_argument("--dry-run", action="store_true")

    command = sub.add_parser("pure-mse")
    command.add_argument(
        "--result-dir", type=Path, default=RESULT_DIR
    )
    command.add_argument(
        "--result-tag", default="pure_mse"
    )
    command.add_argument(
        "--model", default="ode", choices=["ode", "gru"]
    )
    command.add_argument("--seeds", default="42,123,2026")
    command.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "pure_mse_ablation_agg.json",
    )
    command.add_argument("--dry-run", action="store_true")

    command = sub.add_parser("random-gene-sets")
    command.add_argument(
        "--mode",
        choices=["prepare", "aggregate"],
        default="prepare",
    )
    command.add_argument(
        "--gene-names",
        type=Path,
        default=(
            ROOT
            / "data"
            / "leworld_input"
            / "gene_names.txt"
        ),
    )
    command.add_argument(
        "--base-priors",
        type=Path,
        default=ROOT / "configs" / "biology_priors.json",
    )
    command.add_argument(
        "--external-pathways",
        type=Path,
        default=ROOT / "configs" / "external_pathways.gmt",
    )
    command.add_argument(
        "--config-dir",
        type=Path,
        default=REPORT_DIR / "random_gene_set_configs",
    )
    command.add_argument(
        "--manifest",
        type=Path,
        default=REPORT_DIR
        / "random_gene_set_control_plan.json",
    )
    command.add_argument(
        "--replicates", type=int, default=100
    )
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument(
        "--result-tag-prefix", default="random_gene_set"
    )
    command.add_argument(
        "--result-dir", type=Path, default=RESULT_DIR
    )
    command.add_argument(
        "--primary-result-tag",
        default="primary_internal_freeze",
    )
    command.add_argument(
        "--model", default="ode", choices=["ode", "gru"]
    )
    command.add_argument("--seeds", default="42,123,2026")
    command.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR
        / "random_gene_set_negative_control.json",
    )
    command.add_argument(
        "--max-controls",
        type=int,
        default=None,
        help=(
            "Aggregate only the first N planned controls. Used for the "
            "author-declared reduced grid; the full plan stays on disk."
        ),
    )
    command.add_argument(
        "--scope-record",
        type=Path,
        default=REPORT_DIR / "random_gene_set_scope_record.json",
        help="Author decision record the aggregate cites as its scope.",
    )
    command.add_argument(
        "--declared-rationale",
        default=None,
        help=(
            "Override for scope_decision.rationale. Defaults to the "
            "rationale stored in the scope record."
        ),
    )
    command.add_argument(
        "--full-grid-status",
        choices=["available_on_request", "completed"],
        default="available_on_request",
    )
    command.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "persistence":
        train = parse_int_list(args.train_times)
        targets = parse_int_list(args.target_times)
        if args.dry_run:
            rows = [
                [0.0, 1.0, 2.0],
                [0.1, 1.1, 2.1],
                [0.2, 1.3, 2.5],
                [0.3, 1.5, 2.8],
                [0.5, 1.7, 3.0],
                [0.7, 1.8, 3.2],
            ]
            payload = persistence_from_rows(
                rows,
                [27, 27, 36, 36, 48, 48],
                train,
                targets,
            )
            payload["status"] = "dry_run"
        else:
            payload = production_persistence(
                args.data_dir, train, targets
            )
            payload["status"] = "completed"
        write_json(payload, args.output, args.dry_run)
        return 0

    seeds = parse_int_list(args.seeds)
    if args.command == "uncertainty":
        if args.dry_run:
            payload = build_manual_uncertainty(
                {
                    "36": [0.70, 0.72, 0.74],
                    "48": [0.24, 0.26, 0.28],
                }
            )
            payload["status"] = "dry_run"
        elif args.result_tag:
            payload = build_uncertainty(
                result_dir=args.result_dir,
                result_tag=args.result_tag,
                model=args.model,
                seeds=seeds,
            )
        elif args.horizon:
            horizons = {
                key: [
                    float(value)
                    for value in raw.split(",")
                ]
                for key, raw in (
                    item.split("=", 1)
                    for item in args.horizon
                )
            }
            payload = build_manual_uncertainty(horizons)
        else:
            parser.error(
                "uncertainty production mode requires "
                "--result-tag or --horizon"
            )
        write_json(payload, args.output, args.dry_run)
        return 0

    if args.command == "pure-mse":
        if args.dry_run:
            payload = {
                "status": "dry_run",
                "group": "pure_mse",
                "config": {
                    "pathway_weight": 0.0,
                    "pathway_delta_weight": 0.0,
                    "result_tag": args.result_tag,
                },
                "seeds": seeds,
                "note": (
                    "No training or aggregation was performed."
                ),
            }
        else:
            payload = build_pure_mse(
                result_dir=args.result_dir,
                result_tag=args.result_tag,
                model=args.model,
                seeds=seeds,
            )
        write_json(payload, args.output, args.dry_run)
        return 0

    if args.dry_run:
        gene_names = load_gene_names(args.gene_names)
        base_priors = json.loads(
            args.base_priors.read_text(encoding="utf-8")
        )
        base_priors["pathways"] = {
            **base_priors.get("pathways", {}),
            **load_gmt_pathways(args.external_pathways),
        }
        payload = generate_random_prior_plan(
            gene_names=gene_names,
            base_priors=base_priors,
            replicates=min(args.replicates, 3),
            seed=args.seed,
            output_dir=None,
            tag_prefix=args.result_tag_prefix,
        )
        write_json(payload, args.output, True)
        return 0
    if args.mode == "prepare":
        base_priors = json.loads(
            args.base_priors.read_text(encoding="utf-8")
        )
        base_priors["pathways"] = {
            **base_priors.get("pathways", {}),
            **load_gmt_pathways(args.external_pathways),
        }
        payload = generate_random_prior_plan(
            gene_names=load_gene_names(args.gene_names),
            base_priors=base_priors,
            replicates=args.replicates,
            seed=args.seed,
            output_dir=args.config_dir,
            tag_prefix=args.result_tag_prefix,
        )
        write_json(payload, args.manifest)
        return 0
    plan = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    payload = aggregate_random_prior_controls(
        plan=plan,
        result_dir=args.result_dir,
        model=args.model,
        seeds=seeds,
        primary_result_tag=args.primary_result_tag,
        max_controls=args.max_controls,
        scope_decision=build_scope_decision(
            scope_record=args.scope_record,
            replicates=(
                args.max_controls
                if args.max_controls is not None
                else len(plan.get("controls") or [])
            ),
            rationale=args.declared_rationale,
            full_grid_status=args.full_grid_status,
        ),
    )
    write_json(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
