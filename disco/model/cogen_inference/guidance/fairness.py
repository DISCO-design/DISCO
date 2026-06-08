"""Fairness audit for planned ProteinGenerator-vs-DISCO comparison matrices.

The comparison planner writes commands and per-run metadata.  This module reads
that `planned_runs.csv` and checks whether the matrix is fair enough to execute
or interpret: same modes, same variants, same sequence lengths, same motif
sets, matched recycle count, balanced replicate counts, matched seeds, and
explicitly documented non-equivalent budget knobs for PG and DISCO.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FairnessIssue:
    """One fairness finding for a planned comparison matrix."""

    severity: str
    check: str
    mode: str
    variant: str
    system: str
    replicate: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def read_plan_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_parameters(row: dict[str, str]) -> dict[str, Any]:
    try:
        parsed = json.loads(row.get("parameters_json", "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def issue(
    row: dict[str, str],
    *,
    severity: str,
    check: str,
    message: str,
    path: str | Path | None = None,
) -> FairnessIssue:
    return FairnessIssue(
        severity=severity,
        check=check,
        mode=row.get("mode", ""),
        variant=row.get("variant", ""),
        system=row.get("system", ""),
        replicate=row.get("replicate", ""),
        path=str(path if path is not None else row.get("run_dir", "")),
        message=message,
    )


def normalized_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return value


def value_by_row(rows: list[dict[str, str]], key: str) -> dict[tuple[str, str], Any]:
    """Return `{(system, replicate): parameters[key]}` for a group."""

    values: dict[tuple[str, str], Any] = {}
    for row in rows:
        values[(row.get("system", ""), row.get("replicate", ""))] = normalized_value(
            parse_parameters(row).get(key)
        )
    return values


def distinct_values(values: dict[tuple[str, str], Any]) -> set[Any]:
    return {value for value in values.values()}


def check_group_parameter_equal(
    rows: list[dict[str, str]],
    *,
    key: str,
    severity: str = "error",
) -> list[FairnessIssue]:
    """Check one shared planned parameter within a mode/variant group."""

    values = value_by_row(rows, key)
    observed = distinct_values(values)
    if len(observed) <= 1:
        return []
    first = rows[0]
    return [
        issue(
            first,
            severity=severity,
            check=f"{key}_matched",
            message=f"{key} is not matched across systems/replicates: {values}",
            path="planned_runs.csv",
        )
    ]


def check_required_parameter_present(
    rows: list[dict[str, str]],
    *,
    system: str,
    key: str,
) -> list[FairnessIssue]:
    issues: list[FairnessIssue] = []
    for row in rows:
        if row.get("system", "") != system:
            continue
        parameters = parse_parameters(row)
        value = parameters.get(key)
        if value in ["", None]:
            issues.append(
                issue(
                    row,
                    severity="error",
                    check=f"{system.lower()}_{key}_present",
                    message=f"{system} row is missing parameter {key}",
                )
            )
            continue
        try:
            if float(value) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                issue(
                    row,
                    severity="error",
                    check=f"{system.lower()}_{key}_positive",
                    message=f"{system} parameter {key} must be positive: {value!r}",
                )
            )
    return issues


def check_group_balance(rows: list[dict[str, str]]) -> list[FairnessIssue]:
    """Check replicate and seed balance across systems for one group."""

    by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_system[row.get("system", "")].append(row)

    issues: list[FairnessIssue] = []
    counts = {system: len(system_rows) for system, system_rows in by_system.items()}
    if len(set(counts.values())) > 1:
        issues.append(
            issue(
                rows[0],
                severity="error",
                check="replicate_count_balance",
                message=f"unequal replicate counts by system: {counts}",
                path="planned_runs.csv",
            )
        )

    seed_sets = {
        system: tuple(sorted(row.get("seed", "") for row in system_rows))
        for system, system_rows in by_system.items()
    }
    if len(set(seed_sets.values())) > 1:
        issues.append(
            issue(
                rows[0],
                severity="error",
                check="seed_set_balance",
                message=f"unequal seed sets by system: {seed_sets}",
                path="planned_runs.csv",
            )
        )
    return issues


def check_statuses(rows: list[dict[str, str]]) -> list[FairnessIssue]:
    issues: list[FairnessIssue] = []
    for row in rows:
        status = row.get("status", "")
        if status == "planned":
            continue
        issues.append(
            issue(
                row,
                severity="error",
                check="planned_status",
                message=f"run is not executable/planned: {status}",
            )
        )
    return issues


def check_recommended_n_cycle(
    rows: list[dict[str, str]],
    *,
    recommended_n_cycle: int,
) -> list[FairnessIssue]:
    issues: list[FairnessIssue] = []
    for row in rows:
        value = parse_parameters(row).get("n_cycle")
        if value == recommended_n_cycle:
            continue
        issues.append(
            issue(
                row,
                severity="warning",
                check="n_cycle_reference",
                message=(
                    f"n_cycle={value!r} differs from the reference "
                    f"N={recommended_n_cycle}; document why this is fair"
                ),
            )
        )
    return issues


def budget_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Return human-readable budget fields for one group."""

    disco_values = [
        parse_parameters(row).get("disco_n_step")
        for row in rows
        if row.get("system", "") == "DISCO"
    ]
    pg_values = [
        parse_parameters(row).get("pg_T")
        for row in rows
        if row.get("system", "") == "ProteinGenerator"
    ]
    return {
        "disco_n_step_values": sorted({str(value) for value in disco_values}),
        "pg_T_values": sorted({str(value) for value in pg_values}),
        "budget_interpretation": (
            "DISCO sample_diffusion.N_step and ProteinGenerator --T are not "
            "mathematically identical; treat them as documented compute/sampling "
            "budget knobs, not as equal diffusion-time variables."
        ),
    }


def audit_fairness_rows(
    rows: list[dict[str, str]],
    *,
    required_systems: Iterable[str] = ("DISCO", "ProteinGenerator"),
    recommended_n_cycle: int = 4,
) -> tuple[list[FairnessIssue], dict[str, Any]]:
    """Audit planned comparison rows and return issues plus JSON summary."""

    issues: list[FairnessIssue] = []
    required = tuple(required_systems)
    if not rows:
        return [
            FairnessIssue(
                severity="error",
                check="planned_rows_present",
                mode="",
                variant="",
                system="",
                replicate="",
                path="planned_runs.csv",
                message="planned_runs.csv has no rows",
            )
        ], {
            "status": "fail",
            "num_groups": 0,
            "groups": [],
            "error_count": 1,
            "warning_count": 0,
        }

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("mode", ""), row.get("variant", ""))].append(row)

    group_summaries: list[dict[str, Any]] = []
    for (mode, variant), group_rows in sorted(groups.items()):
        systems = sorted({row.get("system", "") for row in group_rows})
        missing_systems = [system for system in required if system not in systems]
        if missing_systems:
            issues.append(
                issue(
                    group_rows[0],
                    severity="error",
                    check="required_systems",
                    message=(
                        "group is missing required system(s): "
                        + ",".join(missing_systems)
                    ),
                    path="planned_runs.csv",
                )
            )

        issues.extend(check_statuses(group_rows))
        issues.extend(check_group_balance(group_rows))
        for key in ["n_cycle", "length", "motifs", "designs_per_replicate"]:
            issues.extend(check_group_parameter_equal(group_rows, key=key))
        issues.extend(check_recommended_n_cycle(
            group_rows,
            recommended_n_cycle=recommended_n_cycle,
        ))
        issues.extend(
            check_required_parameter_present(
                group_rows,
                system="DISCO",
                key="disco_n_step",
            )
        )
        issues.extend(
            check_required_parameter_present(
                group_rows,
                system="ProteinGenerator",
                key="pg_T",
            )
        )

        group_summaries.append(
            {
                "mode": mode,
                "variant": variant,
                "systems": systems,
                "num_rows": len(group_rows),
                "replicates_by_system": {
                    system: sorted(
                        row.get("replicate", "")
                        for row in group_rows
                        if row.get("system", "") == system
                    )
                    for system in systems
                },
                "seeds_by_system": {
                    system: sorted(
                        row.get("seed", "")
                        for row in group_rows
                        if row.get("system", "") == system
                    )
                    for system in systems
                },
                "parameters": {
                    key: sorted(str(value) for value in distinct_values(
                        value_by_row(group_rows, key)
                    ))
                    for key in [
                        "n_cycle",
                        "length",
                        "motifs",
                        "designs_per_replicate",
                    ]
                },
                "budget": budget_summary(group_rows),
            }
        )

    error_count = sum(1 for item in issues if item.severity == "error")
    warning_count = sum(1 for item in issues if item.severity == "warning")
    return issues, {
        "status": "pass" if error_count == 0 else "fail",
        "required_systems": list(required),
        "recommended_n_cycle": recommended_n_cycle,
        "num_groups": len(group_summaries),
        "num_rows": len(rows),
        "error_count": error_count,
        "warning_count": warning_count,
        "groups": group_summaries,
    }


def audit_fairness_csv(
    planned_runs_csv: Path,
    *,
    required_systems: Iterable[str] = ("DISCO", "ProteinGenerator"),
    recommended_n_cycle: int = 4,
) -> tuple[list[FairnessIssue], dict[str, Any]]:
    rows = read_plan_rows(planned_runs_csv)
    issues, summary = audit_fairness_rows(
        rows,
        required_systems=required_systems,
        recommended_n_cycle=recommended_n_cycle,
    )
    summary["planned_runs_csv"] = str(planned_runs_csv)
    return issues, summary


def write_fairness_report(
    *,
    report_json_path: Path,
    issues_csv_path: Path,
    issues: list[FairnessIssue],
    summary: dict[str, Any],
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    issues_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(
            {
                **summary,
                "issues": [item.as_dict() for item in issues],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    fieldnames = [
        "severity",
        "check",
        "mode",
        "variant",
        "system",
        "replicate",
        "path",
        "message",
    ]
    with issues_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([issue.as_dict() for issue in issues])
