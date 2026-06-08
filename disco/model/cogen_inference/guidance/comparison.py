"""Normalize and compare ProteinGenerator and DISCO motif summary CSVs.

The normalizer keeps score provenance explicit.  ProteinGenerator summary CSVs
are expected to come from PDB-derived analyzers where sequence and pLDDT are
read from the same saved structure.  DISCO summary CSVs are expected to come
from `guidance_analyze_disco.py`, where sequence records are paired to matching
summary-confidence JSON files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value) -> float | None:
    if value in ["", None]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool | None:
    if value in ["", None]:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _first(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ["", None]:
            return str(value)
    return ""


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    normalized = value.replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _required_motif_count(present: str, missing: str) -> int:
    return len(set(_split_list(present) + _split_list(missing)))


def _motif_counts_from_fields(row: dict[str, str]) -> tuple[int, int, int]:
    """Return `(required, unique, total)` from columns like `GHK_count`.

    This covers DISCO evolutionary summaries, which store one count column per
    requested motif but do not always include `present_motifs/missing_motifs`.
    Lowercase diagnostic counters such as `list1_count` are intentionally
    ignored.
    """

    count_fields = [
        key
        for key in row
        if key.endswith("_count") and key[: -len("_count")].isupper()
    ]
    counts = [_as_int(row.get(key, "")) for key in count_fields]
    return len(counts), sum(1 for count in counts if count > 0), sum(counts)


def _comparison_score(
    *,
    confidence: float,
    motif_score: float,
    fixed_preserved: bool | None,
) -> float:
    """Return a transparent default quality score for cross-system ranking.

    This is not a biological truth metric.  It is a stable first-pass ranking
    score that combines confidence and exact motif coverage.  Immunogen rows
    with failed fixed fragments are assigned zero because those designs are
    invalid for that mode regardless of pLDDT/confidence.
    """

    if fixed_preserved is False:
        return 0.0
    return 0.5 * confidence + 0.5 * motif_score


@dataclass(frozen=True)
class NormalizedDesignRow:
    """One comparable design row independent of source system."""

    system: str
    variant: str
    design_id: str
    sequence: str
    confidence: float
    unique_motifs: int
    total_motifs: int
    fixed_preserved: bool | None = None
    source_path: str = ""
    mode: str = ""
    confidence_metric: str = ""
    confidence_match_status: str = ""
    confidence_path: str = ""
    sequence_path: str = ""
    score_provenance_status: str = ""
    motif_score: float = 0.0
    motif_coverage: float = 0.0
    comparison_score: float = 0.0
    runtime_seconds: float | None = None
    diagnostics_jsonl: str = ""
    longest_homopolymer: int | None = None
    aa_entropy_norm: float | None = None
    seed_sequence_identity: float | None = None
    ca_rmsd_to_seed: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "mode": self.mode,
            "variant": self.variant,
            "design_id": self.design_id,
            "sequence": self.sequence,
            "length": len(self.sequence),
            "confidence": f"{self.confidence:.6f}",
            "confidence_metric": self.confidence_metric,
            "confidence_match_status": self.confidence_match_status,
            "confidence_path": self.confidence_path,
            "unique_motifs": self.unique_motifs,
            "total_motifs": self.total_motifs,
            "motif_score": f"{self.motif_score:.6f}",
            "motif_coverage": f"{self.motif_coverage:.6f}",
            "comparison_score": f"{self.comparison_score:.6f}",
            "fixed_preserved": "" if self.fixed_preserved is None else self.fixed_preserved,
            "runtime_seconds": (
                "" if self.runtime_seconds is None else f"{self.runtime_seconds:.6f}"
            ),
            "diagnostics_jsonl": self.diagnostics_jsonl,
            "longest_homopolymer": (
                "" if self.longest_homopolymer is None else self.longest_homopolymer
            ),
            "aa_entropy_norm": (
                "" if self.aa_entropy_norm is None else f"{self.aa_entropy_norm:.6f}"
            ),
            "seed_sequence_identity": (
                ""
                if self.seed_sequence_identity is None
                else f"{self.seed_sequence_identity:.6f}"
            ),
            "ca_rmsd_to_seed": (
                "" if self.ca_rmsd_to_seed is None else f"{self.ca_rmsd_to_seed:.6f}"
            ),
            "sequence_path": self.sequence_path,
            "score_provenance_status": self.score_provenance_status,
            "source_path": self.source_path,
        }


def normalize_pg_row(
    row: dict[str, str],
    *,
    variant: str,
    source_path: str,
    mode: str = "",
) -> NormalizedDesignRow:
    """Normalize a row from `analyze_motif_outputs.py` or immunogen summary."""

    fixed = _as_bool(row.get("fixed_fragments_preserved", ""))
    confidence = _as_float(_first(row, ["pg_plddt_mean", "mean_plddt", "plddt"]))
    unique = _as_int(_first(row, ["num_unique_motifs_present", "num_unique_motifs"]))
    total = _as_int(
        _first(row, ["num_motif_occurrences_total", "num_motif_hits", "total_motifs"])
    )
    count_required, count_unique, count_total = _motif_counts_from_fields(row)
    if unique == 0 and count_unique:
        unique = count_unique
    if total == 0 and count_total:
        total = count_total
    present = _first(row, ["motifs_present", "unique_motifs", "present_motifs"])
    missing = _first(row, ["motifs_missing", "missing_motifs"])
    required = _required_motif_count(present, missing) or count_required
    coverage = (
        _as_float(row.get("motif_coverage_fraction"))
        if row.get("motif_coverage_fraction", "") != ""
        else (unique / required if required else 0.0)
    )
    motif_score = _as_float(row.get("motif_score"), coverage)
    return NormalizedDesignRow(
        system="ProteinGenerator",
        variant=variant,
        design_id=row.get("design_id", ""),
        sequence=row.get("sequence", ""),
        confidence=confidence,
        unique_motifs=unique,
        total_motifs=total,
        fixed_preserved=fixed,
        source_path=source_path,
        mode=mode,
        confidence_metric="pg_plddt_mean",
        confidence_match_status="same_pdb_ca_bfactor",
        confidence_path=source_path,
        sequence_path=source_path,
        score_provenance_status="pg_sequence_and_plddt_from_same_saved_pdb",
        motif_score=motif_score,
        motif_coverage=coverage,
        comparison_score=_comparison_score(
            confidence=confidence,
            motif_score=motif_score,
            fixed_preserved=fixed,
        ),
        longest_homopolymer=(
            None
            if row.get("longest_homopolymer", "") == ""
            else _as_int(row.get("longest_homopolymer"))
        ),
        aa_entropy_norm=_optional_float(row.get("aa_entropy_norm", "")),
        seed_sequence_identity=_optional_float(row.get("seed_sequence_identity", "")),
        ca_rmsd_to_seed=_optional_float(row.get("ca_rmsd_to_seed", "")),
    )


def normalize_disco_row(
    row: dict[str, str],
    *,
    variant: str,
    source_path: str,
    mode: str = "",
) -> NormalizedDesignRow:
    """Normalize a DISCO evolution or completed-dump analysis row."""

    fixed = _as_bool(row.get("fixed_fragments_preserved", ""))
    confidence = _as_float(row.get("confidence"))
    unique = _as_int(row.get("unique_motifs"))
    total = _as_int(row.get("total_motifs"))
    count_required, count_unique, count_total = _motif_counts_from_fields(row)
    if unique == 0 and count_unique:
        unique = count_unique
    if total == 0 and count_total:
        total = count_total
    present = row.get("present_motifs", "")
    missing = row.get("missing_motifs", "")
    required = _required_motif_count(present, missing) or count_required
    coverage = unique / required if required else 0.0
    motif_score = _as_float(row.get("motif_score"), coverage)
    confidence_status = row.get("confidence_match_status", "")
    evaluation_status = row.get("evaluation_status", "")
    if evaluation_status in {"same_sample_complete", "same_sample_no_diagnostics"}:
        provenance = "disco_exact_sequence_structure_confidence_match"
    elif confidence_status == "exact_sample":
        provenance = "disco_exact_sequence_confidence_match"
    elif confidence_status == "single_confidence_fallback":
        provenance = "disco_single_confidence_fallback"
    elif confidence_status:
        provenance = "disco_unmatched_confidence"
    else:
        provenance = "disco_confidence_provenance_unknown"
    return NormalizedDesignRow(
        system="DISCO",
        variant=variant,
        design_id=row.get("individual", row.get("design_id", "")),
        sequence=row.get("sequence", ""),
        confidence=confidence,
        unique_motifs=unique,
        total_motifs=total,
        fixed_preserved=fixed,
        source_path=source_path,
        mode=mode,
        confidence_metric=row.get("confidence_metric", ""),
        confidence_match_status=confidence_status,
        confidence_path=row.get("confidence_path", ""),
        sequence_path=row.get("sequence_path", ""),
        score_provenance_status=provenance,
        motif_score=motif_score,
        motif_coverage=coverage,
        comparison_score=_comparison_score(
            confidence=confidence,
            motif_score=motif_score,
            fixed_preserved=fixed,
        ),
        runtime_seconds=_optional_float(row.get("runtime_seconds", "")),
        diagnostics_jsonl=row.get("diagnostics_jsonl", ""),
        longest_homopolymer=(
            None
            if row.get("longest_homopolymer", "") == ""
            else _as_int(row.get("longest_homopolymer"))
        ),
        aa_entropy_norm=_optional_float(row.get("aa_entropy_norm", "")),
        seed_sequence_identity=_optional_float(row.get("seed_sequence_identity", "")),
        ca_rmsd_to_seed=_optional_float(row.get("ca_rmsd_to_seed", "")),
    )


def read_normalized_csv(
    path: Path,
    *,
    system: str,
    variant: str | None = None,
    mode: str = "",
) -> list[NormalizedDesignRow]:
    """Read one source CSV into normalized rows."""

    variant_name = variant or path.parent.name
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if system == "ProteinGenerator":
        return [
            normalize_pg_row(
                row,
                variant=variant_name,
                source_path=str(path),
                mode=mode,
            )
            for row in rows
        ]
    if system == "DISCO":
        return [
            normalize_disco_row(
                row,
                variant=variant_name,
                source_path=str(path),
                mode=mode,
            )
            for row in rows
        ]
    raise ValueError(f"Unknown system: {system}")


def summarize_rows(rows: list[NormalizedDesignRow]) -> dict[str, object]:
    """Aggregate normalized rows for one system/variant group."""

    if not rows:
        raise ValueError("Cannot summarize an empty row set")
    fixed_values = [row.fixed_preserved for row in rows if row.fixed_preserved is not None]
    exact_matches = [
        row
        for row in rows
        if row.confidence_match_status in {"exact_sample", "same_pdb_ca_bfactor"}
    ]
    fallback_matches = [
        row for row in rows if row.confidence_match_status == "single_confidence_fallback"
    ]
    runtimes = [row.runtime_seconds for row in rows if row.runtime_seconds is not None]
    entropy_values = [
        row.aa_entropy_norm for row in rows if row.aa_entropy_norm is not None
    ]
    rmsd_values = [row.ca_rmsd_to_seed for row in rows if row.ca_rmsd_to_seed is not None]
    seed_identity_values = [
        row.seed_sequence_identity
        for row in rows
        if row.seed_sequence_identity is not None
    ]
    best_confidence = max(rows, key=lambda row: row.confidence)
    best_comparison = max(rows, key=lambda row: row.comparison_score)
    return {
        "system": rows[0].system,
        "mode": rows[0].mode,
        "variant": rows[0].variant,
        "num_designs": len(rows),
        "mean_confidence": sum(row.confidence for row in rows) / len(rows),
        "best_confidence": best_confidence.confidence,
        "best_confidence_design_id": best_confidence.design_id,
        "mean_unique_motifs": sum(row.unique_motifs for row in rows) / len(rows),
        "best_unique_motifs": max(row.unique_motifs for row in rows),
        "mean_total_motifs": sum(row.total_motifs for row in rows) / len(rows),
        "best_total_motifs": max(row.total_motifs for row in rows),
        "mean_motif_score": sum(row.motif_score for row in rows) / len(rows),
        "best_motif_score": max(row.motif_score for row in rows),
        "mean_comparison_score": (
            sum(row.comparison_score for row in rows) / len(rows)
        ),
        "best_comparison_score": best_comparison.comparison_score,
        "best_comparison_design_id": best_comparison.design_id,
        "fixed_preserved_fraction": (
            sum(1 for value in fixed_values if value) / len(fixed_values)
            if fixed_values
            else ""
        ),
        "confidence_exact_fraction": len(exact_matches) / len(rows),
        "confidence_fallback_fraction": len(fallback_matches) / len(rows),
        "mean_runtime_seconds": (
            sum(runtimes) / len(runtimes)
            if runtimes
            else ""
        ),
        "mean_aa_entropy_norm": (
            sum(entropy_values) / len(entropy_values)
            if entropy_values
            else ""
        ),
        "mean_seed_sequence_identity": (
            sum(seed_identity_values) / len(seed_identity_values)
            if seed_identity_values
            else ""
        ),
        "mean_ca_rmsd_to_seed": (
            sum(rmsd_values) / len(rmsd_values)
            if rmsd_values
            else ""
        ),
    }


def summarize_by_group(rows: list[NormalizedDesignRow]) -> list[dict[str, object]]:
    """Aggregate normalized rows by `(system, variant)`."""

    groups: dict[tuple[str, str], list[NormalizedDesignRow]] = {}
    for row in rows:
        groups.setdefault((row.system, row.variant), []).append(row)
    return [
        summarize_rows(group_rows)
        for _, group_rows in sorted(groups.items(), key=lambda item: item[0])
    ]


def write_normalized_rows(path: Path, rows: list[NormalizedDesignRow]) -> None:
    fieldnames = [
        "system",
        "mode",
        "variant",
        "design_id",
        "sequence",
        "length",
        "confidence",
        "confidence_metric",
        "confidence_match_status",
        "confidence_path",
        "unique_motifs",
        "total_motifs",
        "motif_score",
        "motif_coverage",
        "comparison_score",
        "fixed_preserved",
        "runtime_seconds",
        "diagnostics_jsonl",
        "longest_homopolymer",
        "aa_entropy_norm",
        "seed_sequence_identity",
        "ca_rmsd_to_seed",
        "sequence_path",
        "score_provenance_status",
        "source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.as_dict() for row in rows])


def write_summary_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "system",
        "mode",
        "variant",
        "num_designs",
        "mean_confidence",
        "best_confidence",
        "best_confidence_design_id",
        "mean_unique_motifs",
        "best_unique_motifs",
        "mean_total_motifs",
        "best_total_motifs",
        "mean_motif_score",
        "best_motif_score",
        "mean_comparison_score",
        "best_comparison_score",
        "best_comparison_design_id",
        "fixed_preserved_fraction",
        "confidence_exact_fraction",
        "confidence_fallback_fraction",
        "mean_runtime_seconds",
        "mean_aa_entropy_norm",
        "mean_seed_sequence_identity",
        "mean_ca_rmsd_to_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
