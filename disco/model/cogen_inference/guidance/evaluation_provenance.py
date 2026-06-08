"""Evaluation-provenance checks for DISCO/ProteinGenerator comparisons.

ProteinGenerator's immunogen ``--immunogen_repredict`` flag exists for one
reason: the saved sequence, coordinates, and pLDDT must describe the same final
design state.  DISCO stores sequence, structure, confidence JSON, and optional
diagnostics as separate files, so the equivalent guard is a file-provenance
check.  This module keeps that check explicit and reusable by analyzers and
quality gates.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


SAMPLE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_sample_(?P<sample>\d+)$")

EXACT_CONFIDENCE_STATUSES = {"exact_sample"}
EXACT_STRUCTURE_STATUSES = {
    "exact_structure_stem",
    "exact_sample",
    "exact_sequence_index",
}
ACCEPTED_DISCO_BUNDLE_STATUSES = {
    "same_sample_complete",
    "same_sample_no_diagnostics",
}
ACCEPTED_SCORE_PROVENANCE_STATUSES = {
    "pg_sequence_and_plddt_from_same_saved_pdb",
    "disco_exact_sequence_structure_confidence_match",
}


@dataclass(frozen=True)
class EvaluationBundle:
    """One row-level statement about whether all metrics share one sample."""

    status: str
    expected_sample_keys: str
    confidence_sample_key: str
    structure_sample_key: str
    diagnostics_status: str
    notes: str

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        return {f"evaluation_{key}": value for key, value in values.items()}


def split_sample_suffix(value: str) -> tuple[str, int | None]:
    """Split `name_sample_0` into (`name`, 0) when that suffix is present."""

    match = SAMPLE_SUFFIX_RE.match(value)
    if not match:
        return value, None
    return match.group("base"), int(match.group("sample"))


def sample_key(base_id: str | None, sample_index: int | None) -> str:
    """Return a stable human-readable sample key."""

    if base_id in ["", None] or sample_index is None:
        return ""
    return f"{base_id}:sample_{int(sample_index)}"


def expected_sample_pairs(sequence_base_id: str, sequence_index: int) -> set[tuple[str, int]]:
    """Return sample identities acceptable for one parsed sequence row."""

    pairs = {(sequence_base_id, int(sequence_index))}
    base_without_sample, sample_from_suffix = split_sample_suffix(sequence_base_id)
    if sample_from_suffix is not None:
        pairs.add((base_without_sample, sample_from_suffix))
    return pairs


def _pair_matches_expected(
    *,
    base_id: str | None,
    sample_index: int | None,
    expected_pairs: set[tuple[str, int]],
) -> bool:
    if base_id in ["", None] or sample_index is None:
        return False
    return (str(base_id), int(sample_index)) in expected_pairs


def build_evaluation_bundle(
    *,
    sequence_base_id: str,
    sequence_index: int,
    confidence_base_id: str | None = None,
    confidence_sample_index: int | None = None,
    confidence_match_status: str = "",
    structure_base_id: str | None = None,
    structure_sample_index: int | None = None,
    structure_match_status: str = "",
    diagnostics_present: bool = False,
) -> EvaluationBundle:
    """Evaluate whether sequence, confidence, structure, and diagnostics align."""

    expected_pairs = expected_sample_pairs(sequence_base_id, sequence_index)
    expected_keys = ",".join(
        sample_key(base, index) for base, index in sorted(expected_pairs)
    )
    confidence_key = sample_key(confidence_base_id, confidence_sample_index)
    structure_key = sample_key(structure_base_id, structure_sample_index)
    diagnostics_status = "exact_base" if diagnostics_present else "missing"
    notes: list[str] = []

    confidence_exact = (
        confidence_match_status in EXACT_CONFIDENCE_STATUSES
        and _pair_matches_expected(
            base_id=confidence_base_id,
            sample_index=confidence_sample_index,
            expected_pairs=expected_pairs,
        )
    )
    structure_exact = (
        structure_match_status in EXACT_STRUCTURE_STATUSES
        and _pair_matches_expected(
            base_id=structure_base_id,
            sample_index=structure_sample_index,
            expected_pairs=expected_pairs,
        )
    )

    if not confidence_match_status or confidence_match_status == "missing":
        notes.append("missing confidence")
    elif not confidence_exact:
        notes.append(f"confidence status/key not exact: {confidence_match_status}")

    if not structure_match_status or structure_match_status == "missing_structure":
        notes.append("missing structure")
    elif not structure_exact:
        notes.append(f"structure status/key not exact: {structure_match_status}")

    if confidence_exact and structure_exact and diagnostics_present:
        status = "same_sample_complete"
    elif confidence_exact and structure_exact:
        status = "same_sample_no_diagnostics"
    elif not confidence_match_status or confidence_match_status == "missing":
        status = "missing_confidence"
    elif not structure_match_status or structure_match_status == "missing_structure":
        status = "missing_structure"
    elif not confidence_exact or not structure_exact:
        status = "sample_mismatch_or_fallback"
    else:
        status = "unknown"

    return EvaluationBundle(
        status=status,
        expected_sample_keys=expected_keys,
        confidence_sample_key=confidence_key,
        structure_sample_key=structure_key,
        diagnostics_status=diagnostics_status,
        notes="; ".join(notes),
    )


def row_has_accepted_same_artifact_provenance(row: dict[str, str]) -> bool:
    """Return whether a summary/normalized row proves same-artifact scoring."""

    bundle_status = str(row.get("evaluation_status", "")).strip()
    if bundle_status in ACCEPTED_DISCO_BUNDLE_STATUSES:
        return True

    provenance = str(row.get("score_provenance_status", "")).strip()
    if provenance in ACCEPTED_SCORE_PROVENANCE_STATUSES:
        return True

    return False
