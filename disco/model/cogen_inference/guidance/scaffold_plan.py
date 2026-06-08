"""Sequence-level scaffold input planning for DISCO motif experiments.

ProteinGenerator's contig machinery can hold PDB-derived structure regions and
independently free sequence or structure positions.  DISCO's current inference
JSON path does not expose that same PG/RFdiffusion-style contig interface.  The
portable subset is therefore narrower: write DISCO `proteinChain.sequence`
records where fixed amino-acid letters are preserved and `-` positions remain
designable.

This module is intentionally a planner, not a sampler hook.  It produces input
JSONs plus a manifest that says exactly which PG scaffold semantics were
preserved and which structural semantics were not.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from disco.model.cogen_inference.guidance.experiment_plan import FixedFragmentPlan
from disco.model.cogen_inference.guidance.immunogen import AMINO_ACIDS, spec_from_dict

MASK_CHAR = "-"


@dataclass(frozen=True)
class ScaffoldFixedRange:
    """1-based inclusive range copied from a seed sequence."""

    start_1based: int
    end_1based: int
    label: str = ""

    @property
    def start_0based(self) -> int:
        return self.start_1based - 1

    @property
    def end_0based_exclusive(self) -> int:
        return self.end_1based

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "start_1based": self.start_1based,
            "end_1based": self.end_1based,
        }
        if self.label:
            row["label"] = self.label
        return row


@dataclass(frozen=True)
class ScaffoldIssue:
    """Structured planner finding for unsupported or suspicious inputs."""

    severity: str
    code: str
    message: str
    value: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ScaffoldPlanResult:
    """Materialized DISCO input plus audit metadata."""

    input_json_path: Path
    manifest_json_path: Path
    sequence: str
    issues: list[ScaffoldIssue]

    @property
    def has_unsupported_pg_semantics(self) -> bool:
        return any(issue.severity == "unsupported" for issue in self.issues)

    @property
    def status(self) -> str:
        if self.has_unsupported_pg_semantics:
            return "partial_sequence_scaffold_with_unsupported_pg_structure_semantics"
        return "ready_sequence_scaffold"


def parse_fixed_range(value: str) -> ScaffoldFixedRange:
    """Parse `start:end[:label]` into a fixed seed-sequence range."""

    parts = value.split(":", 2)
    if len(parts) < 2:
        raise ValueError("fixed ranges must use start_1based:end_1based[:label]")
    start_1based = int(parts[0])
    end_1based = int(parts[1])
    label = parts[2].strip() if len(parts) == 3 else ""
    if start_1based < 1:
        raise ValueError("fixed range start_1based must be >= 1")
    if end_1based < start_1based:
        raise ValueError("fixed range end_1based must be >= start_1based")
    return ScaffoldFixedRange(
        start_1based=start_1based,
        end_1based=end_1based,
        label=label,
    )


def fixed_fragments_from_json(path: Path) -> list[FixedFragmentPlan]:
    """Load the PG/DISCO immunogen fixed-fragment JSON schema."""

    spec = spec_from_dict(json.loads(path.read_text()))
    return [
        FixedFragmentPlan(
            start_1based=fragment.start_1based,
            sequence=fragment.sequence,
            label=fragment.label,
        )
        for fragment in spec.fixed_fragments
    ]


def _normalize_seed_sequence(seed_sequence: str | None) -> str | None:
    if seed_sequence is None:
        return None
    normalized = "".join(seed_sequence.split()).upper()
    if not normalized:
        raise ValueError("seed sequence must not be empty")
    invalid = sorted(set(normalized) - AMINO_ACIDS)
    if invalid:
        raise ValueError(f"seed sequence contains non-standard amino acids: {invalid}")
    return normalized


def _validate_fixed_fragment(fragment: FixedFragmentPlan) -> None:
    if fragment.start_1based < 1:
        raise ValueError("fixed fragment start_1based must be >= 1")
    if not fragment.sequence:
        raise ValueError("fixed fragment sequence must not be empty")
    invalid = sorted(set(fragment.sequence.upper()) - AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"fixed fragment {fragment.label or fragment.sequence!r} contains "
            f"non-standard amino acids: {invalid}"
        )


def _place_letters(
    sequence: list[str],
    occupied_by: list[str | None],
    *,
    start_1based: int,
    letters: str,
    label: str,
) -> None:
    start_0based = start_1based - 1
    end = start_0based + len(letters)
    if start_1based < 1:
        raise ValueError(f"{label!r} starts before position 1")
    if end > len(sequence):
        raise ValueError(
            f"{label!r} at {start_1based} extends past scaffold length {len(sequence)}"
        )

    for offset, aa in enumerate(letters):
        position = start_0based + offset
        owner = occupied_by[position]
        existing = sequence[position]
        if owner is not None and existing != aa:
            raise ValueError(
                f"conflicting fixed residues at position {position + 1}: "
                f"{owner!r} has {existing!r}, {label!r} wants {aa!r}"
            )
        occupied_by[position] = label
        sequence[position] = aa


def sequence_from_scaffold_constraints(
    *,
    length: int,
    fixed_fragments: Iterable[FixedFragmentPlan] = (),
    fixed_ranges: Iterable[ScaffoldFixedRange] = (),
    seed_sequence: str | None = None,
) -> str:
    """Return a DISCO `proteinChain.sequence` scaffold string.

    Fixed fragments provide literal amino-acid letters.  Fixed ranges copy the
    letters from `seed_sequence`.  All remaining positions are `-`, which DISCO's
    cogen input path treats as masked/designable protein residues.
    """

    if length < 1:
        raise ValueError("length must be >= 1")
    seed_sequence = _normalize_seed_sequence(seed_sequence)
    if seed_sequence is not None and len(seed_sequence) != length:
        raise ValueError(
            f"seed sequence length {len(seed_sequence)} does not match scaffold length {length}"
        )

    sequence = [MASK_CHAR] * length
    occupied_by: list[str | None] = [None] * length

    for fragment in fixed_fragments:
        _validate_fixed_fragment(fragment)
        label = fragment.label or fragment.sequence
        _place_letters(
            sequence,
            occupied_by,
            start_1based=fragment.start_1based,
            letters=fragment.sequence.upper(),
            label=label,
        )

    for fixed_range in fixed_ranges:
        if seed_sequence is None:
            raise ValueError("fixed ranges require --seed-sequence")
        if fixed_range.end_0based_exclusive > length:
            raise ValueError(
                f"fixed range {fixed_range.start_1based}:{fixed_range.end_1based} "
                f"extends past scaffold length {length}"
            )
        letters = seed_sequence[
            fixed_range.start_0based : fixed_range.end_0based_exclusive
        ]
        label = fixed_range.label or f"seed_range_{fixed_range.start_1based}_{fixed_range.end_1based}"
        _place_letters(
            sequence,
            occupied_by,
            start_1based=fixed_range.start_1based,
            letters=letters,
            label=label,
        )

    return "".join(sequence)


_CHAIN_RANGE_RE = re.compile(r"\b[A-Za-z][0-9]+(?:-[0-9]+)?\b")
_INPAINT_RE = re.compile(r"\b[A-Za-z][0-9]+(?:-[0-9]+)?\b|[,:/]")


def audit_pg_scaffold_arguments(
    *,
    pg_contigs: Sequence[str] = (),
    pg_seed_pdb: Path | None = None,
    pg_inpaint_seq: Sequence[str] = (),
    pg_inpaint_str: Sequence[str] = (),
) -> list[ScaffoldIssue]:
    """Report PG scaffold semantics that this DISCO planner cannot preserve."""

    issues: list[ScaffoldIssue] = []
    if pg_seed_pdb is not None:
        issues.append(
            ScaffoldIssue(
                severity="unsupported",
                code="pg_seed_pdb_structure_template_not_ported",
                message=(
                    "DISCO input planning records the seed PDB path for provenance, "
                    "but does not hold those coordinates as a PG-style structure template."
                ),
                value=str(pg_seed_pdb),
            )
        )

    for contig in pg_contigs:
        stripped = str(contig).strip()
        if not stripped or stripped == "0":
            continue
        if _CHAIN_RANGE_RE.search(stripped) or "," in stripped or "/" in stripped:
            issues.append(
                ScaffoldIssue(
                    severity="unsupported",
                    code="pg_contig_structure_mask_not_ported",
                    message=(
                        "PG contig strings can select PDB-derived structural template "
                        "residues. This planner does not translate that structure mask; "
                        "use fixed fragments/ranges for sequence-level constraints only."
                    ),
                    value=stripped,
                )
            )
        else:
            issues.append(
                ScaffoldIssue(
                    severity="warning",
                    code="pg_contig_not_interpreted",
                    message=(
                        "The PG contig argument was recorded for provenance but not "
                        "used to derive DISCO positions."
                    ),
                    value=stripped,
                )
            )

    for label, values in (
        ("pg_inpaint_seq_not_ported", pg_inpaint_seq),
        ("pg_inpaint_str_not_ported", pg_inpaint_str),
    ):
        for value in values:
            stripped = str(value).strip()
            if not stripped:
                continue
            severity = "unsupported" if _INPAINT_RE.search(stripped) else "warning"
            issues.append(
                ScaffoldIssue(
                    severity=severity,
                    code=label,
                    message=(
                        "DISCO does not expose PG's separate inpaint_seq/inpaint_str "
                        "mask controls through this planner."
                    ),
                    value=stripped,
                )
            )

    return issues


def build_disco_input_record(*, name: str, sequence: str, count: int = 1) -> list[dict]:
    """Build the minimal DISCO JSON record for one protein design job."""

    if count < 1:
        raise ValueError("count must be >= 1")
    return [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": count,
                    }
                }
            ],
            "name": name,
        }
    ]


def write_scaffold_plan(
    *,
    input_json_path: Path,
    manifest_json_path: Path,
    length: int,
    name: str,
    fixed_fragments: Iterable[FixedFragmentPlan] = (),
    fixed_ranges: Iterable[ScaffoldFixedRange] = (),
    seed_sequence: str | None = None,
    pg_contigs: Sequence[str] = (),
    pg_seed_pdb: Path | None = None,
    pg_inpaint_seq: Sequence[str] = (),
    pg_inpaint_str: Sequence[str] = (),
    count: int = 1,
) -> ScaffoldPlanResult:
    """Write a DISCO input JSON and a manifest for a sequence scaffold plan."""

    fixed_fragments = list(fixed_fragments)
    fixed_ranges = list(fixed_ranges)
    issues = audit_pg_scaffold_arguments(
        pg_contigs=pg_contigs,
        pg_seed_pdb=pg_seed_pdb,
        pg_inpaint_seq=pg_inpaint_seq,
        pg_inpaint_str=pg_inpaint_str,
    )
    sequence = sequence_from_scaffold_constraints(
        length=length,
        fixed_fragments=fixed_fragments,
        fixed_ranges=fixed_ranges,
        seed_sequence=seed_sequence,
    )

    input_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json_path.parent.mkdir(parents=True, exist_ok=True)
    input_json_path.write_text(
        json.dumps(build_disco_input_record(name=name, sequence=sequence, count=count), indent=2)
        + "\n"
    )

    result = ScaffoldPlanResult(
        input_json_path=input_json_path,
        manifest_json_path=manifest_json_path,
        sequence=sequence,
        issues=issues,
    )
    manifest = {
        "status": result.status,
        "input_json_path": str(input_json_path),
        "sequence": sequence,
        "length": length,
        "name": name,
        "count": count,
        "num_fixed_positions": sum(aa != MASK_CHAR for aa in sequence),
        "num_designable_positions": sum(aa == MASK_CHAR for aa in sequence),
        "fixed_fragments": [fragment.as_dict() for fragment in fixed_fragments],
        "fixed_ranges": [fixed_range.as_dict() for fixed_range in fixed_ranges],
        "seed_sequence_provided": seed_sequence is not None,
        "pg_seed_pdb": "" if pg_seed_pdb is None else str(pg_seed_pdb),
        "pg_contigs": list(pg_contigs),
        "pg_inpaint_seq": list(pg_inpaint_seq),
        "pg_inpaint_str": list(pg_inpaint_str),
        "issues": [issue.as_dict() for issue in issues],
        "disco_semantics": (
            "Fixed amino-acid letters are preserved in proteinChain.sequence; "
            "`-` positions are designable/masked."
        ),
        "not_ported_semantics": (
            "PG/RFdiffusion-style coordinate contigs, fixed-backbone templates, "
            "and independent inpaint_seq/inpaint_str structure masks."
        ),
    }
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result
