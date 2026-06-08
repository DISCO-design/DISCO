"""Analyze finished DISCO output folders for motif-comparison studies.

The functions here are deliberately file-based.  They do not start DISCO
inference and they keep sequence, confidence, and motif metrics tied to the
same output sample whenever the filenames expose a matching sample id.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from disco.model.cogen_inference.guidance.alternating import (
    combined_alternating_score,
    parse_motif_list,
    score_alternation,
)
from disco.model.cogen_inference.guidance.evaluation_provenance import (
    build_evaluation_bundle,
)
from disco.model.cogen_inference.guidance.immunogen import (
    ImmunogenFixedSpec,
    score_fixed_fragments,
    spacing_metrics,
)
from disco.model.cogen_inference.guidance.scoring import (
    HardMotifScorer,
    parse_motifs,
)

AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")
SEQUENCE_TEXT_CHARS = AMINO_ACIDS | {"-"}
CONFIDENCE_RE = re.compile(r"^(?P<base>.+)_summary_confidence_sample_(?P<sample>\d+)$")
DIAGNOSTIC_METADATA_RE = re.compile(r"^(?P<base>.+)_run_metadata$")
SAMPLE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_sample_(?P<sample>\d+)$")


@dataclass(frozen=True)
class DiscoSequenceRecord:
    """One concrete protein sequence parsed from a DISCO sequence text file."""

    design_id: str
    base_id: str
    sequence_index: int
    sequence_path: Path
    sequence: str


@dataclass(frozen=True)
class DiscoConfidenceRecord:
    """One DISCO summary-confidence JSON tied to a structure sample."""

    base_id: str
    sample_index: int
    confidence: float
    metric: str
    confidence_path: Path


@dataclass(frozen=True)
class DiscoStructureRecord:
    """One saved DISCO structure file tied to a structure sample index."""

    base_id: str
    sample_index: int
    structure_path: Path
    file_format: str


def split_sample_suffix(value: str) -> tuple[str, int | None]:
    """Split `foo_sample_0` into (`foo`, 0) when possible."""

    match = SAMPLE_SUFFIX_RE.match(value)
    if not match:
        return value, None
    return match.group("base"), int(match.group("sample"))


def structure_records_in_dump(dump_dir: Path) -> list[DiscoStructureRecord]:
    """Return saved PDB/CIF structure records under a DISCO dump directory."""

    paths = sorted(
        [
            *list((dump_dir / "pdbs").glob("**/*.pdb")),
            *list((dump_dir / "pdbs").glob("**/*.cif")),
        ]
    )
    records: list[DiscoStructureRecord] = []
    for path in paths:
        base_id, sample_index = split_sample_suffix(path.stem)
        if sample_index is None:
            continue
        records.append(
            DiscoStructureRecord(
                base_id=base_id,
                sample_index=sample_index,
                structure_path=path,
                file_format=path.suffix.lstrip(".").lower(),
            )
        )
    return records


def longest_homopolymer(sequence: str) -> int:
    """Return the longest run of one repeated amino acid."""

    if not sequence:
        return 0
    longest = 1
    current = 1
    for previous, current_aa in zip(sequence, sequence[1:]):
        if previous == current_aa:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def normalized_entropy(sequence: str) -> float:
    """Return amino-acid diversity normalized to the range 0..1."""

    if not sequence:
        return 0.0
    counts: dict[str, int] = {}
    for aa in sequence:
        counts[aa] = counts.get(aa, 0) + 1
    entropy = 0.0
    for count in counts.values():
        probability = count / len(sequence)
        entropy -= probability * math.log(probability)
    return entropy / math.log(20)


def sequence_distance(sequence: str, seed_sequence: str | None) -> dict[str, object]:
    """Measure sequence identity and changed positions against a seed string."""

    if seed_sequence is None:
        return {
            "seed_sequence_identity": "",
            "num_sequence_changes_vs_seed": "",
            "changed_positions_1based": "",
        }
    seed_sequence = seed_sequence.upper()
    if len(sequence) != len(seed_sequence):
        return {
            "seed_sequence_identity": "",
            "num_sequence_changes_vs_seed": "",
            "changed_positions_1based": "length_mismatch",
        }
    changed = [
        index + 1
        for index, (observed, seed) in enumerate(zip(sequence, seed_sequence))
        if observed != seed
    ]
    identity = 1.0 - len(changed) / max(len(sequence), 1)
    return {
        "seed_sequence_identity": f"{identity:.3f}",
        "num_sequence_changes_vs_seed": len(changed),
        "changed_positions_1based": ";".join(str(position) for position in changed),
    }


def ca_coordinates_from_pdb(pdb_path: Path) -> np.ndarray:
    """Extract CA atom coordinates from a PDB file."""

    coordinates = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        coordinates.append(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]
        )
    return np.array(coordinates, dtype=float)


def kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Calculate CA-RMSD after optimal rigid alignment."""

    if mobile.shape != reference.shape or mobile.size == 0:
        return float("nan")
    mobile_centered = mobile - mobile.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    covariance = mobile_centered.T @ reference_centered
    v, _, wt = np.linalg.svd(covariance)
    if np.linalg.det(v @ wt) < 0:
        v[:, -1] *= -1
    rotation = v @ wt
    aligned = mobile_centered @ rotation
    diff = aligned - reference_centered
    return float(np.sqrt((diff * diff).sum() / mobile.shape[0]))


def confidence_from_json(path: Path) -> tuple[float, str]:
    """Read a normalized confidence-like value and the source metric name."""

    data = json.loads(path.read_text())
    for key in ["ranking_score", "plddt", "mean_plddt", "ptm"]:
        if key not in data:
            continue
        value = float(data[key])
        return (value / 100.0 if value > 1.0 else value), key
    return 0.0, "missing"


def confidence_records_in_dump(dump_dir: Path) -> list[DiscoConfidenceRecord]:
    """Return all summary confidence records under a DISCO dump directory."""

    paths = sorted(set((dump_dir / "pdbs").glob("**/*summary_confidence*.json")))
    records: list[DiscoConfidenceRecord] = []
    for path in paths:
        match = CONFIDENCE_RE.match(path.stem)
        if not match:
            continue
        confidence, metric = confidence_from_json(path)
        records.append(
            DiscoConfidenceRecord(
                base_id=match.group("base"),
                sample_index=int(match.group("sample")),
                confidence=confidence,
                metric=metric,
                confidence_path=path,
            )
        )
    return records


def sequence_entries_from_disco_sequence_text(text: str) -> list[tuple[int, str]]:
    """Extract one or more protein sequences from a DISCO sequence text file."""

    entries: list[tuple[int, str]] = []
    current_index = 0
    current_residues: list[str] = []
    saw_header = False

    def flush() -> None:
        if current_residues:
            entries.append((current_index, "".join(current_residues)))
            current_residues.clear()

    def normalize_sequence_line(line: str) -> str:
        sequence = line.strip().upper()
        if set(sequence).issubset(SEQUENCE_TEXT_CHARS):
            return sequence
        return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("ligand_smiles"):
            continue
        if line.startswith(">"):
            flush()
            saw_header = True
            parts = line.split()
            current_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else len(entries)
            continue
        if line.startswith("CHAIN "):
            parts = dict(item.split("=", 1) for item in line.split() if "=" in item)
            if parts.get("type") == "protein" and "seq" in parts:
                sequence = normalize_sequence_line(parts["seq"])
                if sequence:
                    current_residues.append(sequence)
            continue
        sequence = normalize_sequence_line(line)
        if sequence:
            current_residues.append(sequence)

    flush()
    if entries:
        return entries
    if not saw_header and text.strip():
        sequence = "".join(
            normalize_sequence_line(line)
            for line in text.splitlines()
        )
        if sequence:
            return [(0, sequence)]
    return []


def sequence_records_in_dump(dump_dir: Path) -> list[DiscoSequenceRecord]:
    """Return concrete sequence records from `dump_dir/sequences/*.txt`."""

    records: list[DiscoSequenceRecord] = []
    for path in sorted((dump_dir / "sequences").glob("*.txt")):
        entries = sequence_entries_from_disco_sequence_text(path.read_text())
        for sequence_index, sequence in entries:
            design_id = path.stem if len(entries) == 1 else f"{path.stem}_seq{sequence_index}"
            records.append(
                DiscoSequenceRecord(
                    design_id=design_id,
                    base_id=path.stem,
                    sequence_index=sequence_index,
                    sequence_path=path,
                    sequence=sequence,
                )
            )
    return records


def diagnostic_metadata_in_dump(dump_dir: Path) -> dict[str, dict[str, object]]:
    """Return run diagnostic metadata keyed by DISCO sequence base id."""

    metadata = {}
    for path in sorted((dump_dir / "diagnostics").glob("*_run_metadata.json")):
        match = DIAGNOSTIC_METADATA_RE.match(path.stem)
        if not match:
            continue
        data = json.loads(path.read_text())
        data["diagnostics_manifest"] = str(path)
        metadata[match.group("base")] = data
    return metadata


def provenance_metadata_in_dump(dump_dir: Path) -> dict[str, dict[str, object]]:
    """Return artifact-provenance metadata keyed by DISCO sequence base id."""

    metadata = {}
    for path in sorted((dump_dir / "provenance").glob("*_artifact_provenance.json")):
        data = json.loads(path.read_text())
        sample_id = str(data.get("sample_id") or path.stem.removesuffix("_artifact_provenance"))
        data["artifact_provenance_manifest"] = str(path)
        index_path = data.get("intermediate_state_index", {})
        if isinstance(index_path, dict):
            data["intermediate_state_index"] = index_path.get("path", "")
        metadata[sample_id] = data
    return metadata


def match_confidence_record(
    sequence: DiscoSequenceRecord,
    confidence_records: Iterable[DiscoConfidenceRecord],
) -> tuple[DiscoConfidenceRecord | None, str]:
    """Return the confidence record that belongs to a sequence sample."""

    sequence_base_without_sample, sequence_sample = split_sample_suffix(sequence.base_id)
    candidates = [
        record
        for record in confidence_records
        if record.base_id in {sequence.base_id, sequence_base_without_sample}
    ]
    if not candidates:
        return None, "missing"

    exact = [
        record
        for record in candidates
        if record.base_id == sequence.base_id
        and record.sample_index == sequence.sequence_index
    ]
    if not exact and sequence_sample is not None:
        exact = [
            record
            for record in candidates
            if record.base_id == sequence_base_without_sample
            and record.sample_index == sequence_sample
        ]
    if not exact:
        exact = [
            record
            for record in candidates
            if record.base_id == sequence_base_without_sample
            and record.sample_index == sequence.sequence_index
        ]
    if exact:
        return exact[0], "exact_sample"

    if len(candidates) == 1:
        return candidates[0], "single_confidence_fallback"

    return None, "missing_exact_sample"


def match_structure_record(
    sequence: DiscoSequenceRecord,
    structure_records: Iterable[DiscoStructureRecord],
) -> tuple[DiscoStructureRecord | None, str]:
    """Return the saved structure file that belongs to a sequence sample."""

    sequence_base_without_sample, sequence_sample = split_sample_suffix(sequence.base_id)
    records = list(structure_records)
    exact_stem = [
        record for record in records if record.structure_path.stem == sequence.base_id
    ]
    if exact_stem:
        return exact_stem[0], "exact_structure_stem"

    candidates = [
        record
        for record in records
        if record.base_id in {sequence.base_id, sequence_base_without_sample}
    ]
    if sequence_sample is not None:
        exact_sample = [
            record
            for record in candidates
            if record.base_id == sequence_base_without_sample
            and record.sample_index == sequence_sample
        ]
        if exact_sample:
            return exact_sample[0], "exact_sample"
    exact_index = [
        record
        for record in candidates
        if record.sample_index == sequence.sequence_index
    ]
    if exact_index:
        return exact_index[0], "exact_sequence_index"
    if len(candidates) == 1:
        return candidates[0], "single_structure_fallback"

    return None, "missing_structure"


def disco_design_row(
    *,
    sequence: DiscoSequenceRecord,
    confidence: DiscoConfidenceRecord | None,
    confidence_match_status: str,
    motifs: Iterable[str],
    motif_all_weight: float = 0.8,
    motif_many_weight: float = 0.2,
    motif_many_cap: int = 0,
    fixed_spec: ImmunogenFixedSpec | None = None,
    tcell_motifs: Iterable[str] | None = None,
    cleavage_motifs: Iterable[str] | None = None,
    alternating_list1: Iterable[str] | None = None,
    alternating_list2: Iterable[str] | None = None,
    expected_start_group: str = "list1",
    diagnostic_metadata: dict[str, object] | None = None,
    provenance_metadata: dict[str, object] | None = None,
    structure: DiscoStructureRecord | None = None,
    structure_match_status: str = "",
    seed_sequence: str | None = None,
    seed_pdb_ca: np.ndarray | None = None,
) -> dict[str, object]:
    """Build one CSV-ready DISCO analysis row for a finished sequence."""

    motif_list = parse_motifs(motifs)
    scorer = HardMotifScorer(
        motifs=motif_list,
        motif_all_weight=motif_all_weight,
        motif_many_weight=motif_many_weight,
        motif_many_cap=motif_many_cap,
    )
    motif_score = scorer.score_sequence(sequence.sequence)

    row: dict[str, object] = {
        "design_id": sequence.design_id,
        "base_id": sequence.base_id,
        "sequence_index": sequence.sequence_index,
        "sequence": sequence.sequence,
        "sequence_length": len(sequence.sequence),
        "sequence_path": str(sequence.sequence_path),
        "structure_path": str(structure.structure_path) if structure else "",
        "structure_format": structure.file_format if structure else "",
        "structure_match_status": structure_match_status,
        "confidence": f"{(confidence.confidence if confidence else 0.0):.6f}",
        "confidence_metric": confidence.metric if confidence else "",
        "confidence_path": str(confidence.confidence_path) if confidence else "",
        "confidence_match_status": confidence_match_status,
        "motif_score": f"{motif_score.motif_score:.6f}",
        "unique_motifs": motif_score.num_unique,
        "total_motifs": motif_score.num_total,
        "present_motifs": ",".join(motif_score.present),
        "missing_motifs": ",".join(motif_score.missing),
        "motif_positions_json": json.dumps(motif_score.positions, sort_keys=True),
    }
    row.update(sequence_distance(sequence.sequence, seed_sequence))
    bundle = build_evaluation_bundle(
        sequence_base_id=sequence.base_id,
        sequence_index=sequence.sequence_index,
        confidence_base_id=confidence.base_id if confidence is not None else None,
        confidence_sample_index=(
            confidence.sample_index if confidence is not None else None
        ),
        confidence_match_status=confidence_match_status,
        structure_base_id=structure.base_id if structure is not None else None,
        structure_sample_index=structure.sample_index if structure is not None else None,
        structure_match_status=structure_match_status,
        diagnostics_present=diagnostic_metadata is not None,
    )
    row.update(bundle.as_dict())
    row["longest_homopolymer"] = longest_homopolymer(sequence.sequence)
    row["aa_entropy_norm"] = f"{normalized_entropy(sequence.sequence):.3f}"
    row["ca_rmsd_to_seed"] = ""
    if seed_pdb_ca is not None and structure is not None and structure.file_format == "pdb":
        ca_rmsd = kabsch_rmsd(ca_coordinates_from_pdb(structure.structure_path), seed_pdb_ca)
        row["ca_rmsd_to_seed"] = "" if math.isnan(ca_rmsd) else f"{ca_rmsd:.3f}"
    if diagnostic_metadata is not None:
        row.update(
            {
                "runtime_seconds": diagnostic_metadata.get("runtime_seconds", ""),
                "diagnostics_jsonl": diagnostic_metadata.get("diagnostics_jsonl", ""),
                "diagnostics_csv": diagnostic_metadata.get("diagnostics_csv", ""),
                "diagnostics_manifest": diagnostic_metadata.get(
                    "diagnostics_manifest", ""
                ),
                "num_diagnostic_records": diagnostic_metadata.get(
                    "num_diagnostic_records", ""
                ),
                "diagnostic_best_step": diagnostic_metadata.get("best_step", ""),
                "diagnostic_best_score": diagnostic_metadata.get("best_score", ""),
                "diagnostic_final_step": diagnostic_metadata.get("final_step", ""),
                "diagnostic_best_output_selected": diagnostic_metadata.get(
                    "best_output_selected", ""
                ),
            }
        )
    if provenance_metadata is not None:
        row.update(
            {
                "artifact_provenance_manifest": provenance_metadata.get(
                    "artifact_provenance_manifest", ""
                ),
                "intermediate_state_index": provenance_metadata.get(
                    "intermediate_state_index", ""
                ),
                "num_intermediate_states": provenance_metadata.get(
                    "num_intermediate_states", ""
                ),
                "trajectory_manifest": (
                    provenance_metadata.get("trajectory_manifest", {}) or {}
                ).get("path", ""),
            }
        )
    for motif in motif_list:
        row[f"{motif}_count"] = motif_score.counts.get(motif, 0)

    if fixed_spec is not None:
        fixed_score = score_fixed_fragments(sequence.sequence, fixed_spec)
        row.update(
            {
                "fixed_fragments_preserved": fixed_score["fixed_fragments_preserved"],
                "num_fixed_mismatches": fixed_score["num_fixed_mismatches"],
                "fixed_mismatches_json": json.dumps(
                    fixed_score["mismatches"], sort_keys=True
                ),
            }
        )

    tcell_list = parse_motif_list(tcell_motifs)
    if tcell_list:
        tcell_score = HardMotifScorer(motifs=tcell_list).score_sequence(
            sequence.sequence
        )
        row["tcell_unique_motifs"] = tcell_score.num_unique
        row["tcell_total_motifs"] = tcell_score.num_total
        row["tcell_missing_motifs"] = ",".join(tcell_score.missing)

    cleavage_list = parse_motif_list(cleavage_motifs)
    if cleavage_list:
        cleavage_score = HardMotifScorer(motifs=cleavage_list).score_sequence(
            sequence.sequence
        )
        row["cleavage_unique_motifs"] = cleavage_score.num_unique
        row["cleavage_total_motifs"] = cleavage_score.num_total
        row["cleavage_missing_motifs"] = ",".join(cleavage_score.missing)

    list1 = parse_motif_list(alternating_list1)
    list2 = parse_motif_list(alternating_list2)
    if list1 and list2:
        alternating = score_alternation(
            sequence.sequence,
            list1,
            list2,
            expected_start_group=expected_start_group,
        )
        row["alternating_combined_score"] = f"{combined_alternating_score(sequence.sequence, list1, list2):.6f}"
        for key, value in alternating.items():
            row[f"alternating_{key}"] = value

    if motif_list:
        row.update(spacing_metrics(sequence.sequence, motif_list))

    return row


def analyze_disco_dump_rows(
    *,
    dump_dir: Path,
    motifs: Iterable[str],
    motif_all_weight: float = 0.8,
    motif_many_weight: float = 0.2,
    motif_many_cap: int = 0,
    fixed_spec: ImmunogenFixedSpec | None = None,
    tcell_motifs: Iterable[str] | None = None,
    cleavage_motifs: Iterable[str] | None = None,
    alternating_list1: Iterable[str] | None = None,
    alternating_list2: Iterable[str] | None = None,
    expected_start_group: str = "list1",
    seed_sequence: str | None = None,
    seed_pdb: Path | None = None,
) -> list[dict[str, object]]:
    """Analyze all sequence records in one completed DISCO dump directory."""

    sequence_records = sequence_records_in_dump(dump_dir)
    if not sequence_records:
        raise ValueError(f"No DISCO sequence files found in {dump_dir / 'sequences'}")

    confidence_records = confidence_records_in_dump(dump_dir)
    structure_records = structure_records_in_dump(dump_dir)
    diagnostic_metadata = diagnostic_metadata_in_dump(dump_dir)
    provenance_metadata = provenance_metadata_in_dump(dump_dir)
    seed_pdb_ca = ca_coordinates_from_pdb(seed_pdb) if seed_pdb is not None else None
    rows = []
    for sequence in sequence_records:
        confidence, status = match_confidence_record(sequence, confidence_records)
        structure, structure_status = match_structure_record(sequence, structure_records)
        rows.append(
            disco_design_row(
                sequence=sequence,
                confidence=confidence,
                confidence_match_status=status,
                motifs=motifs,
                motif_all_weight=motif_all_weight,
                motif_many_weight=motif_many_weight,
                motif_many_cap=motif_many_cap,
                fixed_spec=fixed_spec,
                tcell_motifs=tcell_motifs,
                cleavage_motifs=cleavage_motifs,
                alternating_list1=alternating_list1,
                alternating_list2=alternating_list2,
                expected_start_group=expected_start_group,
                diagnostic_metadata=diagnostic_metadata.get(sequence.base_id),
                provenance_metadata=provenance_metadata.get(sequence.base_id),
                structure=structure,
                structure_match_status=structure_status,
                seed_sequence=seed_sequence,
                seed_pdb_ca=seed_pdb_ca,
            )
        )
    return rows
