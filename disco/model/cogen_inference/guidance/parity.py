"""ProteinGenerator-to-DISCO motif-guidance parity audit.

This module is deliberately static plus evidence-based: the feature list records
what ProteinGenerator capability we tried to preserve, while the audit checks
whether the corresponding DISCO files and named tests are present in the current
checkout.  It is not a scientific result validator; it is a completion map for
the implementation layer before real PG-vs-DISCO runs are interpreted.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS_ORDER = ("ported", "partial", "not_ported", "not_applicable")
STATUSES_REQUIRING_DISCO_EVIDENCE = {"ported", "partial"}


@dataclass(frozen=True)
class ParityItem:
    """One ProteinGenerator feature and its current DISCO transfer status."""

    feature_id: str
    category: str
    protein_generator_handles: tuple[str, ...]
    disco_implementation: str
    status: str
    evidence_paths: tuple[str, ...]
    test_names: tuple[str, ...]
    notes: str


def default_parity_items() -> list[ParityItem]:
    """Return the PG feature map that should stay in sync with this port.

    `status` describes the conceptual implementation state.  The audit then
    separately records whether the listed files and tests exist in the checkout.
    """

    return [
        ParityItem(
            feature_id="generic_hard_motif_scoring",
            category="standard_motif",
            protein_generator_handles=("HardMotifScorer",),
            disco_implementation="guidance/scoring.py::HardMotifScorer",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/scoring.py",),
            test_names=("test_hard_motif_scorer_counts_overlaps",),
            notes="Hard sequence-level audit of exact motif occurrences.",
        ),
        ParityItem(
            feature_id="sequence_motifs_logit_guidance",
            category="standard_motif",
            protein_generator_handles=("motifs_bias", "SequenceMotifsBias"),
            disco_implementation="guidance/motif.py::MotifLogitGuidance",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/motif.py",),
            test_names=("test_gradient_guidance_increases_requested_motif_logits",),
            notes="DISCO analogue edits decoder logits before path-planning sampling.",
        ),
        ParityItem(
            feature_id="local_rescue",
            category="standard_motif",
            protein_generator_handles=("MotifLocalSearchRescue", "motif_local_search"),
            disco_implementation="guidance/motif.py local rescue mode",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/motif.py",),
            test_names=("test_local_rescue_respects_fixed_positions",),
            notes="Discrete rescue-style logit boost for missing motif windows.",
        ),
        ParityItem(
            feature_id="save_best_combined",
            category="selection",
            protein_generator_handles=("save_best_combined", "adjust_combined_score"),
            disco_implementation="guidance/controller.py + cogen_inference_loop_strategies.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/controller.py",
                "disco/model/cogen_inference/cogen_inference_loop_strategies.py",
            ),
            test_names=("test_controller_tracks_best_step",),
            notes="Tracks best observed combined score and can select that state at finalization.",
        ),
        ParityItem(
            feature_id="per_sample_batch_save_best",
            category="selection",
            protein_generator_handles=("save_best_combined",),
            disco_implementation="guidance/controller.py::select_best_outputs",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/controller.py",),
            test_names=("test_controller_tracks_and_selects_best_step_per_batch_sample",),
            notes="DISCO-specific extension so each batch sample can keep its own best state.",
        ),
        ParityItem(
            feature_id="alternating_hard_metrics",
            category="alternating",
            protein_generator_handles=("combined_alternating_score", "alternating_save"),
            disco_implementation="guidance/alternating.py scoring functions",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/alternating.py",),
            test_names=(
                "test_alternating_score_prefers_switching_groups",
                "test_alternating_metrics_report_pattern",
            ),
            notes="Hard audit metrics for list1/list2 order, coverage, and penalties.",
        ),
        ParityItem(
            feature_id="alternating_slot_guidance",
            category="alternating",
            protein_generator_handles=("alternating_motifs_bias", "AlternatingMotifsBias"),
            disco_implementation="guidance/alternating.py::AlternatingLogitGuidance",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/alternating.py",),
            test_names=(
                "test_alternating_logit_guidance_increases_slot_objective",
                "test_alternating_logit_guidance_respects_fixed_positions",
            ),
            notes="Slot-based alternating motif logit objective.",
        ),
        ParityItem(
            feature_id="alternating_transition_guidance",
            category="alternating",
            protein_generator_handles=(
                "soft_transition_alternating_motifs_bias",
                "SoftTransitionAlternatingMotifsBias",
            ),
            disco_implementation="guidance/alternating.py::SoftTransitionAlternatingLogitGuidance",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/alternating.py",),
            test_names=(
                "test_soft_transition_logit_guidance_prefers_alternating_order",
                "test_controller_can_apply_transition_only_alternating_guidance",
            ),
            notes="Slot-free transition objective that encourages list switching.",
        ),
        ParityItem(
            feature_id="multi_sequence_motifs_bias",
            category="alternating",
            protein_generator_handles=("multi_motifs_bias", "MultiSequenceMotifsBias"),
            disco_implementation="guidance/alternating.py hard/slot/transition modes",
            status="partial",
            evidence_paths=("disco/model/cogen_inference/guidance/alternating.py",),
            test_names=("test_alternating_score_prefers_switching_groups",),
            notes="The useful two-list behavior is covered; the old hardcoded PG potential is not copied verbatim.",
        ),
        ParityItem(
            feature_id="immunogen_fixed_fragments",
            category="immunogen",
            protein_generator_handles=("fixed_fragments_json", "score_fixed_fragments"),
            disco_implementation="guidance/immunogen.py fixed-fragment masks and scoring",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/immunogen.py",
                "disco/model/cogen_inference/guidance/controller.py",
            ),
            test_names=(
                "test_immunogen_fixed_mask_and_scoring",
                "test_controller_blocks_immunogen_fixed_positions_in_change_mask",
            ),
            notes="Fixed B-cell/immunogen fragments are protected through DISCO change masks.",
        ),
        ParityItem(
            feature_id="immunogen_mutable_window_guidance",
            category="immunogen",
            protein_generator_handles=(
                "immunogen_mutable_motifs_bias",
                "MutableWindowSequenceMotifsBias",
            ),
            disco_implementation="guidance/immunogen.py::MutableWindowMotifLogitGuidance",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/immunogen.py",),
            test_names=(
                "test_immunogen_mutable_window_guidance_ignores_fixed_overlap",
                "test_immunogen_mutable_window_guidance_uses_mutable_windows",
            ),
            notes="Motif logit guidance is restricted to windows that do not overlap fixed fragments.",
        ),
        ParityItem(
            feature_id="immunogen_spacing_score",
            category="immunogen",
            protein_generator_handles=("immunogen_spacing_aware", "spacing_score"),
            disco_implementation="guidance/immunogen.py::spacing_score",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/immunogen.py",),
            test_names=("test_immunogen_spacing_score_prefers_spread_motifs",),
            notes="Completed-sequence score rewards spread motif placement.",
        ),
        ParityItem(
            feature_id="immunogen_category_validation",
            category="immunogen",
            protein_generator_handles=(
                "validate_immunogen_summary.py",
                "B-cell",
                "T-cell",
                "cleavage",
            ),
            disco_implementation="guidance/immunogen_validation.py + runner/guidance_validate_immunogen.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/immunogen_validation.py",
                "runner/guidance_validate_immunogen.py",
            ),
            test_names=(
                "test_immunogen_validation_accepts_category_complete_row",
                "test_immunogen_validation_reports_category_failures",
            ),
            notes="Separate completed-design gates for fixed/B-cell, T-cell, and cleavage motif categories.",
        ),
        ParityItem(
            feature_id="immunogen_second_pass_reprediction_plan",
            category="immunogen",
            protein_generator_handles=("immunogen_repredict", "run_immunogen_workflow.py"),
            disco_implementation="guidance/reprediction.py + runner/guidance_plan_reprediction.py",
            status="partial",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/reprediction.py",
                "runner/guidance_plan_reprediction.py",
            ),
            test_names=(
                "test_reprediction_plan_writes_fixed_sequence_input_and_commands",
                "test_reprediction_plan_filters_and_ranks_designs",
            ),
            notes="Writes fixed-sequence DISCO second-pass input JSONs and commands; execution remains an explicit operator step.",
        ),
        ParityItem(
            feature_id="evolutionary_outer_loop",
            category="evolution",
            protein_generator_handles=("run_codeocean_evolution_original_motifs.py",),
            disco_implementation="guidance/evolution.py + runner/guidance_evolution.py",
            status="partial",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/evolution.py",
                "runner/guidance_evolution.py",
            ),
            test_names=(
                "test_evolution_command_contains_guidance_overrides",
                "test_evolution_next_generation_preserves_population_size",
            ),
            notes="Genome, mutation, crossover, templates, command building, and fitness are present; full repeated DISCO execution remains an external run workflow.",
        ),
        ParityItem(
            feature_id="evolution_template_materialization",
            category="evolution",
            protein_generator_handles=("template", "elite"),
            disco_implementation="guidance/evolution.py template materialization helpers",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/evolution.py",),
            test_names=(
                "test_evolution_materializes_template_input_json",
                "test_evolution_apply_template_filters_matching_lengths_and_preserves_context",
            ),
            notes="Elite/template sequences can be converted into DISCO input JSON records.",
        ),
        ParityItem(
            feature_id="per_step_diagnostics",
            category="logging",
            protein_generator_handles=("save_args", "trace_timesteps"),
            disco_implementation="guidance/diagnostics.py + runner/inference.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/diagnostics.py",
                "runner/inference.py",
            ),
            test_names=(
                "test_diagnostics_summary_selects_best_guidance_step",
                "test_write_diagnostic_outputs_serializes_tensor_values",
            ),
            notes="Writes per-step JSONL, CSV, and run metadata for guided DISCO sampling.",
        ),
        ParityItem(
            feature_id="trajectory_analysis",
            category="logging",
            protein_generator_handles=("trace_timesteps", "trace_timestep_pdbs"),
            disco_implementation="guidance/trajectory.py + runner/guidance_analyze_trajectory.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/trajectory.py",
                "runner/guidance_analyze_trajectory.py",
            ),
            test_names=(
                "test_trajectory_rows_track_deltas_and_best_step",
                "test_write_trajectory_outputs_writes_csv_and_summary",
            ),
            notes="Offline sequence/score/noise trajectory analysis from guidance diagnostics.",
        ),
        ParityItem(
            feature_id="coordinate_trajectory_dump",
            category="logging",
            protein_generator_handles=("trace_timestep_pdbs",),
            disco_implementation="guidance/trajectory_dump.py + runner/inference.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/trajectory_dump.py",
                "runner/inference.py",
            ),
            test_names=("test_write_coordinate_trajectory_outputs_saves_tensors_and_manifest",),
            notes="Optional disabled-by-default tensor dumps for denoised and trajectory coordinates.",
        ),
        ParityItem(
            feature_id="disco_completed_dump_analysis",
            category="evaluation",
            protein_generator_handles=("analyze_immunogen_outputs.py", "summary.csv"),
            disco_implementation="guidance/disco_analysis.py + runner/guidance_analyze_disco.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/disco_analysis.py",
                "runner/guidance_analyze_disco.py",
            ),
            test_names=(
                "test_disco_analysis_parses_multiple_sequence_entries",
                "test_disco_analysis_adds_quality_and_structure_metrics",
            ),
            notes="Normalizes completed DISCO dump outputs into motif/confidence/quality CSV rows.",
        ),
        ParityItem(
            feature_id="same_artifact_evaluation_provenance",
            category="evaluation",
            protein_generator_handles=("save_best_combined", "pred_lddt"),
            disco_implementation="guidance/evaluation_provenance.py",
            status="ported",
            evidence_paths=("disco/model/cogen_inference/guidance/evaluation_provenance.py",),
            test_names=("test_comparison_normalizes_disco_same_artifact_provenance",),
            notes="Tracks whether sequence, confidence, structure, and diagnostics come from the same sample artifact.",
        ),
        ParityItem(
            feature_id="completed_quality_gates",
            category="evaluation",
            protein_generator_handles=("validate_immunogen_summary.py",),
            disco_implementation="guidance/quality_validation.py + runner/guidance_validate_quality.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/quality_validation.py",
                "runner/guidance_validate_quality.py",
            ),
            test_names=(
                "test_quality_validation_accepts_clean_disco_analysis_row",
                "test_quality_validation_rejects_mixed_evaluation_artifacts",
            ),
            notes="Configurable gates for confidence, motif coverage, sequence quality, RMSD, provenance, and diagnostics.",
        ),
        ParityItem(
            feature_id="pg_vs_disco_experiment_planning",
            category="comparison",
            protein_generator_handles=("run_immunogen_workflow.py", "run_alternating_motif_workflow.py"),
            disco_implementation="guidance/experiment_plan.py + runner/guidance_plan_comparison.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/experiment_plan.py",
                "runner/guidance_plan_comparison.py",
            ),
            test_names=(
                "test_experiment_plan_materializes_core_matrix",
                "test_experiment_plan_marks_missing_pg_immunogen_inputs",
            ),
            notes="Creates Standard, Immunogen, and Alternating command matrices without launching heavy model runs.",
        ),
        ParityItem(
            feature_id="pg_vs_disco_fairness_audit",
            category="comparison",
            protein_generator_handles=("n_cycle", "--T"),
            disco_implementation="guidance/fairness.py + runner/guidance_audit_fairness.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/fairness.py",
                "runner/guidance_audit_fairness.py",
            ),
            test_names=(
                "test_fairness_audit_accepts_balanced_standard_plan",
                "test_fairness_audit_rejects_unmatched_group_parameters",
            ),
            notes="Audits planned rows for matched systems, lengths, motifs, designs, seeds, N-cycle, and diffusion-budget caveats.",
        ),
        ParityItem(
            feature_id="comparison_validation",
            category="comparison",
            protein_generator_handles=("compare_immunogen_runs.py",),
            disco_implementation="guidance/comparison_validation.py + runner/guidance_validate_comparison.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/comparison_validation.py",
                "runner/guidance_validate_comparison.py",
            ),
            test_names=(
                "test_comparison_validation_accepts_completed_disco_summary_with_diagnostics",
                "test_comparison_validation_requires_both_systems_by_default",
            ),
            notes="Validates planned and completed PG-vs-DISCO matrices before comparison reduction.",
        ),
        ParityItem(
            feature_id="run_reproducibility_bundle",
            category="logging",
            protein_generator_handles=("save_args", "run metadata"),
            disco_implementation="guidance/run_bundle.py + runner/guidance_bundle_run.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/run_bundle.py",
                "runner/guidance_bundle_run.py",
            ),
            test_names=(
                "test_run_bundle_collects_summary_diagnostics_and_hashes",
                "test_run_bundle_validation_reports_missing_required_artifacts",
            ),
            notes="Post-run manifest, command, summary, diagnostics, trajectory, and hash bundle.",
        ),
        ParityItem(
            feature_id="comparison_reduction_and_report",
            category="comparison",
            protein_generator_handles=("compare_immunogen_runs.py", "EXPERIMENTS.md"),
            disco_implementation="guidance/comparison.py + guidance/comparison_report.py",
            status="ported",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/comparison.py",
                "disco/model/cogen_inference/guidance/comparison_report.py",
                "runner/guidance_compare.py",
                "runner/guidance_report_comparison.py",
            ),
            test_names=(
                "test_comparison_normalizes_pg_and_disco_rows",
                "test_comparison_report_renders_validation_and_top_designs",
            ),
            notes="Normalizes PG/DISCO CSVs, computes transparent comparison scores, and renders Markdown reports.",
        ),
        ParityItem(
            feature_id="best_plddt_equivalent",
            category="evaluation",
            protein_generator_handles=("save_best_plddt", "pred_lddt", "pg_plddt_mean"),
            disco_implementation="guidance/disco_analysis.py confidence extraction and matching",
            status="partial",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/disco_analysis.py",
                "disco/model/cogen_inference/guidance/evaluation_provenance.py",
            ),
            test_names=(
                "test_disco_analysis_pairs_sequence_with_matching_confidence",
                "test_quality_validation_rejects_mixed_evaluation_artifacts",
            ),
            notes="DISCO confidence rows can be matched and gated, but this is not identical to PG in-loop pred_lddt save-best behavior.",
        ),
        ParityItem(
            feature_id="structural_seed_contig_scaffolding",
            category="structure",
            protein_generator_handles=("--pdb", "--contigs", "--inpaint_seq", "--inpaint_str"),
            disco_implementation="guidance/scaffold_plan.py + runner/guidance_plan_scaffold.py",
            status="partial",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/scaffold_plan.py",
                "runner/guidance_plan_scaffold.py",
            ),
            test_names=(
                "test_scaffold_plan_writes_fixed_fragment_input_json",
                "test_scaffold_plan_reports_structural_contig_unsupported",
            ),
            notes=(
                "Sequence-level fixed fragments/ranges are materialized as DISCO "
                "proteinChain strings; PG/RFdiffusion-style PDB coordinate contigs "
                "and independent structure masks remain unsupported."
            ),
        ),
        ParityItem(
            feature_id="cleavage_foldswitch_sampler",
            category="cleavage",
            protein_generator_handles=("cleavage_foldswitch_SAMPLER", "cleavage_foldswitch"),
            disco_implementation="not implemented",
            status="not_ported",
            evidence_paths=(),
            test_names=(),
            notes="The Immunogen cleavage motif validation exists; the multi-state cleavage fold-switch sampler itself was not transferred.",
        ),
        ParityItem(
            feature_id="ligand_aware_guidance",
            category="ligand",
            protein_generator_handles=(),
            disco_implementation="not implemented",
            status="not_ported",
            evidence_paths=(),
            test_names=(),
            notes="No custom ligand-objective scorer/guidance has been added to the DISCO motif port.",
        ),
        ParityItem(
            feature_id="external_reprediction_and_rosetta",
            category="evaluation",
            protein_generator_handles=("ESM-2", "RMSD", "Rosetta", "repredict"),
            disco_implementation="guidance/external_quality.py + runner/guidance_merge_external_quality.py",
            status="partial",
            evidence_paths=(
                "disco/model/cogen_inference/guidance/external_quality.py",
                "runner/guidance_merge_external_quality.py",
            ),
            test_names=(
                "test_external_quality_score_uses_pg_rank_style_components",
                "test_external_quality_outputs_merge_metrics_and_report_missing_values",
            ),
            notes="Imports externally produced ESM/RMSD/Rosetta metrics and computes PG-style posthoc rank scores; it does not run external binaries.",
        ),
        ParityItem(
            feature_id="charge_bias",
            category="non_motif",
            protein_generator_handles=("ChargeBias",),
            disco_implementation="not applicable to requested motif comparison port",
            status="not_applicable",
            evidence_paths=(),
            test_names=(),
            notes="Plain PG had non-motif potentials; this DISCO port is scoped to motif/immunogen/alternating comparison functionality.",
        ),
    ]


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _candidate_pg_paths(pg_root: Path) -> list[Path]:
    fixed_paths = [
        pg_root / "AI_README.md",
        pg_root / "code" / "run_codeocean_evolution_original_motifs.py",
        pg_root / "code" / "protein_generator" / "utils" / "potentials.py",
        pg_root / "code" / "protein_generator" / "utils" / "sampler.py",
    ]
    scan_dirs = [
        pg_root / "code" / "protein_generator" / "immunogen",
        pg_root / "code" / "protein_generator" / "alternating_motifs",
        pg_root / "code" / "protein_generator" / "tests",
        pg_root / "code" / "tests",
    ]
    paths = [path for path in fixed_paths if path.exists()]
    for directory in scan_dirs:
        if directory.exists():
            paths.extend(
                path
                for path in directory.rglob("*")
                if path.suffix in {".py", ".md", ".json", ".csv"}
            )
    return sorted(set(paths))


def _build_pg_text_index(pg_root: Path | None) -> str:
    if pg_root is None or not pg_root.exists():
        return ""
    return "\n".join(_safe_read_text(path) for path in _candidate_pg_paths(pg_root))


def test_name_exists(test_file: Path, test_name: str) -> bool:
    """Return whether a named test function exists in the DISCO test file."""

    text = _safe_read_text(test_file)
    return f"def {test_name}(" in text


def _item_to_record(
    item: ParityItem,
    *,
    disco_root: Path,
    test_file: Path,
    test_text: str,
    pg_text_index: str,
    pg_checked: bool,
) -> dict[str, Any]:
    if item.status not in STATUS_ORDER:
        raise ValueError(f"Unknown parity status for {item.feature_id}: {item.status}")

    evidence = [
        {
            "path": path,
            "exists": (disco_root / path).exists(),
        }
        for path in item.evidence_paths
    ]
    tests = [
        {
            "name": name,
            "exists": f"def {name}(" in test_text,
        }
        for name in item.test_names
    ]
    pg_evidence = [
        {
            "handle": handle,
            "found": bool(pg_text_index and handle in pg_text_index),
        }
        for handle in item.protein_generator_handles
    ]

    needs_evidence = item.status in STATUSES_REQUIRING_DISCO_EVIDENCE
    disco_evidence_ok = (
        all(entry["exists"] for entry in evidence)
        if needs_evidence
        else len(evidence) == 0 or all(entry["exists"] for entry in evidence)
    )
    test_evidence_ok = all(entry["exists"] for entry in tests)
    pg_evidence_ok = (
        all(entry["found"] for entry in pg_evidence)
        if pg_checked and pg_evidence
        else None
    )

    if needs_evidence and (not disco_evidence_ok or not test_evidence_ok):
        audit_state = "evidence_missing"
    elif item.status == "ported":
        audit_state = "ported_with_evidence"
    elif item.status == "partial":
        audit_state = "partial_with_evidence"
    else:
        audit_state = item.status

    return {
        "feature_id": item.feature_id,
        "category": item.category,
        "protein_generator_handles": list(item.protein_generator_handles),
        "disco_implementation": item.disco_implementation,
        "status": item.status,
        "audit_state": audit_state,
        "disco_evidence": evidence,
        "disco_evidence_ok": disco_evidence_ok,
        "test_evidence": tests,
        "test_evidence_ok": test_evidence_ok,
        "pg_evidence": pg_evidence,
        "pg_evidence_ok": pg_evidence_ok,
        "notes": item.notes,
        "test_file": str(test_file),
    }


def audit_parity(
    disco_root: Path,
    pg_root: Path | None = None,
    items: list[ParityItem] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit the current checkout against the static PG-to-DISCO feature map."""

    disco_root = disco_root.resolve()
    pg_root = pg_root.resolve() if pg_root is not None else None
    test_file = disco_root / "tests" / "test_guidance.py"
    test_text = _safe_read_text(test_file)
    pg_checked = pg_root is not None and pg_root.exists()
    pg_text_index = _build_pg_text_index(pg_root)

    records = [
        _item_to_record(
            item,
            disco_root=disco_root,
            test_file=test_file,
            test_text=test_text,
            pg_text_index=pg_text_index,
            pg_checked=pg_checked,
        )
        for item in (items if items is not None else default_parity_items())
    ]

    status_counts = Counter(record["status"] for record in records)
    category_counts = Counter(record["category"] for record in records)
    missing_disco_evidence = [
        record["feature_id"]
        for record in records
        if record["status"] in STATUSES_REQUIRING_DISCO_EVIDENCE
        and not record["disco_evidence_ok"]
    ]
    missing_test_evidence = [
        record["feature_id"]
        for record in records
        if record["status"] in STATUSES_REQUIRING_DISCO_EVIDENCE
        and not record["test_evidence_ok"]
    ]
    missing_pg_handles = [
        record["feature_id"]
        for record in records
        if pg_checked and record["pg_evidence_ok"] is False
    ]

    summary = {
        "status": (
            "pass"
            if not missing_disco_evidence and not missing_test_evidence
            else "warning"
        ),
        "num_features": len(records),
        "status_counts": {status: status_counts.get(status, 0) for status in STATUS_ORDER},
        "category_counts": dict(sorted(category_counts.items())),
        "ported_count": status_counts.get("ported", 0),
        "partial_count": status_counts.get("partial", 0),
        "not_ported_count": status_counts.get("not_ported", 0),
        "not_applicable_count": status_counts.get("not_applicable", 0),
        "missing_disco_evidence_count": len(missing_disco_evidence),
        "missing_test_evidence_count": len(missing_test_evidence),
        "missing_pg_handle_count": len(missing_pg_handles),
        "missing_disco_evidence_features": missing_disco_evidence,
        "missing_test_evidence_features": missing_test_evidence,
        "missing_pg_handle_features": missing_pg_handles,
        "disco_root": str(disco_root),
        "pg_root": str(pg_root) if pg_root is not None else "",
        "pg_root_checked": pg_checked,
    }
    return records, summary


def _compact_evidence_state(record: dict[str, Any]) -> str:
    missing_paths = [
        entry["path"] for entry in record["disco_evidence"] if not entry["exists"]
    ]
    missing_tests = [
        entry["name"] for entry in record["test_evidence"] if not entry["exists"]
    ]
    parts: list[str] = []
    parts.append("files ok" if not missing_paths else "missing files: " + ", ".join(missing_paths))
    parts.append("tests ok" if not missing_tests else "missing tests: " + ", ".join(missing_tests))
    return "; ".join(parts)


def render_parity_markdown(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render a human-readable parity report."""

    lines = [
        "# DISCO ProteinGenerator Parity Audit",
        "",
        "## Summary",
        "",
        f"- Status: {summary['status']}",
        f"- Features: {summary['num_features']}",
        f"- Ported: {summary['ported_count']}",
        f"- Partial: {summary['partial_count']}",
        f"- Not ported: {summary['not_ported_count']}",
        f"- Not applicable: {summary['not_applicable_count']}",
        f"- Missing DISCO file evidence: {summary['missing_disco_evidence_count']}",
        f"- Missing DISCO test evidence: {summary['missing_test_evidence_count']}",
        f"- PG root checked: {summary['pg_root_checked']}",
        "",
        "## Feature Map",
        "",
        "| Feature | Category | Status | DISCO implementation | Evidence | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        handles = ", ".join(record["protein_generator_handles"])
        notes = record["notes"].replace("|", "/")
        implementation = record["disco_implementation"].replace("|", "/")
        evidence = _compact_evidence_state(record).replace("|", "/")
        lines.append(
            "| {feature} | {category} | {status} | {impl} | {evidence} | {notes} PG: {handles} |".format(
                feature=record["feature_id"],
                category=record["category"],
                status=record["status"],
                impl=implementation,
                evidence=evidence,
                notes=notes,
                handles=handles,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_parity_report(
    report_json_path: Path,
    report_md_path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write JSON and Markdown parity audit outputs."""

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "features": records,
    }
    report_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_md_path.write_text(render_parity_markdown(records, summary))
