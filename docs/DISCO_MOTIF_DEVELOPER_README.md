# DISCO Motif Guidance Developer README

## What changed

DISCO now has an optional Cambrium guidance layer for motif experiments. It is
off by default.

Enable it with Hydra overrides:

```bash
cogen_guidance.enabled=true
'cogen_guidance.motifs=[GHK,KTTKS]'
cogen_guidance.logit_guidance.enabled=true
```

## Where to look

```text
configs/cogen_guidance/default.yaml
disco/model/cogen_inference/guidance/controller.py
disco/model/cogen_inference/guidance/motif.py
disco/model/cogen_inference/guidance/alternating.py
disco/model/cogen_inference/guidance/immunogen.py
disco/model/cogen_inference/guidance/evolution.py
disco/model/cogen_inference/guidance/external_quality.py
disco/model/cogen_inference/guidance/disco_analysis.py
disco/model/cogen_inference/guidance/diagnostics.py
disco/model/cogen_inference/guidance/evaluation_provenance.py
disco/model/cogen_inference/guidance/artifact_provenance.py
disco/model/cogen_inference/guidance/trajectory.py
disco/model/cogen_inference/guidance/trajectory_dump.py
disco/model/cogen_inference/guidance/experiment_plan.py
disco/model/cogen_inference/guidance/fairness.py
disco/model/cogen_inference/guidance/comparison.py
disco/model/cogen_inference/guidance/comparison_report.py
disco/model/cogen_inference/guidance/comparison_validation.py
disco/model/cogen_inference/guidance/immunogen_validation.py
disco/model/cogen_inference/guidance/quality_validation.py
disco/model/cogen_inference/guidance/parity.py
disco/model/cogen_inference/guidance/reprediction.py
disco/model/cogen_inference/guidance/run_bundle.py
disco/model/cogen_inference/guidance/scaffold_plan.py
disco/model/cogen_inference/guidance/scoring.py
runner/guidance_evolution.py
runner/guidance_merge_external_quality.py
runner/guidance_audit_fairness.py
runner/guidance_audit_parity.py
runner/guidance_analyze_disco.py
runner/guidance_analyze_trajectory.py
runner/guidance_bundle_run.py
runner/guidance_plan_comparison.py
runner/guidance_plan_reprediction.py
runner/guidance_plan_scaffold.py
runner/guidance_validate_comparison.py
runner/guidance_validate_immunogen.py
runner/guidance_validate_quality.py
runner/guidance_compare.py
runner/guidance_report_comparison.py
tests/test_guidance.py
```

The main DISCO loop only has a small hook:

```text
seq logits -> GuidanceController.modify_logits() -> existing sampler
step state -> GuidanceController.observe_step()
```

## Design rule

Do not put motif-specific logic into `cogen_inference_loop_body_impl.py`.
That file should only call the controller. Add new motif/immunogen/alternating
behavior in separate modules.

## Implemented now

- hard exact motif scoring,
- gradient-style logit guidance,
- local top-k rescue,
- alternating hard scoring and optional combined-score contribution,
- slot-based alternating logit guidance adapted from ProteinGenerator
  `AlternatingMotifsBias`,
- slot-free transition-only alternating logit guidance adapted from
  `soft_transition_alternating_motifs_bias`,
- immunogen fixed-fragment spec/mask/scoring,
- immunogen fixed-position sampler-mask blocking,
- immunogen mutable-window logit guidance adapted from
  `immunogen_mutable_motifs_bias`,
- immunogen spacing-aware score,
- DISCO evolutionary outer-loop runner adapted from ProteinGenerator Approach 3,
- evolutionary elite-template materialization into DISCO input JSONs,
- per-step guidance diagnostics exported as JSONL/CSV/runtime metadata,
- offline trajectory CSV/JSON summaries from diagnostic JSONL files,
- optional full coordinate trajectory tensor dumping as `.pt` files plus JSON
  manifests,
- final artifact provenance manifests and intermediate-state CSV indexes that
  keep final outputs separate from per-timestep saved tensors,
- post-run reproducibility bundles with run manifests, summary statistics,
  diagnostics metadata, artifact hashes, and validation issues,
- completed DISCO dump analyzer with sample-aware sequence/confidence/structure
  pairing plus homopolymer, entropy, seed identity, and optional PDB CA-RMSD,
- evaluation-bundle provenance fields proving whether DISCO sequence,
  confidence, structure, and diagnostics come from the same output sample,
- PG-vs-DISCO experiment planner for Standard, Immunogen, and Alternating,
- PG-vs-DISCO fairness audit for matched systems, replicates, seeds, N-cycle,
  lengths, motifs, designs per replicate, and explicit DISCO/PG budget knobs,
- PG-vs-DISCO plan/result validator for required systems, balanced matrices,
  artifacts, summaries, confidence provenance, diagnostics, and Immunogen fixed
  fragments,
- completed-design quality validator for confidence, motif coverage, fixed
  fragments, low-complexity guards, optional RMSD/spread, diagnostics,
  structure presence, exact confidence provenance, and same-artifact evaluation,
- external ESM/RMSD/Rosetta metric import plus PG-style posthoc quality scoring,
- Immunogen category validator for B-cell/fixed-fragment preservation, T-cell
  motif coverage, cleavage motif coverage, and same-artifact provenance,
- Immunogen second-pass reprediction planning from completed analysis CSV rows
  into full-sequence DISCO input JSONs, inference commands, and analyzer
  commands,
- sequence-level scaffold input planning from fixed fragments/ranges into DISCO
  `proteinChain.sequence`, with unsupported PG structure-contig semantics written
  to a manifest,
- PG-vs-DISCO CSV normalization, aggregate comparison, score-provenance fields,
  robust ProteinGenerator Alternating column handling, and Markdown report
  generation,
- PG-to-DISCO implementation parity audit with explicit ported, partial,
  not-ported, and not-applicable feature records,
- optional per-sample save-best tracking and best-output selection for
  batch/duplicate inference,
- Hydra config group,
- CPU-only unit checks.

## Not implemented yet

- automatic execution of DISCO second-pass reprediction; current support writes
  fixed-sequence inputs and commands but does not launch the heavy model run,
- full-scale ProteinGenerator Docker runtime on this Mac; Docker/Colima was not
  available, so the local Immunogen smoke uses the Conda direct-PG path,
- richer T-cell/cleavage motif presets and workflow metadata,
- ligand-aware quality metrics and automatic external-tool execution,
- true PG-style structural seed/contig scaffolding and cleavage fold-switch
  sampling,
- executed full-scale PG-vs-DISCO scientific study report.

## Artifact and score provenance

The local comparison pipeline enforces one rule: a score is only comparable
when it was computed on the same saved protein artifact that the row summarizes.

ProteinGenerator summaries satisfy this in the local smokes by reading both the
sequence and `pg_plddt_mean` from the saved PDB. DISCO currently does not emit a
native pLDDT/summary-confidence file in the exercised inference path, so DISCO
rows are marked `confidence_match_status=missing` and
`score_provenance_status=disco_unmatched_confidence`. Use regular validation
with `--no-require-confidence` for smoke testing. Use strict validation without
that flag when a result is intended to be confidence-comparable.

For DISCO trajectory storage, final and intermediate artifacts are deliberately
separate. Each final sample gets:

```text
disco_output/provenance/*_artifact_provenance.json
disco_output/provenance/*_intermediate_states.csv
```

The JSON manifest binds final sequence, final structure, diagnostics, trajectory
manifest, hashes, and any same-sample confidence files that exist. The CSV has
one row per saved structure-state kind per timestep, for example `x1_denoised`
and `xt_trajectory`. Those rows are marked
`structure_confidence_status=not_computed_for_intermediate_state`; do not reuse
final pLDDT or final confidence for them.

## Quick local check

The repo environment currently lacks `pytest`. Use this until pytest is added:

```bash
.venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('tests/test_guidance.py')
spec = importlib.util.spec_from_file_location('test_guidance', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in [
    'test_hard_motif_scorer_counts_overlaps',
    'test_gradient_guidance_increases_requested_motif_logits',
    'test_local_rescue_respects_fixed_positions',
    'test_controller_tracks_best_step',
    'test_controller_tracks_and_selects_best_step_per_batch_sample',
    'test_alternating_score_prefers_switching_groups',
    'test_alternating_metrics_report_pattern',
    'test_alternating_logit_guidance_increases_slot_objective',
    'test_alternating_logit_guidance_respects_fixed_positions',
    'test_alternating_logit_guidance_noops_when_motifs_do_not_fit',
    'test_controller_can_apply_alternating_without_plain_motif_guidance',
    'test_soft_transition_logit_guidance_prefers_alternating_order',
    'test_soft_transition_logit_guidance_updates_only_changeable_positions',
    'test_controller_can_apply_transition_only_alternating_guidance',
    'test_immunogen_fixed_mask_and_scoring',
    'test_immunogen_fixed_position_mask_does_not_need_seed_sequence',
    'test_immunogen_mutable_window_guidance_ignores_fixed_overlap',
    'test_immunogen_mutable_window_guidance_uses_mutable_windows',
    'test_controller_blocks_immunogen_fixed_positions_in_change_mask',
    'test_controller_can_apply_immunogen_mutable_guidance',
    'test_immunogen_spacing_score_prefers_spread_motifs',
    'test_evolution_command_contains_guidance_overrides',
    'test_evolution_mutation_respects_fixed_n_cycle',
    'test_evolution_next_generation_preserves_population_size',
    'test_evolution_template_to_disco_sequence_converts_x_to_mask',
    'test_evolution_apply_template_filters_matching_lengths_and_preserves_context',
    'test_evolution_materializes_template_input_json',
    'test_evolution_fitness_modes_and_sequence_parser',
    'test_disco_analysis_parses_multiple_sequence_entries',
    'test_disco_analysis_pairs_sequence_with_matching_confidence',
    'test_disco_analysis_adds_quality_and_structure_metrics',
    'test_diagnostics_summary_selects_best_guidance_step',
    'test_write_diagnostic_outputs_serializes_tensor_values',
    'test_trajectory_rows_track_deltas_and_best_step',
    'test_write_trajectory_outputs_writes_csv_and_summary',
    'test_write_coordinate_trajectory_outputs_saves_tensors_and_manifest',
    'test_run_bundle_collects_summary_diagnostics_and_hashes',
    'test_run_bundle_validation_reports_missing_required_artifacts',
    'test_reprediction_plan_writes_fixed_sequence_input_and_commands',
    'test_reprediction_plan_filters_and_ranks_designs',
    'test_scaffold_plan_writes_fixed_fragment_input_json',
    'test_scaffold_plan_fixed_range_uses_seed_sequence',
    'test_scaffold_plan_reports_structural_contig_unsupported',
    'test_parity_audit_reports_core_ported_and_open_features',
    'test_write_parity_report_outputs_json_and_markdown',
    'test_experiment_plan_writes_fixed_fragment_sequence',
    'test_experiment_plan_materializes_core_matrix',
    'test_experiment_plan_materializes_transition_only_alternating_variant',
    'test_experiment_plan_marks_missing_pg_immunogen_inputs',
    'test_experiment_plan_marks_missing_disco_immunogen_fragments',
    'test_fairness_audit_accepts_balanced_standard_plan',
    'test_fairness_audit_rejects_unmatched_group_parameters',
    'test_comparison_normalizes_pg_and_disco_rows',
    'test_comparison_normalizes_pg_alternating_columns',
    'test_comparison_normalizes_disco_evolution_count_fields',
    'test_comparison_normalizes_disco_same_artifact_provenance',
    'test_comparison_summarizes_by_system_and_variant',
    'test_comparison_report_renders_validation_and_top_designs',
    'test_write_comparison_report_from_csvs',
    'test_quality_validation_accepts_clean_disco_analysis_row',
    'test_quality_validation_rejects_mixed_evaluation_artifacts',
    'test_quality_validation_reports_threshold_failures',
    'test_quality_validation_report_writes_json_and_issue_csv',
    'test_external_quality_score_uses_pg_rank_style_components',
    'test_external_quality_outputs_merge_metrics_and_report_missing_values',
    'test_immunogen_validation_accepts_category_complete_row',
    'test_immunogen_validation_reports_category_failures',
    'test_immunogen_validation_report_writes_json_and_issue_csv',
    'test_comparison_validation_accepts_completed_disco_summary_with_diagnostics',
    'test_comparison_validation_allows_missing_summaries_for_planned_matrix',
    'test_comparison_validation_requires_both_systems_by_default',
    'test_comparison_validation_rejects_unmatched_disco_confidence_and_length',
    'test_comparison_validation_report_writes_json_and_issue_csv',
]:
    getattr(mod, name)()
    print(name, 'ok')
PY
```

## Alternating logit guidance example

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.enabled=true \
  cogen_guidance.alternating.enabled=true \
  'cogen_guidance.alternating.motif_list1=[GP,KVK]' \
  'cogen_guidance.alternating.motif_list2=[DR]' \
  cogen_guidance.alternating.logit_guidance_enabled=true \
  cogen_guidance.alternating.logit_gradient_scale=1.0 \
  cogen_guidance.save_best.enabled=true \
  cogen_guidance.save_best.select_best_output=true \
  cogen_guidance.alternating.combined_weight=1.0
```

The logit guidance changes only the probability landscape consumed by the
existing DISCO sampler. The hard alternating score is still the audit and
save-best signal.

Transition-only comparison variant:

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.enabled=true \
  cogen_guidance.alternating.enabled=true \
  'cogen_guidance.alternating.motif_list1=[GP,DK]' \
  'cogen_guidance.alternating.motif_list2=[DR,VW]' \
  cogen_guidance.alternating.logit_guidance_enabled=false \
  cogen_guidance.alternating.transition_logit_guidance_enabled=true \
  cogen_guidance.alternating.transition_logit_gradient_scale=1.0 \
  cogen_guidance.save_best.enabled=true \
  cogen_guidance.save_best.select_best_output=true \
  cogen_guidance.alternating.combined_weight=1.0
```

## Immunogen mutable-window example

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.enabled=true \
  'cogen_guidance.motifs=[GHK,KTTKS]' \
  cogen_guidance.immunogen.enabled=true \
  'cogen_guidance.immunogen.fixed_fragments=[{start_1based:2,sequence:CD,label:bcell}]' \
  cogen_guidance.immunogen.mutable_logit_guidance.enabled=true \
  cogen_guidance.immunogen.mutable_logit_guidance.coverage_guided=true \
  cogen_guidance.immunogen.spacing_aware=true
```

The controller blocks fixed fragment positions in `can_change_idx`, so the
existing DISCO sampler treats them as non-designable. The mutable-window
guidance then scores motif starts only when the whole motif lies outside those
fixed fragments.

## Evolution dry run

```bash
python runner/guidance_evolution.py \
  --motifs GHK,KTTKS \
  --input-json-path input_jsons/unconditional_config.json \
  --out /tmp/disco_guidance_evolution_dry_run \
  --generations 1 \
  --population-size 2 \
  --dry-run \
  --run-unit-checks
```

This writes command/manifests without starting DISCO inference. Remove
`--dry-run` only for an actual model run.

When `--template-from-elites` is enabled, later-generation genomes can carry
PG-style motif templates. Those templates use amino acid letters for fixed
motif residues and `X` for still-designable positions. The runner writes a
per-individual `input_template.json`, converting `X` to DISCO's `-` mask
character and keeping only input JSON records with matching protein-chain
length.

## Analyze a completed DISCO dump

```bash
python runner/guidance_analyze_disco.py \
  --dump-dir /path/to/disco_output \
  --motifs GHK,KTTKS \
  --out-csv /path/to/disco_analysis.csv \
  --fixed-fragment 1:GHK:bcell \
  --tcell-motifs AAK \
  --cleavage-motifs TT \
  --seed-sequence GHK... \
  --seed-pdb /path/to/seed.pdb \
  --require-confidence
```

Use this after normal DISCO inference runs. It does not start inference. The
CSV contains sequence, matched confidence, hard motif counts, optional fixed
fragment preservation, optional T-cell/cleavage exact motif counts, and
optional alternating metrics. If the run wrote `diagnostics/*_run_metadata.json`,
the analyzer also carries runtime seconds and diagnostic trace paths into the
CSV. With seed inputs, it adds seed sequence identity, changed positions, and
PDB CA-RMSD to seed when the matched DISCO structure is a PDB file.

## Plan second-pass reprediction

```bash
python runner/guidance_plan_reprediction.py \
  --analysis-csv /path/to/disco_immunogen_analysis.csv \
  --out /path/to/reprediction_plan \
  --motifs GHK,KTTKS \
  --seeds 0,1,2 \
  --n-cycle 4 \
  --n-step 100 \
  --fixed-fragment 1:GHK:bcell \
  --require-fixed-preserved \
  --require-same-evaluation-artifact \
  --max-designs 10
```

This writes fixed-sequence DISCO input JSONs and command files for a clean
second pass. It does not run inference. The generated inference command disables
`cogen_guidance` so the final hard sequence is re-evaluated rather than edited.

## Plan sequence-level scaffolds

```bash
python runner/guidance_plan_scaffold.py \
  --length 12 \
  --name scaffold_smoke \
  --fixed-fragment 3:GHK:motif \
  --seed-sequence AAAACCCCGGGG \
  --fixed-range 9:12:tail \
  --pg-seed-pdb /path/to/seed.pdb \
  --pg-contig A1-12,0 \
  --out-input-json /tmp/disco_scaffold/input.json \
  --out-manifest-json /tmp/disco_scaffold/manifest.json
```

Use this when you want the DISCO-expressible subset of a PG scaffold: fixed
sequence letters plus designable `-` positions. The manifest will explicitly
mark PG PDB coordinate templates, structural contigs, and separate
`inpaint_seq`/`inpaint_str` masks as unsupported.

## Analyze guidance trajectories

```bash
python runner/guidance_analyze_trajectory.py \
  --diagnostics-jsonl /path/to/diagnostics/sample_guidance_diagnostics.jsonl \
  --out-csv /path/to/sample_trajectory.csv \
  --out-summary-json /path/to/sample_trajectory_summary.json
```

Use this after a guided DISCO run has written diagnostics. It does not start
inference. The output CSV has one row per diagnostic step with score deltas,
motif-count deltas, sequence changes from the previous step, best/final flags,
and sampler uncertainty fields. The JSON summary tells you whether the best
intermediate scored higher than the final sample.

## Dump coordinate trajectories

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.trajectory_dump.enabled=true \
  cogen_guidance.trajectory_dump.dtype=float16 \
  cogen_guidance.trajectory_dump.max_steps=20
```

This writes `.pt` tensors plus `*_trajectory_manifest.json` under
`dump_dir/trajectories/`. It is disabled by default because coordinate
trajectories can be large; use the JSONL/CSV trajectory analyzer for lightweight
score evolution.

## Create a run bundle

```bash
python runner/guidance_bundle_run.py \
  --run-dir /path/to/planned_run_dir \
  --dump-dir /path/to/disco_output \
  --summary-csv /path/to/disco_analysis.csv \
  --out-json /path/to/run_bundle.json \
  --issues-csv /path/to/run_bundle_issues.csv \
  --require-plan \
  --require-summary \
  --require-diagnostics
```

Use this after `command.txt` and `analysis_command.txt` have produced outputs.
The bundle is a compact reproducibility artifact: it records commands,
manifests, score/provenance counts, diagnostics runtime metadata, file
inventories, and bounded SHA256 hashes without embedding large structures.

## Plan PG-vs-DISCO runs

```bash
python runner/guidance_plan_comparison.py \
  --out /path/to/pg_vs_disco_plan \
  --systems DISCO,ProteinGenerator \
  --modes standard,immunogen,alternating \
  --variant-set core \
  --seeds 0,1,2 \
  --n-cycle 4 \
  --disco-n-step 100 \
  --pg-standard-T 60 \
  --pg-short-T 4 \
  --fixed-fragment 1:GHK:bcell \
  --pg-seed-pdb /path/to/seed.pdb \
  --pg-seed-sequence GHK... \
  --pg-fixed-fragments-json /path/to/fixed_fragments.json
```

This writes `planned_runs.csv`, per-run `command.txt` and
`analysis_command.txt`, length-specific DISCO `input.json` files, per-run
manifests, and `compare_command.txt`. It does not execute inference. If PG
Immunogen inputs are missing, that row is marked
`requires_pg_immunogen_seed_inputs` instead of inventing a command.
If DISCO Immunogen fixed fragments are missing, that row is marked
`requires_immunogen_fixed_fragments`.

## Mac-local comparison smoke

Current smoke folders:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04/
outputs/pg_disco_compare_local_immunogen_smoke_2026_06_04/
```

They contain tiny Standard/Alternating and Immunogen matrices for DISCO and
ProteinGenerator with seed `0`, length `16`, N-cycle `1`, DISCO `N_step=2`, and
PG `T=2`. Planning validation, generation, analysis, comparison, regular
validation, report rendering, and provenance writing completed.

Useful files:

```text
planned_runs.csv
normalized_designs.csv
comparison_summary.csv
COMPARISON_REPORT.md
EXECUTION_REPORT.md
validation_report.json
strict_confidence_validation_report.json
```

DISCO CPU smoke overrides used for the local runs:

```bash
fabric.accelerator=cpu \
fabric.strategy=auto \
use_deepspeed_evo_attention=false \
structure_encoder.args.device=cpu \
dtype=float32
```

The normal validation passes with `--no-require-confidence`. Strict confidence
validation intentionally fails for DISCO because the current DISCO inference
path does not emit native same-sample pLDDT/summary confidence. Use the
`EXECUTION_REPORT.md` files for exact smoke results and provenance caveats.

## Audit plan fairness

```bash
python runner/guidance_audit_fairness.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/fairness_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/fairness_issues.csv \
  --recommended-n-cycle 4
```

Run this before expensive jobs. It checks required systems, replicate balance,
seed sets, shared `n_cycle`, lengths, motif lists, designs per replicate, and
the presence of positive DISCO `N_step` / ProteinGenerator `T` budget knobs.

## Audit PG-to-DISCO implementation parity

```bash
python runner/guidance_audit_parity.py \
  --disco-root . \
  --pg-root ../ProteinGenerator_MOTIFS \
  --out-report-json /tmp/disco_parity/parity.json \
  --out-md /tmp/disco_parity/parity.md
```

This does not run either model. It checks the implementation map: which
ProteinGenerator features have DISCO modules/tests, which are partial, and
which remain unported. Use `--require-all-ported` only as a final completion
gate; during active development it is expected to fail while partial/open items
remain intentional.

## Validate a planned/completed comparison

Planning-stage audit, before expensive model runs:

```bash
python runner/guidance_validate_comparison.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/validation_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/validation_issues.csv \
  --allow-missing-summaries
```

Completed-run audit with DISCO diagnostics required:

```bash
python runner/guidance_validate_comparison.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/validation_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/validation_issues.csv \
  --require-diagnostics
```

By default, every mode/variant group must contain both `DISCO` and
`ProteinGenerator`. Use `--required-systems DISCO` only for local single-system
smoke checks.

## Validate completed designs

```bash
python runner/guidance_validate_quality.py \
  --summary-csv /path/to/disco_analysis.csv \
  --out-report-json /path/to/quality_report.json \
  --out-issues-csv /path/to/quality_issues.csv \
  --require-fixed-preserved \
  --require-exact-confidence \
  --require-diagnostics \
  --require-structure \
  --require-same-evaluation-artifact \
  --min-confidence 0.60 \
  --min-motif-coverage 1.0 \
  --max-homopolymer 4 \
  --min-entropy 0.50
```

Use this after `guidance_analyze_disco.py` or on normalized comparison rows.
`--require-same-evaluation-artifact` is the strict provenance gate: DISCO rows
must have `evaluation_status=same_sample_complete` or
`same_sample_no_diagnostics`, while PG rows can pass via
`score_provenance_status=pg_sequence_and_plddt_from_same_saved_pdb`. Optional
numeric gates include `--max-rmsd` and `--min-spread-score`; pass `none` to
skip a numeric gate.

## Merge external quality metrics

```bash
python runner/guidance_merge_external_quality.py \
  --summary-csv /path/to/disco_analysis.csv \
  --external-metrics-csv /path/to/external_metrics.csv \
  --out-csv /path/to/disco_analysis_with_external_quality.csv \
  --report-json /path/to/external_quality_report.json \
  --top-csv /path/to/top_posthoc_quality.csv
```

Use this after optional external tools have produced ESM-2, RMSD, or Rosetta
CSV columns. The runner merges by `design_id`, computes rank-normalized
components, and writes `posthoc_quality_score` with the documented
ProteinGenerator weights. It does not run ESM-2, ESMFold, or Rosetta.

## Validate Immunogen categories

```bash
python runner/guidance_validate_immunogen.py \
  --summary-csv /path/to/disco_immunogen_analysis.csv \
  --out-report-json /path/to/immunogen_validation.json \
  --out-issues-csv /path/to/immunogen_validation_issues.csv \
  --tcell-motifs AAK,LLK \
  --cleavage-motifs TT \
  --require-fixed-preserved \
  --require-same-evaluation-artifact
```

This is the category-specific Immunogen gate. Use it after
`guidance_analyze_disco.py` has been run with the matching `--tcell-motifs`
and `--cleavage-motifs` arguments.

## Compare existing CSVs

```bash
python runner/guidance_compare.py \
  --source ProteinGenerator:standard:standard_gradient_coverage:/path/to/pg_motif_summary.csv \
  --source DISCO:standard:standard_gradient_coverage:/path/to/disco_analysis.csv \
  --out-normalized /path/to/normalized_designs.csv \
  --out-summary /path/to/comparison_summary.csv
```

The source format can also be the older `system:variant:path`; the four-part
form keeps mode names in the normalized CSV and Markdown report.

The normalized CSV includes `score_provenance_status`,
`confidence_match_status`, source paths, `motif_score`, and `comparison_score`.
ProteinGenerator Alternating summaries are handled through their native
`num_unique_motifs` and `num_motif_hits` columns. DISCO evolution summaries are
handled through per-motif count columns such as `GHK_count` when
`present_motifs/missing_motifs` are unavailable.

## Write a comparison report

```bash
python runner/guidance_report_comparison.py \
  --normalized-csv /path/to/normalized_designs.csv \
  --summary-csv /path/to/comparison_summary.csv \
  --validation-report-json /path/to/validation_report.json \
  --out-md /path/to/comparison_report.md
```

This creates a Markdown review artifact with validation status, aggregate
results, top designs by `comparison_score`, provenance counts, and caveats. It
does not replace validation or real model-run evidence.

## Mac-local smoke notes

The completed local smoke lives at:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04
```

Use the ProteinGenerator Conda env for PG commands:

```text
/opt/anaconda3/envs/proteingenerator-mac/bin/python
```

Use the cached DISCO checkpoint explicitly for DISCO commands:

```text
~/.cache/huggingface/hub/models--DISCO-Design--DISCO/snapshots/4f2ff0088c8964ab8fb3fd9938a8670e1dac3263/DISCO.pt
```

For Mac CPU DISCO smoke runs, append:

```text
fabric.accelerator=cpu fabric.strategy=auto use_deepspeed_evo_attention=false
structure_encoder.args.device=cpu dtype=fp32
```

DISCO may log `succeeded` and write outputs, then hang during Python shutdown
with a multiprocessing semaphore warning. The local smoke records this in
`shutdown_note.txt`; do not treat the post-output shutdown hang as a failed
model sample, but do keep it visible in execution reports.
