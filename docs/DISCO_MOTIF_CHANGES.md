# DISCO Motif Guidance Change Log

## 2026-06-05 Provenance-aligned local comparison smokes

### Decisions

- Treat pLDDT/confidence as comparable only when it is computed from the exact
  saved protein artifact summarized by the row.
- Keep DISCO final samples, coordinate trajectory tensors, and intermediate
  timestep rows separate. Intermediate rows must not inherit final pLDDT.
- Break DISCO save-best score ties by preferring fewer unresolved sequence-mask
  characters and then later timesteps. This prevents equal-score step-0 masked
  states from replacing fully decoded final sequences.
- Use the local ProteinGenerator Conda environment for PG Immunogen because the
  Docker/Colima daemon was unavailable on this Mac.
- Include the seed `.trb` beside the seed PDB for direct ProteinGenerator
  Immunogen partial diffusion; without it the base PG sampler has no usable
  contig/partial-diffusion context.

### Files changed

`disco/model/cogen_inference/guidance/artifact_provenance.py`

- Added final sample provenance manifests.
- Added intermediate-state CSV indexes with one row per saved structure-state
  kind per timestep.
- Records final sequence, final PDB, diagnostics, trajectory manifest, hashes,
  and confidence files only when they exist.

`runner/inference.py`

- Writes DISCO provenance manifests after final structures/sequences and
  optional trajectories are saved.
- Keeps missing DISCO confidence explicit instead of filling fake pLDDT.

`disco/model/cogen_inference/guidance/controller.py`

- Adds deterministic save-best tie-breaking:
  score first, fewer `-` mask characters second, later step third.

`disco/model/cogen_inference/guidance/disco_analysis.py`

- Preserves `-` mask characters when parsing DISCO sequence text for length
  checks.
- Carries provenance manifest and intermediate-state index paths into analysis
  rows.

`disco/model/cogen_inference/guidance/experiment_plan.py`

- Enables DISCO trajectory capture by default in comparison plans.
- Adds direct ProteinGenerator Immunogen planning and includes an explicit or
  inferred seed `.trb` when available.

`runner/guidance_plan_comparison.py`

- Adds `--pg-immunogen-direct`.
- Adds `--pg-seed-trb`.
- Adds DISCO trajectory planning flags.

`docs/`

- Documents final/intermediate artifact separation and the same-artifact
  pLDDT/confidence rule.

### Verification

Executed local Standard/Alternating smoke:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04/
Validation pass: 0 error(s), 0 warning(s), 4/4 completed summaries
Strict confidence validation fails for DISCO missing confidence, as expected
```

Executed local Immunogen smoke:

```text
outputs/pg_disco_compare_local_immunogen_smoke_2026_06_04/
Validation pass: 0 error(s), 0 warning(s), 2/2 completed summaries
Strict confidence validation fails for DISCO missing confidence, as expected
```

The DISCO runs write `*_artifact_provenance.json`,
`*_intermediate_states.csv`, and `*_trajectory_manifest.json`. The intermediate
CSV rows are marked `not_computed_for_intermediate_state` for structure
confidence.

Executed longer local T20/S3 comparison:

```text
outputs/pg_disco_compare_t20_s3_2026_06_05/
Validation pass: 0 error(s), 0 warning(s), 18/18 completed summaries
Strict confidence validation fails for the 9 DISCO missing-confidence rows, as expected
```

The first T20 DISCO pass exposed the save-best tie-break issue above. DISCO was
rerun after the fix. Final decoded DISCO Standard and Immunogen rows still have
exact motif score `0` because the saved sequences do not contain `GHK` or `KTT`;
DISCO Alternating placed requested motifs in two of three runs.

## 2026-06-04 Initial guidance scaffold

### Decisions

- Keep DISCO core changes minimal.
- Implement PG-style motif behavior as an optional controller, not as a forked
  DISCO sampler.
- Leave the existing `PathPlanningSamplingStrategy` unchanged for the first
  port. It remains responsible for turning logits into sequence states.
- Separate soft/logit guidance from hard exact-sequence scoring.

### Core files changed

`configs/base.yaml`

- Added default Hydra group `cogen_guidance: default`.

`disco/model/disco.py`

- Passes `self.configs.cogen_guidance` into `sample_diffusion_cogen`.

`disco/model/cogen_inference/cogen_generator.py`

- Accepts optional `cogen_guidance`.
- Forwards it into `InferenceLoopImpl`.

`disco/model/cogen_inference/cogen_inference_loop_body_impl.py`

- Builds optional `GuidanceController`.
- Calls `modify_logits()` after DISCO produces sequence decoder logits and
  before the existing sampler runs.
- Calls `observe_step()` after sequence update and adds guidance metrics to
  diagnostic records.

`disco/model/cogen_inference/cogen_inference_loop_strategies.py`

- If configured, replaces final output tensors with the controller's best
  observed sequence/structure state.

### New files

`configs/cogen_guidance/default.yaml`

- Disabled-by-default config for motifs, logit guidance, hard scoring, and
  save-best behavior.

`disco/model/cogen_inference/guidance/__init__.py`

- Public package exports.

`disco/model/cogen_inference/guidance/scoring.py`

- `HardMotifScorer`
- motif parsing helpers
- exact motif hit counting
- token-to-sequence conversion

`disco/model/cogen_inference/guidance/alternating.py`

- two-list hard motif hit detection
- deterministic non-overlapping hit path
- alternation, balance, coverage, terminal-tail, and start-state metrics
- scalar combined alternating score

`disco/model/cogen_inference/guidance/immunogen.py`

- fixed-fragment data model
- fixed-fragment spec validation
- fixed-sequence mask construction
- exact fixed-fragment preservation scoring

`disco/model/cogen_inference/guidance/motif.py`

- `MotifLogitGuidance`
- gradient-style motif objective over logits
- local top-k motif rescue
- designable-position masking via `can_change_idx`

`disco/model/cogen_inference/guidance/controller.py`

- `GuidanceController`
- optional logit modification
- hard motif scoring per step
- optional alternating-score contribution
- optional immunogen fixed-fragment score contribution
- combined score tracking
- best-state storage

`tests/test_guidance.py`

- Unit checks for scorer, gradient guidance, local rescue fixed-position
  handling, and controller best-state tracking.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  tests/test_guidance.py
```

Executed test functions directly because `pytest` is not installed in the DISCO
`.venv`:

```text
test_hard_motif_scorer_counts_overlaps ok
test_gradient_guidance_increases_requested_motif_logits ok
test_local_rescue_respects_fixed_positions ok
test_controller_tracks_best_step ok
test_alternating_score_prefers_switching_groups ok
test_alternating_metrics_report_pattern ok
test_immunogen_fixed_mask_and_scoring ok
```

Hydra compose check:

```text
cogen_guidance.enabled= False
logit_guidance.enabled= False
```

### Known limitations after this step

- This is the general motif scaffold only.
- Alternating hard scoring was ported at this point; the later section below
  adds slot-based alternating logit guidance.
- Immunogen fixed-fragment scoring was ported at this point; the later section
  below adds mutable-window guidance and spacing scoring.
- Save-best combined currently ranks the first sample when duplicated sequence
  batches are used.
- No long DISCO generation run was executed in this step.

## 2026-06-04 Alternating logit guidance port

### Decisions

- Port the slot-based ProteinGenerator `AlternatingMotifsBias` first because it
  was the chosen default in the ProteinGenerator MOTIFS workflow.
- Keep the transition-only alternating potential out of the active DISCO port
  for now because it was already marked experimental in the ProteinGenerator
  source.
- Keep the existing DISCO sampler unchanged. Alternating guidance modifies
  decoder logits before the sampler, and hard alternating scoring audits the
  sampled sequence after the sampler.
- Use separate `logit_*` config keys for the soft guidance weights so hard
  save-best scoring weights can be tuned independently.

### Files changed

`disco/model/cogen_inference/guidance/alternating.py`

- Added `AlternatingLogitGuidance`.
- Added broad slot planning, Gaussian slot masks, list1/list2 soft motif-window
  scoring, coverage objective, and list1-tail objective.
- Added diagnostics for objective, transition, coverage, tail, slots, and
  gradient norm.

`disco/model/cogen_inference/guidance/controller.py`

- Wires optional alternating logit guidance into `modify_logits()`.
- Allows alternating logit guidance to run with or without the plain motif
  logit guidance.

`configs/cogen_guidance/default.yaml`

- Added `cogen_guidance.alternating.logit_guidance_enabled`.
- Added logit-only alternating knobs:
  `logit_gradient_scale`, `logit_temperature`, `logit_max_slots`,
  `logit_slot_gap`, `logit_slot_width`, `logit_tail_weight`,
  `logit_coverage_weight`, and `logit_transition_weight`.

`tests/test_guidance.py`

- Added tests for alternating logit objective improvement.
- Added fixed-position masking coverage.
- Added controller coverage for alternating-only logit guidance.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  tests/test_guidance.py
```

Executed direct CPU tests:

```text
test_hard_motif_scorer_counts_overlaps ok
test_gradient_guidance_increases_requested_motif_logits ok
test_local_rescue_respects_fixed_positions ok
test_controller_tracks_best_step ok
test_alternating_score_prefers_switching_groups ok
test_alternating_metrics_report_pattern ok
test_alternating_logit_guidance_increases_slot_objective ok
test_alternating_logit_guidance_respects_fixed_positions ok
test_alternating_logit_guidance_noops_when_motifs_do_not_fit ok
test_controller_can_apply_alternating_without_plain_motif_guidance ok
test_immunogen_fixed_mask_and_scoring ok
```

Hydra compose checks passed for:

- alternating logit guidance plus save-best,
- plain motif logit guidance combined with alternating logit guidance.

### Known limitations after this step

- No long DISCO generation run was executed.
- Transition-only alternating soft guidance is not ported.
- Alternating guidance still has no structural motif anchoring; it only pushes
  amino-acid logits.

## 2026-06-04 Immunogen mutable-window port

### Decisions

- Port ProteinGenerator's active immunogen mutable-window potential before
  attempting final re-prediction semantics.
- Keep the DISCO sampler unchanged. Fixed fragments are protected by a small
  controller hook that sets those positions to non-designable in `can_change_idx`.
- Keep mutable-window motif guidance separate from plain motif guidance so
  immunogen runs do not reward motif windows that overlap protected fragments.
- Port spacing-aware scoring as a hard ranking/diagnostic term, not as a
  post-repair sequence editor.

### Files changed

`disco/model/cogen_inference/cogen_inference_loop_body_impl.py`

- Calls `GuidanceController.modify_change_mask()` once per step before noisy
  guidance remasking and sequence sampling.

`disco/model/cogen_inference/guidance/immunogen.py`

- Added `build_fixed_position_mask()` for coordinate-only fixed masks.
- Added `MutableWindowMotifLogitGuidance`, the DISCO logits equivalent of
  ProteinGenerator `immunogen_mutable_motifs_bias`.
- Added motif spacing helpers ported from PG:
  `motif_start_positions`, `best_motif_start_set`, `spacing_metrics`, and
  `spacing_score`.

`disco/model/cogen_inference/guidance/controller.py`

- Blocks fixed immunogen positions in `can_change_idx`.
- Wires optional immunogen mutable-window logit guidance.
- Adds optional spacing-aware score contribution and diagnostics.
- Keeps fixed-fragment preservation in the hard scoring path.

`configs/cogen_guidance/default.yaml`

- Added immunogen-specific motif overrides.
- Added `protect_fixed_fragments`.
- Added `mutable_logit_guidance` config.
- Added `spacing_aware`, `spacing_weight`, and `spacing_target_distance`.

`tests/test_guidance.py`

- Added fixed-position mask coverage.
- Added mutable-window no-op and valid-window tests.
- Added controller tests for fixed-position blocking and mutable guidance.
- Added spacing-score preference test.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  tests/test_guidance.py
```

Executed direct CPU tests:

```text
test_hard_motif_scorer_counts_overlaps ok
test_gradient_guidance_increases_requested_motif_logits ok
test_local_rescue_respects_fixed_positions ok
test_controller_tracks_best_step ok
test_alternating_score_prefers_switching_groups ok
test_alternating_metrics_report_pattern ok
test_alternating_logit_guidance_increases_slot_objective ok
test_alternating_logit_guidance_respects_fixed_positions ok
test_alternating_logit_guidance_noops_when_motifs_do_not_fit ok
test_controller_can_apply_alternating_without_plain_motif_guidance ok
test_immunogen_fixed_mask_and_scoring ok
test_immunogen_fixed_position_mask_does_not_need_seed_sequence ok
test_immunogen_mutable_window_guidance_ignores_fixed_overlap ok
test_immunogen_mutable_window_guidance_uses_mutable_windows ok
test_controller_blocks_immunogen_fixed_positions_in_change_mask ok
test_controller_can_apply_immunogen_mutable_guidance ok
test_immunogen_spacing_score_prefers_spread_motifs ok
```

Hydra compose checks passed for:

- immunogen fixed fragments plus mutable-window guidance,
- immunogen-specific motifs with top-level motifs unset.

### Known limitations after this step

- No long DISCO generation run was executed.
- PG `immunogen_repredict` is not ported yet; DISCO needs a separate final
  re-evaluation design that keeps sequence/structure/scores aligned.
- T-cell and cleavage motif handling currently means "user-provided motif
  lists"; no domain-specific preset library or validation workflow is ported.

## 2026-06-04 Evolutionary outer-loop port

### Decisions

- Port ProteinGenerator Approach 3 as an external runner, not as a DISCO sampler
  or model change.
- Keep the core search logic in `guidance/evolution.py` so it is testable
  without launching DISCO.
- Make `runner/guidance_evolution.py` dry-run capable. Local tests write
  commands/manifests but do not run model inference.
- Use DISCO-native knobs in the genome: `sample_diffusion.N_step`,
  `model.N_cycle`, logit guidance scale/weights/top-k/temperature, local
  rescue parameters, and save-best confidence/motif weighting.

### Files changed

`disco/model/cogen_inference/guidance/evolution.py`

- Added `DiscoGuidanceGenome` and `DiscoEvaluation`.
- Added initial population, mutation, crossover, tournament selection, and
  next-generation logic.
- Added DISCO Hydra command construction.
- Added hard fitness scoring and DISCO sequence-output parsing.
- Added elite-template input materialization. PG-style templates use amino acid
  letters for fixed motif residues and `X` for designable positions; the writer
  converts `X` to DISCO's protein mask character `-` and keeps only input JSON
  records with matching `proteinChain` length.

`runner/guidance_evolution.py`

- Added a dry-run capable evolutionary runner.
- Writes `command.txt`, `batch_manifest.json`, per-generation CSVs,
  `live_evaluations.csv`, and `manifest.json`.
- Can analyze completed DISCO dump directories by reading `sequences/*.txt` and
  confidence JSONs under `pdbs/`.

`tests/test_guidance.py`

- Added command-construction coverage.
- Added mutation fixed-`N_cycle` coverage.
- Added next-generation and motif-template coverage.
- Added template-to-DISCO conversion, length filtering, and materialized input
  JSON coverage.
- Added fitness-mode and DISCO sequence-parser coverage.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_evolution.py \
  tests/test_guidance.py \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py
```

Executed direct CPU tests including the evolution tests:

```text
test_evolution_command_contains_guidance_overrides ok
test_evolution_mutation_respects_fixed_n_cycle ok
test_evolution_next_generation_preserves_population_size ok
test_evolution_template_to_disco_sequence_converts_x_to_mask ok
test_evolution_apply_template_filters_matching_lengths_and_preserves_context ok
test_evolution_materializes_template_input_json ok
test_evolution_fitness_modes_and_sequence_parser ok
```

Executed dry-run runner:

```bash
.venv/bin/python runner/guidance_evolution.py \
  --motifs GHK,KTTKS \
  --input-json-path input_jsons/unconditional_config.json \
  --out /tmp/disco_guidance_evolution_dry_run \
  --generations 1 \
  --population-size 2 \
  --dry-run \
  --run-unit-checks
```

Dry-run wrote:

```text
/tmp/disco_guidance_evolution_dry_run/manifest.json
/tmp/disco_guidance_evolution_dry_run/generation_000_commands.txt
/tmp/disco_guidance_evolution_dry_run/generation_000.csv
/tmp/disco_guidance_evolution_dry_run/evolution_summary.csv
```

### Known limitations after this step

- No real DISCO evolutionary run was executed.
- No real multi-generation DISCO run has validated elite-template input JSONs
  against model output yet; the conversion and runner planning paths are
  covered by CPU tests and dry-runs.
- Fitness uses DISCO sequence files and available confidence JSONs; a richer
  structure/ligand-aware comparison score is still pending.

## 2026-06-04 PG-vs-DISCO comparison scaffold

### Decisions

- Keep comparison separate from generation. The comparison runner consumes
  existing summary CSVs and does not start ProteinGenerator or DISCO.
- Normalize rows to shared fields before aggregation: system, variant, sequence,
  confidence, unique motif count, total motif count, and fixed-fragment status
  when available.
- Treat ProteinGenerator `pg_plddt_mean` and DISCO `confidence` as comparable
  confidence-like fields only after explicit normalization. This is a scaffold,
  not evidence that the metrics are scientifically identical.

### Files changed

`disco/model/cogen_inference/guidance/comparison.py`

- Added `NormalizedDesignRow`.
- Added ProteinGenerator and DISCO row normalizers.
- Added grouped summary aggregation.
- Added CSV writers for normalized and aggregate rows.
- Preserves DISCO `fixed_fragments_preserved` when available from completed-dump
  analysis rows.

`runner/guidance_compare.py`

- Added CLI for `system:variant:path` CSV sources.
- Writes normalized design rows and aggregate summary rows.

`tests/test_guidance.py`

- Added normalization and grouped-summary tests.

### Verification

Executed compile and direct CPU tests, including:

```text
test_comparison_normalizes_pg_and_disco_rows ok
test_comparison_summarizes_by_system_and_variant ok
```

Executed synthetic comparison runner:

```bash
.venv/bin/python runner/guidance_compare.py \
  --source ProteinGenerator:std:/tmp/disco_guidance_compare_test/pg.csv \
  --source DISCO:std:/tmp/disco_guidance_compare_test/disco.csv \
  --out-normalized /tmp/disco_guidance_compare_test/normalized.csv \
  --out-summary /tmp/disco_guidance_compare_test/summary.csv
```

The summary output contained one DISCO row and one ProteinGenerator row with
expected mean confidence and motif counts.

### Known limitations after this step

- No real PG-vs-DISCO study was executed.
- The comparison runner itself does not enforce matched budgets, seeds, motif
  lists, or input lengths; use the experiment planner plus the validation
  runner added later in this change log before interpreting results.
- Structure/ligand-aware metrics are not included yet.

## 2026-06-04 Completed DISCO dump analyzer

### Decisions

- Keep completed-output analysis separate from model inference and separate
  from the cross-system comparison reducer.
- Pair DISCO sequence files with `summary_confidence_sample_*.json` by
  `base_id` and sample index when possible. This prevents a motif score from
  one sequence being combined with a confidence value from a different sampled
  structure.
- Reuse the same hard motif, immunogen, spacing, and alternating scorers used
  by the guidance layer so generation-time scoring and post-run analysis do not
  drift.

### Files changed

`disco/model/cogen_inference/guidance/disco_analysis.py`

- Added parsers for `sequences/*.txt`, including multi-entry `>cogen_seq`
  files.
- Added summary-confidence JSON parsing and sample-aware matching.
- Added CSV-ready row construction with sequence, confidence, motif counts,
  motif positions, optional fixed-fragment preservation, optional T-cell and
  cleavage exact motif counts, optional alternating metrics, and motif-spacing
  fields.

`runner/guidance_analyze_disco.py`

- Added a CLI that consumes a completed DISCO dump directory and writes
  `analysis.csv` plus an analysis manifest.
- Supports `--fixed-fragment`, `--fixed-fragments-json`, `--tcell-motifs`,
  `--cleavage-motifs`, alternating motif lists, and `--require-confidence`.

`runner/guidance_evolution.py`

- Updated completed-run analysis to use the sample-aware analyzer rows when
  picking the best evaluated individual.
- Records the matched confidence path and match status in each finished
  individual manifest.

`tests/test_guidance.py`

- Added parser coverage for multi-entry DISCO sequence files.
- Added completed-dump analysis coverage for exact sequence/confidence pairing,
  fixed-fragment preservation, T-cell motif counts, and cleavage motif counts.

### Verification

Executed compile and direct CPU tests, including:

```text
test_disco_analysis_parses_multiple_sequence_entries ok
test_disco_analysis_pairs_sequence_with_matching_confidence ok
```

Executed synthetic analyzer runner:

```bash
.venv/bin/python runner/guidance_analyze_disco.py \
  --dump-dir /tmp/disco_guidance_analyze_test/dump \
  --motifs GHK,KTTKS \
  --out-csv /tmp/disco_guidance_analyze_test/analysis.csv \
  --fixed-fragment 1:GHK:bcell \
  --tcell-motifs AAK \
  --cleavage-motifs TT \
  --require-confidence
```

The output row had `confidence_match_status=exact_sample`, preserved the fixed
fragment, and reported expected motif, T-cell, and cleavage counts.

Executed the comparison runner against the analyzer CSV and a synthetic
ProteinGenerator CSV. The aggregate summary preserved fixed-fragment fractions
for both systems.

### Known limitations after this step

- The analyzer is file-format based; it has not been validated against a real
  long DISCO inference output from this machine in this implementation pass.
- Confidence pairing depends on DISCO's current filename convention:
  `<base>_summary_confidence_sample_<idx>.json`.

## 2026-06-04 PG-vs-DISCO experiment planner

### Decisions

- Add an explicit planning layer before expensive inference. The planner writes
  commands, manifests, length-specific DISCO input JSONs, and comparison
  scaffolds but does not run either model.
- Plan the three required comparison modes as first-class modes: `standard`,
  `immunogen`, and `alternating`.
- Keep `n_cycle` synchronized between DISCO `model.N_cycle` and
  ProteinGenerator `--n_cycle`, defaulting to `4`.
- Treat ProteinGenerator `--T` and DISCO `sample_diffusion.N_step` as budget
  knobs rather than mathematically identical diffusion-step variables. This is
  recorded in every run's fairness notes.
- For DISCO, write one single-length `input.json` per planned run so outputs do
  not mix lengths from `input_jsons/unconditional_config.json`.
- For DISCO Immunogen, write fixed B-cell/immunogen residues directly into the
  `proteinChain` string and mask all other positions with `-`.
- Do not invent missing PG Immunogen seed inputs. If `--pg-seed-pdb`,
  `--pg-seed-sequence`, or `--pg-fixed-fragments-json` is missing, the planned
  PG Immunogen row is marked `requires_pg_immunogen_seed_inputs`.
- Do not write a broken DISCO Immunogen command without fixed fragments. Such
  rows are marked `requires_immunogen_fixed_fragments`.

### Files changed

`disco/model/cogen_inference/guidance/experiment_plan.py`

- Added `ComparisonPlanConfig`, `VariantPlan`, `FixedFragmentPlan`, and
  `PlannedRun`.
- Added core/exploratory variant definitions for Standard, Immunogen, and
  Alternating comparisons.
- Added DISCO command construction with `cogen_guidance` Hydra overrides.
- Added ProteinGenerator command construction for Standard, Immunogen, and
  Alternating workflows.
- Added per-run command/analysis-command/manifest writing.
- Added matrix-level `planned_runs.csv`, `plan_manifest.json`, and
  `compare_command.txt` generation.

`runner/guidance_plan_comparison.py`

- Added CLI wrapper around `plan_runs(...)`.
- Exposes systems, modes, seeds, lengths, motifs, `n_cycle`, DISCO `N_step`, PG
  `T`, fixed fragments, and required PG Immunogen seed inputs.

`tests/test_guidance.py`

- Added fixed-fragment sequence materialization coverage.
- Added full core-matrix planning coverage with both DISCO and ProteinGenerator.
- Added missing-PG-Immunogen-input status coverage.
- Added missing-DISCO-Immunogen-fixed-fragment status coverage.

### Verification

Executed compile and direct CPU tests, including:

```text
test_experiment_plan_writes_fixed_fragment_sequence ok
test_experiment_plan_materializes_core_matrix ok
test_experiment_plan_marks_missing_pg_immunogen_inputs ok
test_experiment_plan_marks_missing_disco_immunogen_fragments ok
```

Executed planner smoke:

```bash
.venv/bin/python runner/guidance_plan_comparison.py \
  --out /tmp/disco_guidance_plan_test/plan \
  --systems DISCO,ProteinGenerator \
  --modes standard,immunogen,alternating \
  --variant-set core \
  --seeds 0 \
  --standard-length 8 \
  --immunogen-length 8 \
  --alternating-length 8 \
  --standard-motifs GHK,KTTKS \
  --immunogen-motifs AAK \
  --alternating-list1 GP \
  --alternating-list2 DR \
  --fixed-fragment 1:GHK:bcell \
  --pg-seed-pdb /tmp/disco_guidance_plan_test/inputs/seed.pdb \
  --pg-seed-sequence GHKAAAAA \
  --pg-fixed-fragments-json /tmp/disco_guidance_plan_test/inputs/fixed.json
```

The smoke wrote six planned runs: DISCO and ProteinGenerator for Standard,
Immunogen, and Alternating. The DISCO Immunogen input JSON contained
`GHK-----`, and all planned rows had status `planned`.

### Known limitations after this step

- The planner has not executed the planned model commands.
- The planner records budget fairness assumptions but does not yet enforce a
  statistically complete design count, wall-clock budget, or matched hardware
  budget.
- ProteinGenerator `inference.py` in this checkout does not expose a direct
  top-level `--seed` argument. Planned PG replicates use separate output paths
  and `--start_num`; the seed column is a comparison replicate id for PG.

## 2026-06-04 Guidance diagnostics export

### Decisions

- Reuse DISCO's existing `diagnostic_records` list instead of creating a second
  tracing path.
- Keep serialization in a separate guidance module and add only minimal hooks
  to `disco/model/disco.py` and `runner/inference.py`.
- Write lightweight per-step sequence/score diagnostics by default when records
  exist. Do not dump full coordinate trajectory tensors by default because they
  can be large and are not required for the first fair-comparison CSV layer.
- Include runtime seconds and diagnostic file paths in post-run DISCO analysis
  rows so performance and trace provenance travel with each design.

### Files changed

`disco/model/cogen_inference/guidance/diagnostics.py`

- Added tensor-safe JSON conversion.
- Added JSONL and CSV diagnostic writers.
- Added compact diagnostic summary extraction: best step, best score, best
  sequence, final step, final sequence, and whether best-output selection was
  used.
- Added `write_diagnostic_outputs(...)` for the runner.

`disco/model/disco.py`

- Added `diagnostic_records` to the combined prediction dictionary returned by
  `main_inference_loop(...)`.

`runner/inference.py`

- Measures prediction runtime per sample.
- Writes `diagnostics/<sample>_guidance_diagnostics.jsonl`,
  `diagnostics/<sample>_guidance_diagnostics.csv`, and
  `diagnostics/<sample>_run_metadata.json`.
- Adds diagnostic paths and runtime fields to the per-sample log dictionary.

`disco/model/cogen_inference/guidance/disco_analysis.py`

- Reads `diagnostics/*_run_metadata.json` and attaches runtime, diagnostic
  paths, record count, best diagnostic step, and best diagnostic score to each
  DISCO analysis row.

`tests/test_guidance.py`

- Added diagnostic summary coverage.
- Added tensor serialization coverage.
- Extended completed-dump analysis coverage to include diagnostic metadata.

### Verification

Executed compile and direct CPU tests, including:

```text
test_diagnostics_summary_selects_best_guidance_step ok
test_write_diagnostic_outputs_serializes_tensor_values ok
test_disco_analysis_pairs_sequence_with_matching_confidence ok
```

### Known limitations after this step

- Full coordinate trajectory tensor persistence is not enabled by default.
- Runtime seconds currently measure the model prediction call, not the full
  wall-clock time including file writes and analysis commands.

## 2026-06-04 PG-vs-DISCO comparison validator

### Decisions

- Add an explicit validation gate between planned/completed runs and final
  comparison reduction.
- Keep validation separate from both generation and aggregation. The validator
  reads `planned_runs.csv` plus completed summary CSVs; it does not execute
  either model and does not normalize scientific results.
- Require both `DISCO` and `ProteinGenerator` by default in every mode/variant
  group. Local smoke checks can intentionally narrow this with
  `--required-systems DISCO`.
- Treat missing completed summaries as errors by default, but allow
  planning-stage audits with `--allow-missing-summaries`.
- Make DISCO confidence provenance explicit: completed DISCO rows must carry
  `confidence_match_status=exact_sample` or `single_confidence_fallback` when
  confidence is required.

### Files changed

`disco/model/cogen_inference/guidance/comparison_validation.py`

- Added `ValidationIssue`.
- Added CSV readers and validation report writers.
- Added required-system checks per `(mode, variant)`.
- Added planned-matrix balance checks.
- Added shared-parameter consistency checks for `n_cycle`, length, motif list,
  and designs per replicate.
- Added planned-artifact checks for `command.txt`, `analysis_command.txt`,
  `run_manifest.json`, and DISCO `input_json_path`.
- Added completed-summary checks for sequence presence/length,
  confidence/pLDDT-like fields, DISCO confidence matching, optional DISCO
  diagnostics, and Immunogen fixed-fragment preservation.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported `ValidationIssue`, `validate_comparison_plan`,
  `write_validation_report`, and `read_csv_dicts`.

`runner/guidance_validate_comparison.py`

- Added CLI wrapper around `validate_comparison_plan(...)`.
- Writes a summary JSON report and structured issue CSV.
- Exits nonzero on validation errors, and optionally on warnings with
  `--fail-on-warning`.

`tests/test_guidance.py`

- Added synthetic comparison-validation tests for:
  - completed DISCO summary with diagnostics,
  - planning-stage missing-summary warning,
  - default required-system failure for DISCO-only plans,
  - bad DISCO sequence length plus unmatched confidence,
  - JSON/CSV validation report writing.

`docs/DISCO_MOTIF_AI_README.md`

- Documented the validator as the comparison gate before scientific
  interpretation.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added quick CLI usage and direct-test entries for the validator.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_evolution.py \
  runner/guidance_compare.py \
  runner/guidance_analyze_disco.py \
  runner/guidance_plan_comparison.py \
  runner/guidance_validate_comparison.py \
  tests/test_guidance.py \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  runner/inference.py
```

Executed all 39 direct CPU test functions; every test printed `ok`.

Executed validator planning-stage smoke:

```bash
.venv/bin/python runner/guidance_validate_comparison.py \
  --planned-runs-csv /tmp/disco_guidance_validation_smoke/planned_runs.csv \
  --out-report-json /tmp/disco_guidance_validation_smoke/validation_missing.json \
  --out-issues-csv /tmp/disco_guidance_validation_smoke/validation_missing_issues.csv \
  --allow-missing-summaries \
  --required-systems DISCO
```

Result:

```text
Validation pass: 0 error(s), 1 warning(s), 0/1 completed summaries
```

Executed validator completed-summary smoke with synthetic diagnostics:

```bash
.venv/bin/python runner/guidance_validate_comparison.py \
  --planned-runs-csv /tmp/disco_guidance_validation_smoke/planned_runs.csv \
  --out-report-json /tmp/disco_guidance_validation_smoke/validation_complete.json \
  --out-issues-csv /tmp/disco_guidance_validation_smoke/validation_complete_issues.csv \
  --required-systems DISCO \
  --require-diagnostics
```

Result:

```text
Validation pass: 0 error(s), 0 warning(s), 1/1 completed summaries
```

### Known limitations after this step

- The validator proves matrix/result-file consistency; it does not prove that
  any model command has been scientifically adequate or biologically successful.
- Hardware budget, wall-clock budget, and final report statistics are still
  outside this validator.
- Structure-aware, ligand-aware, or external re-prediction metrics are not
  added by this step.

## 2026-06-04 Score provenance and comparison report

### Decisions

- Make comparison provenance explicit in the normalized CSV instead of relying
  on comments in documentation.
- Keep the cross-system `comparison_score` simple and transparent:
  `0.5 * confidence + 0.5 * motif_score`.
- Treat failed Immunogen fixed-fragment preservation as invalid for ranking by
  assigning comparison score `0.0`.
- Preserve backward compatibility with existing `system:variant:path` compare
  sources, but support `system:mode:variant:path` so Standard, Immunogen, and
  Alternating mode labels travel into reports.
- Fix ProteinGenerator Alternating normalization. Those summaries use
  `num_unique_motifs` and `num_motif_hits`, not the standard
  `num_unique_motifs_present` and `num_motif_occurrences_total` fields.
- Handle DISCO evolution summaries that provide per-motif `MOTIF_count` fields
  but no `present_motifs/missing_motifs` columns.
- Add Markdown reporting as a review artifact after validation and CSV
  reduction. It does not replace validation or real model-run evidence.

### Files changed

`disco/model/cogen_inference/guidance/comparison.py`

- Added score-provenance fields to `NormalizedDesignRow`:
  `confidence_metric`, `confidence_match_status`, `confidence_path`,
  `sequence_path`, `score_provenance_status`, diagnostics path, runtime, motif
  score, motif coverage, and comparison score.
- Added robust ProteinGenerator column handling for Standard, Immunogen, and
  Alternating summary CSVs.
- Added per-motif count-column fallback for DISCO evolution summaries.
- Added DISCO confidence provenance categories:
  `disco_exact_sequence_confidence_match`,
  `disco_single_confidence_fallback`,
  `disco_unmatched_confidence`, and
  `disco_confidence_provenance_unknown`.
- Extended aggregate summaries with best-confidence design,
  best-comparison-score design, motif score aggregates, confidence exact/fallback
  fractions, and runtime means.

`disco/model/cogen_inference/guidance/comparison_report.py`

- Added Markdown report rendering from normalized CSV, summary CSV, and optional
  validation JSON.
- Added aggregate table, top-design table, provenance counts, and
  interpretation guardrails.

`runner/guidance_compare.py`

- Accepts both `system:variant:path` and `system:mode:variant:path`.

`runner/guidance_report_comparison.py`

- Added CLI wrapper for Markdown report generation.

`disco/model/cogen_inference/guidance/experiment_plan.py`

- Updated `compare_command.txt` scaffolding to emit
  `system:mode:variant:path` sources.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported `render_comparison_report` and `write_comparison_report`.

`tests/test_guidance.py`

- Added ProteinGenerator Alternating normalization coverage.
- Added DISCO evolution count-field normalization coverage.
- Added comparison report rendering coverage.
- Added comparison report CSV-to-Markdown coverage.

`docs/DISCO_MOTIF_AI_README.md`

- Documented score-provenance fields, `comparison_score`, four-part source
  specs, and the report runner.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file map, quick-test entries, and CLI examples for the report path.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_evolution.py \
  runner/guidance_compare.py \
  runner/guidance_analyze_disco.py \
  runner/guidance_plan_comparison.py \
  runner/guidance_validate_comparison.py \
  runner/guidance_report_comparison.py \
  tests/test_guidance.py \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  runner/inference.py
```

Executed all 43 direct CPU test functions; every test printed `ok`.

Executed synthetic compare/report smoke with ProteinGenerator Alternating
columns and DISCO per-motif evolution count columns:

```bash
.venv/bin/python runner/guidance_compare.py \
  --source ProteinGenerator:alternating:alternating_slot_tail_guard:/tmp/disco_guidance_report_smoke/pg_alternating.csv \
  --source DISCO:standard:evolution:/tmp/disco_guidance_report_smoke/disco_evolution.csv \
  --out-normalized /tmp/disco_guidance_report_smoke/normalized.csv \
  --out-summary /tmp/disco_guidance_report_smoke/summary.csv

.venv/bin/python runner/guidance_report_comparison.py \
  --normalized-csv /tmp/disco_guidance_report_smoke/normalized.csv \
  --summary-csv /tmp/disco_guidance_report_smoke/summary.csv \
  --validation-report-json /tmp/disco_guidance_report_smoke/validation.json \
  --out-md /tmp/disco_guidance_report_smoke/report.md \
  --title 'Synthetic PG-vs-DISCO Report'
```

The report contained aggregate rows for DISCO and ProteinGenerator, top designs
by `comparison_score`, and provenance counts including
`pg_sequence_and_plddt_from_same_saved_pdb`.

### Known limitations after this step

- `comparison_score` is a transparent first-pass ranker, not a validated
  biological objective.
- The report summarizes completed CSVs; it does not prove adequate sample size,
  matched hardware budget, or biological validity.
- Additional structure-aware metrics beyond the later PDB CA-RMSD layer, plus
  ligand-aware metrics, still require separate scoring modules or external
  re-prediction workflows.

## 2026-06-04 DISCO analyzer quality metrics

### Decisions

- Port the lightweight quality metrics from ProteinGenerator Immunogen analysis
  into the DISCO completed-dump analyzer.
- Keep these metrics in the analysis layer, not generation. They audit completed
  designs and do not change sampling.
- Add structure-path matching for saved DISCO PDB/CIF files so sequence,
  confidence, motif metrics, diagnostics, and structure provenance can be
  reviewed together.
- Compute CA-RMSD only when the matched DISCO structure and seed are PDB files.
  CIF structures keep their path and match status, but RMSD remains blank
  instead of being guessed through a partial parser.

### Files changed

`disco/model/cogen_inference/guidance/disco_analysis.py`

- Added `DiscoStructureRecord`.
- Added structure-file discovery and matching for `pdbs/*_sample_<idx>.pdb`
  and `pdbs/*_sample_<idx>.cif`.
- Hardened confidence matching for DISCO sequence files whose stem includes a
  `_sample_<idx>` suffix.
- Added `longest_homopolymer`, `normalized_entropy`,
  `sequence_distance`, `ca_coordinates_from_pdb`, and `kabsch_rmsd`.
- Added output fields:
  `structure_path`, `structure_format`, `structure_match_status`,
  `longest_homopolymer`, `aa_entropy_norm`, `seed_sequence_identity`,
  `num_sequence_changes_vs_seed`, `changed_positions_1based`, and
  `ca_rmsd_to_seed`.

`runner/guidance_analyze_disco.py`

- Added `--seed-sequence`.
- Added `--seed-pdb`.
- Prioritized the new quality and structure fields in the CSV writer.
- Records seed input provenance in the analysis manifest.

`disco/model/cogen_inference/guidance/comparison.py`

- Carries `longest_homopolymer`, `aa_entropy_norm`,
  `seed_sequence_identity`, and `ca_rmsd_to_seed` through normalized rows.
- Adds aggregate means for entropy, seed identity, and CA-RMSD when available.

`tests/test_guidance.py`

- Added synthetic PDB-backed analyzer coverage for structure matching,
  homopolymer length, entropy, seed-sequence identity, changed positions, and
  CA-RMSD.

`docs/DISCO_MOTIF_AI_README.md`

- Documented the new analyzer metrics and CLI flags.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Updated the compact workflow and test list.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_evolution.py \
  runner/guidance_compare.py \
  runner/guidance_analyze_disco.py \
  runner/guidance_plan_comparison.py \
  runner/guidance_validate_comparison.py \
  runner/guidance_report_comparison.py \
  tests/test_guidance.py \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  runner/inference.py
```

Executed all 44 direct CPU test functions; every test printed `ok`.

Executed synthetic analyzer smoke:

```bash
.venv/bin/python runner/guidance_analyze_disco.py \
  --dump-dir /tmp/disco_quality_analyzer_smoke/dump \
  --motifs AAA \
  --out-csv /tmp/disco_quality_analyzer_smoke/analysis.csv \
  --seed-sequence AAAT \
  --seed-pdb /tmp/disco_quality_analyzer_smoke/seed.pdb \
  --require-confidence
```

The output row included:

```text
structure_match_status exact_sequence_index
longest_homopolymer 3
aa_entropy_norm 0.188
seed_sequence_identity 0.750
ca_rmsd_to_seed 0.000
confidence_match_status exact_sample
```

### Known limitations after this step

- CA-RMSD is PDB-only in this pass.
- Ligand-aware and external re-prediction metrics are still not implemented.
- The quality metrics audit completed outputs; they are not additional guidance
  objectives.

## 2026-06-04 Completed-design quality validator

### Decisions

- Port the ProteinGenerator `validate_immunogen_summary.py` idea into a shared
  PG/DISCO quality-gate module.
- Keep these gates separate from comparison-matrix validation. Matrix
  validation proves the study is structurally complete; quality validation
  decides whether individual completed designs satisfy acceptance thresholds.
- Make the gates configurable so they can be used for strict Immunogen rows,
  DISCO analysis rows, normalized PG/DISCO rows, or lighter smoke checks.
- Preserve the rule that validators never edit sequences, never choose
  replacement designs, and never launch model inference.

### Files changed

`disco/model/cogen_inference/guidance/quality_validation.py`

- Added `QualityGateConfig`.
- Added `QualityIssue`.
- Added row-level gates for:
  - confidence / pLDDT,
  - exact or fallback confidence provenance,
  - motif coverage,
  - unique motif count,
  - fixed-fragment preservation and mismatch count,
  - longest homopolymer,
  - normalized amino-acid entropy,
  - optional CA-RMSD,
  - optional motif spread score,
  - optional diagnostics file presence,
  - optional structure file presence.
- Added JSON and CSV report writers.

`runner/guidance_validate_quality.py`

- Added CLI wrapper for quality validation.
- Writes `quality_report.json` and `quality_issues.csv`.
- Exits nonzero when any error-level gate fails.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported `QualityGateConfig`, `QualityIssue`, `validate_summary_csv`, and
  `write_quality_report`.

`tests/test_guidance.py`

- Added clean-row quality validation coverage.
- Added multi-gate failure coverage.
- Added quality report JSON/CSV serialization coverage.

`docs/DISCO_MOTIF_AI_README.md`

- Documented completed-design quality validation and CLI usage.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file map, quick-test entries, and a compact quality-validation example.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_evolution.py \
  runner/guidance_compare.py \
  runner/guidance_analyze_disco.py \
  runner/guidance_plan_comparison.py \
  runner/guidance_validate_comparison.py \
  runner/guidance_report_comparison.py \
  runner/guidance_validate_quality.py \
  tests/test_guidance.py \
  disco/model/cogen_inference/cogen_generator.py \
  disco/model/cogen_inference/cogen_inference_loop_body_impl.py \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  disco/model/disco.py \
  runner/inference.py
```

Executed all 47 direct CPU test functions; every test printed `ok`.

Executed passing quality-validator smoke:

```bash
.venv/bin/python runner/guidance_validate_quality.py \
  --summary-csv /tmp/disco_quality_validation_smoke/pass.csv \
  --out-report-json /tmp/disco_quality_validation_smoke/pass_report.json \
  --out-issues-csv /tmp/disco_quality_validation_smoke/pass_issues.csv \
  --require-fixed-preserved \
  --require-exact-confidence \
  --require-diagnostics \
  --require-structure \
  --max-rmsd 1.0 \
  --min-spread-score 0.5
```

Result:

```text
Quality validation pass: 0 error(s), 0 warning(s), 1/1 passed rows
```

Executed failing quality-validator smoke:

```bash
.venv/bin/python runner/guidance_validate_quality.py \
  --summary-csv /tmp/disco_quality_validation_smoke/fail.csv \
  --out-report-json /tmp/disco_quality_validation_smoke/fail_report.json \
  --out-issues-csv /tmp/disco_quality_validation_smoke/fail_issues.csv \
  --require-fixed-preserved \
  --require-exact-confidence \
  --max-rmsd 1.0 \
  --min-spread-score 0.5
```

Result:

```text
Quality validation fail: 7 error(s), 0 warning(s), 0/1 passed rows
```

The failure CSV included `min_confidence`, `exact_confidence_provenance`,
`min_motif_coverage`, `fixed_fragments_preserved`,
`fixed_fragment_mismatches`, `max_homopolymer`, and `min_entropy`.

### Known limitations after this step

- The validator only gates fields that already exist in the CSV. It does not
  compute missing pLDDT, entropy, RMSD, diagnostics, or structure paths.
- Acceptance thresholds are experiment policy, not universal biology.
- Ligand-aware and external re-prediction gates remain future work.

## 2026-06-04 Guidance trajectory analyzer

### Decisions

- Keep intermediate-design analysis fully offline. The analyzer consumes the
  JSONL files already written by `diagnostics.py`; it does not launch DISCO
  inference and does not touch model tensors.
- Treat the diagnostic JSONL as the authoritative per-step record. The new CSV
  therefore derives deltas, best/final markers, and motif gains/losses from the
  same sequence/score records that save-best observes during sampling.
- Keep coordinate trajectory persistence out of scope for this step. The new
  analyzer is a sequence/score/noise trajectory export, not a full structure
  tensor dump.

### Files changed

`disco/model/cogen_inference/guidance/trajectory.py`

- Added JSONL reader for `*_guidance_diagnostics.jsonl`.
- Added filename-based sample-id inference.
- Added `trajectory_rows(...)` to normalize per-step records into a table with
  score deltas, motif-count deltas, sequence Hamming changes, newly gained/lost
  motifs, best-step flags, final-step flags, save-best selection markers, and
  sampler entropy/masking fields.
- Added `trajectory_summary(...)`, `aggregate_summary(...)`, and
  `write_trajectory_outputs(...)` for per-sample and multi-sample summaries.

`runner/guidance_analyze_trajectory.py`

- Added CLI for one or more `--diagnostics-jsonl` files.
- Writes a combined trajectory CSV and a JSON summary.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported the trajectory analysis helpers.

`tests/test_guidance.py`

- Added trajectory row delta/best-step coverage.
- Added JSONL-to-CSV/JSON export coverage.

`docs/DISCO_MOTIF_AI_README.md`

- Documented the trajectory analyzer in the status table, file map, diagnostics
  architecture, CLI usage, and remaining work.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added the trajectory module/runner to the quick file map, implemented feature
  list, direct-test snippet, and CLI examples.

### Verification

Executed:

```bash
.venv/bin/python -m compileall \
  disco/model/cogen_inference/guidance/trajectory.py \
  runner/guidance_analyze_trajectory.py \
  tests/test_guidance.py
```

Executed all 49 direct CPU test functions; every test printed `ok`.

Executed trajectory CLI smoke:

```bash
.venv/bin/python runner/guidance_analyze_trajectory.py \
  --diagnostics-jsonl /tmp/disco_trajectory_smoke/sample_guidance_diagnostics.jsonl \
  --out-csv /tmp/disco_trajectory_smoke/trajectory.csv \
  --out-summary-json /tmp/disco_trajectory_smoke/summary.json
```

Result:

```text
Trajectory analysis wrote 3 step row(s) across 1 sample(s).
```

The smoke CSV marked the intermediate `AAGHKA` row as `is_best_combined=True`
and the final `AAGHAA` row as `best_output_selected_record=True`; the summary
reported `best_beats_final=true`.

### Known limitations after this step

- The analyzer can only report fields that were present in the diagnostic
  records. It does not reconstruct missing logits, structures, or pLDDT.
- Sequence changes are reported as Hamming distance only when consecutive
  sequence strings have the same length.
- Multi-sample inference is summarized file-by-file; per-batch save-best
  semantics still depend on the controller's current first-sample policy.

## 2026-06-04 Alternating transition-only logit guidance

### Decisions

- Port ProteinGenerator's `soft_transition_alternating_motifs_bias` as a second
  DISCO logit-guidance class, not as a sampler change.
- Keep the existing slot-based `AlternatingLogitGuidance` as the default/core
  Alternating path. The transition-only class is exposed as an explicit
  comparison mode through Hydra and the exploratory comparison planner.
- Reuse the existing hard `score_alternation(...)` and save-best combined score
  for audit/ranking. The new class only changes the soft logit pressure before
  DISCO's existing path-planning sampler samples residues.

### Files changed

`disco/model/cogen_inference/guidance/alternating.py`

- Added `SoftTransitionAlternatingLogitGuidance`.
- The class scores ordered soft motif-start pairs:
  - rewards list1->list2 and list2->list1,
  - penalizes list1->list1 and list2->list2,
  - adds coverage, expected-start, late-list2 end, and list1-tail stabilizers.
- Added cached ordered-pair weights per sequence length/device.
- Extended `AlternatingLogitGuidanceResult` with optional same-group, start, and
  end diagnostics while preserving the old `guidance_alternating_logit_*` keys.

`disco/model/cogen_inference/guidance/controller.py`

- Added `alternating_transition_logit_guidance`.
- Added diagnostics under the `guidance_alternating_transition_logit_*` prefix.

`configs/cogen_guidance/default.yaml`

- Added disabled transition-only Hydra knobs:
  `transition_logit_guidance_enabled`,
  `transition_logit_gradient_scale`, `transition_logit_temperature`,
  `transition_same_group_penalty_weight`, `transition_soft_start_weight`,
  `transition_soft_end_weight`, `transition_distance_temperature`, and
  `transition_min_separation`.

`disco/model/cogen_inference/guidance/experiment_plan.py`

- Added `VariantPlan.alternating_guidance_mode`.
- Added exploratory `transition_only_tail_guard`.
- DISCO transition-only planned commands now enable
  `cogen_guidance.alternating.transition_logit_guidance_enabled=true`.
- ProteinGenerator transition-only planned commands use
  `--potentials soft_transition_alternating_motifs_bias`.

`tests/test_guidance.py`

- Added objective preference test for alternating order over same-group order.
- Added logit-update/fixed-position test for the transition-only class.
- Added controller wiring coverage.
- Added exploratory comparison-plan coverage for DISCO/PG transition-only
  command materialization.

`docs/DISCO_MOTIF_AI_README.md`

- Updated status table, Alternating explanation, CLI example, and pending work.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Updated implemented-feature list, quick-test list, not-implemented list, and
  Alternating examples.

### Verification

Executed:

```bash
.venv/bin/python -m compileall \
  disco/model/cogen_inference/guidance/alternating.py \
  disco/model/cogen_inference/guidance/controller.py \
  disco/model/cogen_inference/guidance/__init__.py \
  disco/model/cogen_inference/guidance/experiment_plan.py \
  tests/test_guidance.py
```

Executed all 53 direct CPU test functions; every test printed `ok`.

Executed comparison-planner CLI smoke:

```bash
.venv/bin/python runner/guidance_plan_comparison.py \
  --out /tmp/disco_transition_plan_smoke \
  --systems DISCO,ProteinGenerator \
  --modes alternating \
  --variant-set exploratory \
  --seeds 0 \
  --alternating-length 12 \
  --alternating-list1 GP \
  --alternating-list2 DR
```

Result:

```text
Wrote 6 planned runs to /private/tmp/disco_transition_plan_smoke
Statuses: planned
```

The planned transition-only DISCO command contained
`cogen_guidance.alternating.transition_logit_guidance_enabled=true`; the
ProteinGenerator command contained
`--potentials soft_transition_alternating_motifs_bias`.

### Known limitations after this step

- The transition-only class changes logit pressure only. It does not force a
  discrete alternating pattern and still relies on save-best/hard scoring for
  ranking.
- The exploratory planner includes the transition-only variant, but no real
  DISCO/PG scientific run has been launched in this step.

## 2026-06-04 Same-artifact evaluation provenance

### Decisions

- Treat ProteinGenerator `--immunogen_repredict` as an evaluation invariant:
  final sequence, pLDDT/confidence, and coordinates must describe the same
  design state.
- Do not claim a DISCO second-pass forward equivalent. DISCO stores sequence,
  confidence, structure, and diagnostics as separate files, so the ported
  behavior is a strict provenance bundle that proves whether those files refer
  to the same output sample.
- Keep the provenance logic in a separate module so analysis, normalization, and
  quality gates can share one definition.

### Files changed

`disco/model/cogen_inference/guidance/evaluation_provenance.py`

- Added `EvaluationBundle`.
- Added sample-key matching for sequence, confidence JSON, structure file, and
  diagnostics metadata.
- Added accepted same-artifact status helpers for DISCO and PG comparison rows.

`disco/model/cogen_inference/guidance/disco_analysis.py`

- Adds `evaluation_status`, `evaluation_expected_sample_keys`,
  `evaluation_confidence_sample_key`, `evaluation_structure_sample_key`,
  `evaluation_diagnostics_status`, and `evaluation_notes` to each completed
  DISCO analysis row.
- `same_sample_complete` means sequence, confidence, structure, and diagnostics
  match the same sample.
- `same_sample_no_diagnostics` means sequence, confidence, and structure match,
  but diagnostics are absent.

`disco/model/cogen_inference/guidance/quality_validation.py`

- Added `QualityGateConfig.require_same_evaluation_artifact`.
- The gate accepts DISCO `same_sample_complete` /
  `same_sample_no_diagnostics` and PG
  `pg_sequence_and_plddt_from_same_saved_pdb`.

`runner/guidance_validate_quality.py`

- Added `--require-same-evaluation-artifact`.

`disco/model/cogen_inference/guidance/comparison.py`

- Normalized DISCO rows with exact evaluation bundles now receive
  `score_provenance_status=disco_exact_sequence_structure_confidence_match`.

`tests/test_guidance.py`

- Added same-artifact assertions in DISCO dump analysis tests.
- Added quality-gate rejection coverage for mixed/fallback evaluation artifacts.
- Added comparison-normalization coverage for exact DISCO same-artifact
  provenance.

`docs/DISCO_MOTIF_AI_README.md`

- Documented `evaluation_provenance.py`, the DISCO repredict caveat, the new
  `--require-same-evaluation-artifact` gate, and updated pending work.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added the file-map entry, implemented-feature entry, quick-test entries, and
  quality-validation usage.

### Verification

Executed:

```bash
.venv/bin/python -m compileall \
  disco/model/cogen_inference/guidance/evaluation_provenance.py \
  disco/model/cogen_inference/guidance/disco_analysis.py \
  disco/model/cogen_inference/guidance/quality_validation.py \
  disco/model/cogen_inference/guidance/comparison.py \
  disco/model/cogen_inference/guidance/__init__.py \
  runner/guidance_validate_quality.py \
  tests/test_guidance.py
```

Executed all 55 direct CPU test functions; every test printed `ok`.

Executed synthetic DISCO analysis + quality CLI smoke:

```bash
.venv/bin/python runner/guidance_analyze_disco.py \
  --dump-dir /tmp/disco_eval_provenance_smoke/dump \
  --motifs GHK,KTTKS \
  --out-csv /tmp/disco_eval_provenance_smoke/analysis.csv \
  --require-confidence

.venv/bin/python runner/guidance_validate_quality.py \
  --summary-csv /tmp/disco_eval_provenance_smoke/analysis.csv \
  --out-report-json /tmp/disco_eval_provenance_smoke/quality.json \
  --out-issues-csv /tmp/disco_eval_provenance_smoke/issues.csv \
  --require-exact-confidence \
  --require-structure \
  --require-diagnostics \
  --require-same-evaluation-artifact \
  --min-confidence 0.5 \
  --min-motif-coverage 1.0 \
  --min-unique-motifs 2 \
  --max-homopolymer 3 \
  --min-entropy 0.2
```

Result:

```text
Quality validation pass: 0 error(s), 0 warning(s), 1/1 passed rows
```

The analysis CSV reported `evaluation_status=same_sample_complete`.

### Known limitations after this step

- This is a provenance/evaluation guard, not a new DISCO model forward pass.
- CIF structure identity is matched by filename/sample id; CA-RMSD extraction
  still only runs for PDB files.

## 2026-06-04 Immunogen category validation

### Decisions

- Keep Immunogen category validation separate from generic quality validation.
  A design can pass global motif coverage while still failing a T-cell or
  cleavage sub-objective, so the output needs category-specific checks.
- Validate completed CSV rows only. The validator does not edit designs and
  does not launch either DISCO or ProteinGenerator.
- Treat protected B-cell/fixed-fragment preservation as the first Immunogen
  category. T-cell and cleavage motif coverage are optional categories activated
  by passing motif lists to the runner.

### Files changed

`disco/model/cogen_inference/guidance/immunogen_validation.py`

- Added `ImmunogenValidationConfig`.
- Added category checks for:
  - `bcell_fixed`: `fixed_fragments_preserved` and
    `num_fixed_mismatches`,
  - `tcell`: `tcell_unique_motifs`, `tcell_total_motifs`, and
    `tcell_missing_motifs`,
  - `cleavage`: `cleavage_unique_motifs`, `cleavage_total_motifs`, and
    `cleavage_missing_motifs`,
  - `evaluation`: same-artifact provenance.
- Added JSON/CSV report writers with per-category pass counts.

`runner/guidance_validate_immunogen.py`

- Added CLI for Immunogen category validation.
- Supports `--tcell-motifs`, `--cleavage-motifs`,
  `--min-tcell-coverage`, `--min-cleavage-coverage`,
  `--min-tcell-total`, `--min-cleavage-total`,
  `--require-fixed-preserved`, and
  `--require-same-evaluation-artifact`.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported the Immunogen validation API.

`tests/test_guidance.py`

- Added category-complete pass coverage.
- Added multi-category failure coverage.
- Added JSON/CSV report serialization coverage.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table entry, file-map entries, architecture notes, CLI example,
  implemented coverage, and pending-work update.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature entry, quick-test entries, CLI
  example, and pending-work update.

### Verification

Executed:

```bash
.venv/bin/python -m compileall \
  disco/model/cogen_inference/guidance/immunogen_validation.py \
  disco/model/cogen_inference/guidance/__init__.py \
  runner/guidance_validate_immunogen.py \
  tests/test_guidance.py
```

Executed all 58 direct CPU test functions; every test printed `ok`.

Executed passing Immunogen validator smoke:

```bash
.venv/bin/python runner/guidance_validate_immunogen.py \
  --summary-csv /tmp/disco_immunogen_validation_smoke/pass.csv \
  --out-report-json /tmp/disco_immunogen_validation_smoke/pass_report.json \
  --out-issues-csv /tmp/disco_immunogen_validation_smoke/pass_issues.csv \
  --tcell-motifs GHK,KTTKS \
  --cleavage-motifs AAK \
  --require-same-evaluation-artifact
```

Result:

```text
Immunogen validation pass: 0 error(s), 0 warning(s), 1/1 passed rows
Category pass rows: fixed=1 tcell=1 cleavage=1 evaluation=1
```

Executed failing Immunogen validator smoke with an intentionally invalid row.
The issues CSV contained `fixed_fragments_preserved`,
`fixed_fragment_mismatches`, `tcell_coverage`, `cleavage_coverage`, and
`same_evaluation_artifact`.

### Known limitations after this step

- This validates exact motif strings in category fields that the analyzer
  already wrote. It does not implement immunogenicity prediction or epitope
  binding models.
- T-cell and cleavage motif presets are still caller-provided motif lists; no
  curated biological preset library has been added.

## 2026-06-04 Per-sample save-best for batch inference

### Decisions

- Resolve the earlier first-sample save-best limitation before running
  multi-design comparison batches.
- Keep DISCO's sequence sampler unchanged. The sampler still consumes guided
  logits and executes the path-planning update; save-best selection remains a
  controller/finalization concern.
- Preserve the old single-state API through `get_best_state()` while adding a
  per-sample API for duplicated or batched inference.

### Files changed

`disco/model/cogen_inference/guidance/controller.py`

- Added `sample_index` to `BestGuidanceState`.
- Added `best_states_by_sample` tracking.
- Changed `observe_step()` from first-row-only scoring to a per-sample scoring
  loop.
- Added `get_best_states_by_sample()` for tests and downstream inspection.
- Added `select_best_outputs()`, which returns tensors with each batch row
  replaced by that row's own best observed sequence and structure.
- Added batch-aware diagnostic fields such as `guidance_batch_size`,
  `guidance_batch_sample_scores`, `guidance_global_best_sample_index`,
  `guidance_global_best_step`, and `guidance_global_best_score`.

`disco/model/cogen_inference/cogen_inference_loop_strategies.py`

- Replaced the previous global best-output finalization path with the
  controller's sample-aware `select_best_outputs()` method.
- The loop still performs the standard `FeatureDictUpdater` call when selected
  best sequences differ from the current final sequences.

`tests/test_guidance.py`

- Added `test_controller_tracks_and_selects_best_step_per_batch_sample`.
- The test verifies that two batch samples can keep different best steps and
  that final output selection does not copy the global best candidate into all
  batch rows.

`docs/DISCO_MOTIF_AI_README.md`

- Updated Save-Best status, architecture notes, test coverage, and pending
  work.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Updated implemented-feature status and quick-test list.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  disco/model/cogen_inference/cogen_inference_loop_strategies.py \
  tests/test_guidance.py
```

Executed all 59 direct CPU test functions; every test printed `ok`.

### Known limitations after this step

- This makes save-best selection batch-safe inside the controller and loop
  finalization. It does not add a large DISCO inference run proving biological
  quality; that remains part of the later PG-vs-DISCO comparison study.

## 2026-06-04 Run reproducibility bundle

### Decisions

- Add run-level reproducibility as a separate post-processing module, not as
  more logic inside DISCO inference.
- Keep the bundle compact: record file paths, sizes, hashes, score summaries,
  diagnostics metadata, commands, manifests, and environment metadata instead
  of embedding large structures or trajectory tensors.
- Make validation explicit. A bundle can be created from partial artifacts, but
  `--require-plan`, `--require-summary`, and `--require-diagnostics` turn missing
  evidence into structured issues.

### Files changed

`disco/model/cogen_inference/guidance/run_bundle.py`

- Added `RunBundleConfig` and `RunBundleIssue`.
- Added file metadata and bounded SHA256 hashing.
- Added summary-CSV aggregation for score columns, provenance counts,
  evaluation-status counts, and confidence-match counts.
- Added diagnostics aggregation from `diagnostics/*_run_metadata.json`.
- Added sequence/structure file inventory summaries.
- Added optional git and Python/platform metadata.
- Added `create_run_bundle()`, `write_run_bundle()`, and
  `validate_run_bundle()`.

`runner/guidance_bundle_run.py`

- Added CLI for creating `run_bundle.json`.
- Supports `--run-dir`, `--dump-dir`, `--summary-csv`, manifest overrides,
  `--issues-csv`, `--repo-dir`, identity fields, bounded hash size, `--no-git`,
  and required artifact gates.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported run-bundle config, issue, create, write, and validate functions.

`tests/test_guidance.py`

- Added `test_run_bundle_collects_summary_diagnostics_and_hashes`.
- Added `test_run_bundle_validation_reports_missing_required_artifacts`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table entry, file-map entries, architecture description, CLI
  example, and test coverage note.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and CLI
  usage example.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_bundle_run.py \
  tests/test_guidance.py
```

Executed all 61 direct CPU test functions; every test passed.

Executed a CLI smoke with synthetic run artifacts:

```text
Run bundle pass: 0 error(s), 0 warning(s)
{"diagnostics_metadata_records": 1, "summary_rows": 1}
cli smoke ok
```

### Known limitations after this step

- The bundle validates artifact presence and provenance metadata; it does not
  independently recompute motif scores, pLDDT, or structure metrics.
- Large files above the configured hash threshold are listed but not hashed by
  default.

## 2026-06-04 PG-vs-DISCO fairness audit

### Decisions

- Add a machine-readable fairness audit before expensive comparison runs.
- Keep it separate from the planner and completed-result validator: the planner
  writes the matrix, the fairness audit checks whether the matrix is balanced,
  and the result validator checks completed summaries.
- Treat DISCO `sample_diffusion.N_step` and ProteinGenerator `--T` as explicit
  budget knobs, not as mathematically identical diffusion steps.
- Keep `n_cycle=4` as the default reference and warn when plans deviate from it
  without blocking alternative justified setups.

### Files changed

`disco/model/cogen_inference/guidance/fairness.py`

- Added `FairnessIssue`.
- Added `audit_fairness_rows()` and `audit_fairness_csv()`.
- Checks required systems per `(mode, variant)` group.
- Checks replicate-count balance and seed-set balance across systems.
- Checks matched `n_cycle`, sequence length, motif list, and
  `designs_per_replicate`.
- Checks positive DISCO `disco_n_step` and ProteinGenerator `pg_T` parameters.
- Writes JSON/CSV reports with per-group budget summaries and issue details.

`runner/guidance_audit_fairness.py`

- Added CLI for auditing `planned_runs.csv`.
- Supports `--required-systems`, `--recommended-n-cycle`, and `--allow-warnings`.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported the fairness audit API.

`tests/test_guidance.py`

- Added `test_fairness_audit_accepts_balanced_standard_plan`.
- Added `test_fairness_audit_rejects_unmatched_group_parameters`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table entry, file-map entries, architecture notes, CLI example,
  and test coverage notes.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and CLI
  usage example.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/guidance_audit_fairness.py \
  tests/test_guidance.py
```

Executed all 63 direct CPU test functions; every test passed.

Executed a CLI smoke:

```text
Wrote 4 planned runs to .../plan
Fairness audit pass: 0 error(s), 0 warning(s)
fairness cli smoke ok
```

### Known limitations after this step

- The audit checks planned comparability, not biological outcome quality.
- It cannot prove DISCO `N_step` and ProteinGenerator `T` are equivalent; it
  only ensures both budget knobs are present, positive, and documented.

## 2026-06-04 Optional coordinate trajectory tensor dump

### Decisions

- Keep coordinate trajectory persistence disabled by default because the tensors
  can be large.
- Implement the feature as a separate guidance utility plus a small
  `runner/inference.py` hook. No sampler/model logic is changed.
- Store tensors as `.pt` files and write a compact JSON manifest with shapes,
  dtype, paths, noise levels, and truncation status.

### Files changed

`disco/model/cogen_inference/guidance/trajectory_dump.py`

- Added `TrajectoryTensorSpec`.
- Added `write_coordinate_trajectory_outputs()`.
- Saves `x1_denoised`, `xt_trajectory`, and optional final `coordinate`
  tensors.
- Supports storage dtype selection (`float32`, `float16`, `bfloat16`) and
  optional first-`max_steps` truncation.
- Writes `*_trajectory_manifest.json`.

`configs/cogen_guidance/default.yaml`

- Added disabled-by-default `trajectory_dump` config:
  `enabled`, `save_x1_denoised`, `save_xt_trajectory`,
  `save_final_coordinate`, `dtype`, and `max_steps`.

`runner/inference.py`

- Imports `write_coordinate_trajectory_outputs`.
- When `cogen_guidance.trajectory_dump.enabled=true`, writes trajectory tensors
  under `dump_dir/trajectories/` and adds manifest/path metadata to the local
  run record.

`disco/model/cogen_inference/guidance/run_bundle.py`

- Adds `dump_dir/trajectories` to the reproducibility bundle file inventory.

`disco/model/cogen_inference/guidance/__init__.py`

- Exports `TrajectoryTensorSpec` and `write_coordinate_trajectory_outputs`.

`tests/test_guidance.py`

- Added `test_write_coordinate_trajectory_outputs_saves_tensors_and_manifest`.
- Extended the run-bundle test to include trajectory files.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table entry, file-map entry, architecture note, CLI example,
  implemented coverage, and pending-work update.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entry, implemented-feature note, quick-test entry, and CLI
  usage example.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance \
  runner/inference.py \
  tests/test_guidance.py
```

Executed all 64 direct CPU test functions; every test passed.

### Known limitations after this step

- This stores tensors, not rendered PDB/CIF structures for every step.
- It has not been exercised in a real DISCO inference run in this session.

## 2026-06-04 PG-to-DISCO implementation parity audit

### Decisions

- Add a machine-readable implementation coverage map instead of relying only on
  the manual status table in the AI README.
- Keep the audit separate from scientific validation. It verifies that DISCO
  port files and named CPU tests exist for each mapped ProteinGenerator feature;
  outcome quality remains the responsibility of the quality, fairness, and
  comparison validators.
- Record partial and not-ported items explicitly so the project does not imply
  false completion. Current open/partial areas include DISCO confidence versus
  PG in-loop `pred_lddt`, autonomous heavy-run execution for evolution,
  PG-style contig scaffolding, cleavage fold-switch sampling, ligand-aware
  guidance, and external reprediction/Rosetta gates.

### Files changed

`disco/model/cogen_inference/guidance/parity.py`

- Added `ParityItem`.
- Added `default_parity_items()` with ProteinGenerator feature records grouped
  by standard motif, alternating, immunogen, evolution, logging, evaluation,
  comparison, structure, cleavage, ligand, and non-motif categories.
- Added `audit_parity(...)`, which checks DISCO evidence paths, named tests,
  and optional ProteinGenerator source handles.
- Added JSON and Markdown report writing helpers.

`runner/guidance_audit_parity.py`

- Added CLI wrapper around `audit_parity(...)`.
- Writes `parity_report.json` and `parity_report.md`.
- Supports `--require-all-ported` for a final strict completion gate.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported parity audit helpers.

`tests/test_guidance.py`

- Added `test_parity_audit_reports_core_ported_and_open_features`.
- Added `test_write_parity_report_outputs_json_and_markdown`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table and file-map entries.
- Added a parity-audit CLI section and current local count summary.
- Added parity coverage and explicit remaining gaps to the testing/pending-work
  sections.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and
  CLI usage for `runner/guidance_audit_parity.py`.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance/parity.py \
  runner/guidance_audit_parity.py \
  tests/test_guidance.py
```

Executed parity CLI smoke:

```bash
.venv/bin/python runner/guidance_audit_parity.py \
  --disco-root . \
  --pg-root ../ProteinGenerator_MOTIFS \
  --out-report-json /tmp/disco_parity_audit/parity.json \
  --out-md /tmp/disco_parity_audit/parity.md
```

Result:

```text
Parity audit pass: 24 ported, 3 partial, 4 not ported, 1 not applicable
```

Executed all 66 direct CPU test functions; every test passed.

### Known limitations after this step

- The parity audit is not a biological benchmark and does not execute DISCO or
  ProteinGenerator.
- ProteinGenerator handle checks are text-presence checks against likely source
  files; they are useful for traceability but not a substitute for source-code
  review.

## 2026-06-04 DISCO second-pass reprediction planner

### Decisions

- Port the planning layer of ProteinGenerator Immunogen `--immunogen_repredict`
  to DISCO. The DISCO port writes fixed-sequence input JSONs and exact commands
  for a clean second pass, but does not launch expensive inference
  automatically.
- Keep this separate from the guidance controller and sampler. Second-pass
  reprediction is a post-run evaluation workflow, not a per-step motif objective.
- Disable `cogen_guidance` in generated inference commands. The purpose is to
  re-evaluate the final hard sequence, not to keep optimizing motifs.
- Preserve the same-artifact discipline by allowing filters for fixed-fragment
  preservation and accepted sequence/structure/confidence provenance before a
  design is selected for reprediction.

### Files changed

`disco/model/cogen_inference/guidance/reprediction.py`

- Added `RepredictionPlanConfig` and `RepredictionRun`.
- Added CSV row loading, concrete-sequence validation, fixed-sequence DISCO
  input JSON writing, command construction, filtering, ranking, and plan
  manifest generation.
- Writes `planned_repredictions.csv`, `reprediction_plan_manifest.json`, and
  per-design `input.json`, `command.txt`, `analysis_command.txt`, and
  `run_manifest.json`.

`runner/guidance_plan_reprediction.py`

- Added CLI wrapper around `write_reprediction_plan(...)`.
- Supports motif lists, seeds, N-cycle, N-step, fixed fragments, T-cell motifs,
  cleavage motifs, quality/provenance filters, ranking, and `--max-designs`.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported reprediction planning helpers.

`disco/model/cogen_inference/guidance/parity.py`

- Added `immunogen_second_pass_reprediction_plan` as a partial port with
  explicit evidence files and tests.

`tests/test_guidance.py`

- Added `test_reprediction_plan_writes_fixed_sequence_input_and_commands`.
- Added `test_reprediction_plan_filters_and_ranks_designs`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table/file-map entries, a Second-pass reprediction planning
  section, updated parity counts, and updated pending-work wording.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and a
  concise CLI example.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance/reprediction.py \
  disco/model/cogen_inference/guidance/parity.py \
  runner/guidance_plan_reprediction.py \
  tests/test_guidance.py
```

Executed all 68 direct CPU test functions; every test passed.

Executed a synthetic CLI smoke:

```bash
.venv/bin/python runner/guidance_plan_reprediction.py \
  --analysis-csv /tmp/disco_reprediction_smoke/analysis.csv \
  --out /tmp/disco_reprediction_smoke/plan \
  --motifs GHK,KTT \
  --disco-root . \
  --python .venv/bin/python \
  --seeds 0 \
  --n-cycle 4 \
  --n-step 12 \
  --fixed-fragment 1:GHK:bcell \
  --tcell-motifs AAK \
  --cleavage-motifs TT \
  --require-fixed-preserved \
  --require-same-evaluation-artifact
```

Result:

```text
Reprediction plan: 1 planned, 0 skipped
```

Executed parity CLI after this addition:

```text
Parity audit pass: 24 ported, 4 partial, 4 not ported, 1 not applicable
```

### Known limitations after this step

- This is a command/input materializer, not an executor. Real second-pass
  DISCO outputs still require running the generated `command.txt` files.
- The generated second-pass analysis can require confidence files, but it cannot
  validate them until the second-pass run has actually completed.

## 2026-06-04 External quality metric merge and posthoc rank score

### Decisions

- Port the reusable comparison layer of ProteinGenerator's post-hoc
  `evaluate_generated_proteins.py` without assuming optional external
  dependencies are installed.
- Do not run ESM-2, ESMFold, or Rosetta from DISCO. The new runner imports
  externally produced CSV columns and makes missing metrics explicit.
- Reuse the documented ProteinGenerator rank-style score:
  `0.30 * confidence + 0.20 * ESM2 rank + 0.20 * RMSD rank +
  0.20 * Rosetta rank + 0.10 * motif_score`.
- Treat higher ESM-2 mean token probability as better, lower ESM-2 perplexity
  as better when probability is absent, lower RMSD as better, and lower Rosetta
  score per residue as better.

### Files changed

`disco/model/cogen_inference/guidance/external_quality.py`

- Added `ExternalQualityWeights`.
- Added external metric CSV merge by `design_id`.
- Added PG-style percentile-rank component scoring for ESM-2, RMSD, and Rosetta.
- Added `posthoc_quality_score`, `external_metric_status`, and
  `external_metric_missing` annotations.
- Added CSV and JSON report writing helpers.

`runner/guidance_merge_external_quality.py`

- Added CLI wrapper for merging external metrics into an existing DISCO or
  normalized PG/DISCO summary CSV.
- Supports custom component weights, key field override, report JSON, and top
  ranked CSV output.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported external-quality helpers.

`disco/model/cogen_inference/guidance/parity.py`

- Updated `external_reprediction_and_rosetta` from `not_ported` to `partial`.

`tests/test_guidance.py`

- Added `test_external_quality_score_uses_pg_rank_style_components`.
- Added `test_external_quality_outputs_merge_metrics_and_report_missing_values`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table/file-map entries, external metric merge usage, updated
  parity counts, implemented coverage, and pending-work wording.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and
  CLI usage for `runner/guidance_merge_external_quality.py`.

### Verification

Executed:

```bash
.venv/bin/python -m compileall -q \
  disco/model/cogen_inference/guidance/external_quality.py \
  disco/model/cogen_inference/guidance/parity.py \
  runner/guidance_merge_external_quality.py \
  tests/test_guidance.py
```

Executed a synthetic CLI smoke:

```bash
.venv/bin/python runner/guidance_merge_external_quality.py \
  --summary-csv /tmp/disco_external_quality_smoke/summary.csv \
  --external-metrics-csv /tmp/disco_external_quality_smoke/external.csv \
  --out-csv /tmp/disco_external_quality_smoke/augmented.csv \
  --report-json /tmp/disco_external_quality_smoke/report.json \
  --top-csv /tmp/disco_external_quality_smoke/top.csv
```

Result:

```text
External quality merge: 2 rows, best=A (0.890000)
```

Executed all 70 direct CPU test functions; every test passed.

Executed parity CLI after this addition:

```text
Parity audit pass: 24 ported, 5 partial, 3 not ported, 1 not applicable
```

### Known limitations after this step

- This is an import/ranking layer, not an external-tool executor.
- Rosetta remains a licensed external dependency; this module only consumes
  Rosetta scores when a separate process has already produced them.

## 2026-06-04 Sequence-level scaffold input planner

### Decisions

- Port only the scaffold subset that DISCO's current inference JSON can express:
  fixed amino-acid letters inside `proteinChain.sequence` and `-` for designable
  positions.
- Do not pretend that ProteinGenerator/RFdiffusion-style structural contigs are
  supported. PG `--pdb`, chain-residue `--contigs`, `--inpaint_seq`, and
  `--inpaint_str` are recorded as provenance and explicit unsupported issues in
  the manifest.
- Keep the implementation outside the DISCO sampler and loop body. This is input
  materialization and provenance auditing, not a per-step guidance objective.
- Reuse the existing fixed-fragment coordinate contract:
  `start_1based:sequence[:label]` and PG/DISCO fixed-fragments JSON.

### Files changed

`disco/model/cogen_inference/guidance/scaffold_plan.py`

- Added `ScaffoldFixedRange`, `ScaffoldIssue`, and `ScaffoldPlanResult`.
- Added `parse_fixed_range(...)` for `start_1based:end_1based[:label]` seed
  ranges.
- Added `fixed_fragments_from_json(...)` for the PG/DISCO immunogen
  fixed-fragments JSON schema.
- Added `sequence_from_scaffold_constraints(...)`, which builds a DISCO
  scaffold sequence by combining literal fixed fragments, optional seed-derived
  fixed ranges, and `-` designable positions.
- Added `audit_pg_scaffold_arguments(...)`, which marks seed PDBs, structural
  contigs, and inpaint masks as unsupported instead of silently dropping them.
- Added `write_scaffold_plan(...)`, which writes both the DISCO `input.json` and
  a manifest with fixed-position counts, issues, and not-ported semantics.

`runner/guidance_plan_scaffold.py`

- Added a small CLI wrapper around `write_scaffold_plan(...)`.
- Supports `--fixed-fragment`, `--fixed-fragments-json`, `--fixed-range`,
  `--seed-sequence`, PG scaffold provenance arguments, and explicit output paths.
- Does not execute DISCO inference.

`disco/model/cogen_inference/guidance/__init__.py`

- Exported scaffold planning helpers.

`disco/model/cogen_inference/guidance/parity.py`

- Updated `structural_seed_contig_scaffolding` from `not_ported` to `partial`.
- Evidence is now the sequence-level scaffold planner and runner.
- Notes still state that true PDB coordinate contigs and independent structural
  masks are unsupported.

`tests/test_guidance.py`

- Added `test_scaffold_plan_writes_fixed_fragment_input_json`.
- Added `test_scaffold_plan_fixed_range_uses_seed_sequence`.
- Added `test_scaffold_plan_reports_structural_contig_unsupported`.

`docs/DISCO_MOTIF_AI_README.md`

- Added status-table/file-map entries.
- Added a sequence-level scaffold input planning section.
- Updated parity counts, testing coverage, and pending-work wording.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added file-map entries, implemented-feature note, quick-test entries, and CLI
  usage for `runner/guidance_plan_scaffold.py`.

### Verification

Executed:

```bash
.venv/bin/python -m compileall \
  disco/model/cogen_inference/guidance/scaffold_plan.py \
  runner/guidance_plan_scaffold.py \
  tests/test_guidance.py
```

Executed `git diff --check` on the scaffold-related files; no whitespace errors.

Executed all 73 direct CPU test functions; every test passed.

Executed scaffold CLI smoke:

```bash
.venv/bin/python runner/guidance_plan_scaffold.py \
  --length 12 \
  --name scaffold_smoke \
  --fixed-fragment 3:GHK:motif \
  --seed-sequence AAAACCCCGGGG \
  --fixed-range 9:12:tail \
  --pg-seed-pdb /tmp/seed.pdb \
  --pg-contig A1-12,0 \
  --out-input-json /tmp/disco_scaffold_smoke/input.json \
  --out-manifest-json /tmp/disco_scaffold_smoke/manifest.json
```

Result:

```text
Scaffold plan: partial_sequence_scaffold_with_unsupported_pg_structure_semantics
Sequence: --GHK---GGGG
Issues: 2
- unsupported pg_seed_pdb_structure_template_not_ported: /private/tmp/seed.pdb
- unsupported pg_contig_structure_mask_not_ported: A1-12,0
```

Executed parity CLI after this addition:

```text
Parity audit pass: 24 ported, 6 partial, 2 not ported, 1 not applicable
```

### Known limitations after this step

- This is not fixed-backbone design. It does not hold PDB coordinates.
- This is not PG contig scaffolding. Chain-residue contigs are reported, not
  converted into DISCO coordinate masks.
- It does not choose motif placements automatically; it materializes explicit
  fixed fragments/ranges into a DISCO input JSON.

## 2026-06-04 PG-vs-DISCO local smoke planning and execution attempt

### Decisions

- Materialize a tiny local smoke matrix before attempting expensive scientific
  runs. This checks command generation, validation, fairness auditing, and local
  runtime readiness.
- Keep the Mac smoke intentionally smaller than the later reference benchmark:
  N-cycle `1`, DISCO `N_step=2`, PG `T=2`, one seed, length `24`.
- Also materialize an N-cycle `4` reference plan for later use, but do not run it
  while PG-Immunogen seed inputs and runtime environments are missing.
- Add a configurable Fabric accelerator/strategy path to DISCO inference so Mac
  CPU smoke attempts can bypass MPS+DDP incompatibility.

### Files changed

`configs/fabric/default.yaml`

- Added `fabric.accelerator: auto`.
- Added `fabric.strategy: ddp`.

`runner/inference.py`

- Reads `configs.fabric.strategy`.
- Uses `DDPStrategy(find_unused_parameters=False)` only when
  `fabric.strategy=ddp`.
- Passes `fabric.accelerator` into Lightning Fabric.
- Calls `torch.cuda.set_device(...)` only when the resolved Fabric device is
  CUDA. This allows CPU smoke attempts with:

```text
fabric.accelerator=cpu fabric.strategy=auto
```

`outputs/pg_disco_compare_local_smoke_2026_06_04/`

- Added local smoke plan artifacts:
  - `planned_runs.csv`
  - per-run `command.txt`, `analysis_command.txt`, `run_manifest.json`
  - `planning_validation.json`
  - `fairness_report.json`
  - `EXECUTION_REPORT.md`
  - run logs and exit-code files for attempted runs

`outputs/pg_disco_compare_reference_n4_2026_06_04/`

- Added N=4 reference plan artifacts for later full comparison work.
- PG Immunogen rows are marked `requires_pg_immunogen_seed_inputs` because no
  real PG seed PDB, seed sequence, and fixed-fragments JSON were provided.

`docs/DISCO_MOTIF_AI_README.md`

- Added Mac-local comparison smoke status and current execution blockers.
- Added local PG runtime and DISCO checkpoint availability to pending work.

`docs/DISCO_MOTIF_DEVELOPER_README.md`

- Added CPU smoke overrides and link to the current smoke folder/report.

### Verification

Executed local smoke plan materialization:

```bash
.venv/bin/python runner/guidance_plan_comparison.py \
  --out outputs/pg_disco_compare_local_smoke_2026_06_04 \
  --systems DISCO,ProteinGenerator \
  --modes standard,alternating \
  --variant-set core \
  --seeds 0 \
  --n-cycle 1 \
  --disco-n-step 2 \
  --pg-standard-T 2 \
  --pg-short-T 2 \
  --designs-per-replicate 1 \
  --standard-length 24 \
  --alternating-length 24
```

Result:

```text
Wrote 4 planned runs
Statuses: planned
```

Executed planning validation:

```text
Validation pass: 0 error(s), 4 warning(s), 0/4 completed summaries
```

The four warnings are expected missing-summary warnings before model runs
complete.

Executed fairness audit:

```text
Fairness audit pass: 0 error(s), 0 warning(s)
```

Executed N=4 reference plan materialization:

```text
Wrote 18 planned runs
Statuses: planned, requires_pg_immunogen_seed_inputs
```

Reference-plan validation currently fails with the expected missing PG-Immunogen
seed-input errors and missing-summary warnings.

Executed PG Standard smoke command:

```text
Exit code: 1
Failure: ModuleNotFoundError: No module named 'opt_einsum'
```

Executed DISCO Standard smoke command once with default Mac settings:

```text
Exit code: 1
Failure: Lightning MPS accelerator is incompatible with forced DDP strategy
```

Executed DISCO Standard smoke command again with CPU overrides:

```text
fabric.accelerator=cpu fabric.strategy=auto use_deepspeed_evo_attention=false
structure_encoder.args.device=cpu dtype=float32
```

Result:

```text
Exit code: 124
Progress: Fabric initialized on CPU and DPLM assets were loaded/cached.
Stop reason: DISCO checkpoint access through Hugging Face/Xet stalled with no
log progress; process was terminated to avoid a hanging local job.
```

Executed:

```bash
.venv/bin/python -m compileall configs runner/inference.py
git diff --check -- configs/fabric/default.yaml runner/inference.py
```

Both passed.

### 2026-06-04 Mac-local PG-vs-DISCO smoke completion

Resolved the local smoke blockers:

- ProteinGenerator runs through the local Conda environment
  `/opt/anaconda3/envs/proteingenerator-mac/bin/python`.
- PG planned commands can receive the local checkpoint path through
  `--pg-checkpoint`.
- PG Alternating analysis commands now include
  `PYTHONPATH=<ProteinGenerator_MOTIFS>/code`.
- DISCO checkpoint `DISCO.pt` was downloaded into the Hugging Face cache with
  `HF_HUB_DISABLE_XET=1`.
- DISCO CPU inference now respects the active model/input device in the
  inference noise schedule and pairformer feature moves.
- DISCO sequence analysis now accepts sequence text that contains `-` mask
  characters and preserves them for length validation.

Completed local smoke folder:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04
```

Final artifacts:

```text
Validation pass: 0 error(s), 0 warning(s), 4/4 completed summaries
Fairness audit pass: 0 error(s), 4 warning(s)
```

The fairness warnings are expected for a deliberately tiny CPU smoke
(`N_cycle=1`, two diffusion steps, one seed). They are not scientific-quality
comparison settings.

Smoke summary:

```text
DISCO standard: 0 motifs, sequence contains unresolved mask positions
DISCO alternating: 0 motifs, sequence contains unresolved mask positions
ProteinGenerator standard: 0 motifs, pLDDT ~0.503
ProteinGenerator alternating: 2 unique motifs / 3 motif hits, pLDDT ~0.521
```

On this Mac CPU setup DISCO writes the model outputs and logs `succeeded`, but
the Python process can hang during shutdown with a multiprocessing semaphore
warning. The smoke wrapper terminates that post-output shutdown hang and writes
`shutdown_note.txt` in the affected run directories.
