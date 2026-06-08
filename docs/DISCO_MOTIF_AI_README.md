# DISCO Motif Guidance AI Reference

This document is the machine-facing entry point for the Cambrium DISCO motif
port. It explains how the ProteinGenerator motif architecture maps onto DISCO,
what has been implemented, what remains intentionally separate, and where future
work should attach.

## 1. Purpose

The goal is to make DISCO comparable to the Cambrium ProteinGenerator MOTIFS
fork for sequence-motif design experiments while keeping upstream DISCO files
as unchanged as possible.

The first implemented layer is a generic co-generation guidance controller:

```text
DISCO sequence decoder logits
  -> optional Cambrium GuidanceController
  -> existing DISCO PathPlanningSamplingStrategy
  -> normal DISCO feature update
  -> optional hard motif scoring and save-best tracking
```

This deliberately does not replace DISCO's sampler. It modifies the probability
landscape that the existing sampler consumes.

## 2. ProteinGenerator Concepts Being Ported

ProteinGenerator MOTIFS has three important layers:

1. Soft guidance over continuous sequence probabilities/logits.
2. Local rescue over plausible top-k residue choices.
3. Hard exact-sequence scoring for logging and save-best selection.

The DISCO port mirrors that split.

| ProteinGenerator component | DISCO port location | Status |
| --- | --- | --- |
| `SequenceMotifsBias` | `guidance/motif.py::MotifLogitGuidance` gradient mode | Implemented for logits |
| `MotifLocalSearchRescue` | `guidance/motif.py` local rescue mode | Implemented for logits |
| `HardMotifScorer` | `guidance/scoring.py::HardMotifScorer` | Implemented |
| `save_best_combined` bookkeeping | `guidance/controller.py` + loop finalization | Implemented, including per-sample batch tracking |
| Alternating motif hard metrics | `guidance/alternating.py` | Implemented |
| `AlternatingMotifsBias` slot potential | `guidance/alternating.py::AlternatingLogitGuidance` | Implemented for logits |
| `soft_transition_alternating_motifs_bias` | `guidance/alternating.py::SoftTransitionAlternatingLogitGuidance` | Implemented for logits |
| Immunogen fixed-fragment scoring | `guidance/immunogen.py` | Implemented |
| `immunogen_mutable_motifs_bias` | `guidance/immunogen.py::MutableWindowMotifLogitGuidance` | Implemented for logits |
| Immunogen spacing-aware score | `guidance/immunogen.py::spacing_score` | Implemented |
| Immunogen category validation | `guidance/immunogen_validation.py` + `runner/guidance_validate_immunogen.py` | Implemented for B-cell/T-cell/cleavage gates |
| Immunogen second-pass reprediction planning | `guidance/reprediction.py` + `runner/guidance_plan_reprediction.py` | Partially implemented: fixed-sequence inputs and commands, operator-run inference |
| PG `--pdb`/`--contigs` fixed scaffold subset | `guidance/scaffold_plan.py` + `runner/guidance_plan_scaffold.py` | Partially implemented: sequence-level fixed fragments/ranges only; PDB coordinate contigs unsupported |
| Evolutionary outer loop | `guidance/evolution.py` + `runner/guidance_evolution.py` | Implemented as dry-run capable runner |
| Per-step guidance diagnostics | `guidance/diagnostics.py` + `runner/inference.py` | Implemented for JSONL/CSV/runtime metadata |
| Guidance trajectory analysis | `guidance/trajectory.py` + `runner/guidance_analyze_trajectory.py` | Implemented for per-step CSVs and best/final summaries |
| Coordinate trajectory tensor dump | `guidance/trajectory_dump.py` + `runner/inference.py` | Implemented as optional disabled-by-default `.pt` output |
| Run reproducibility bundle | `guidance/run_bundle.py` + `runner/guidance_bundle_run.py` | Implemented for post-run manifests and validation |
| Completed DISCO dump analysis | `guidance/disco_analysis.py` + `runner/guidance_analyze_disco.py` | Implemented for sequence/confidence/motif CSVs |
| PG-vs-DISCO experiment planning | `guidance/experiment_plan.py` + `runner/guidance_plan_comparison.py` | Implemented as dry-run command/manifests planner |
| PG-vs-DISCO fairness audit | `guidance/fairness.py` + `runner/guidance_audit_fairness.py` | Implemented for planned matrix budget checks |
| PG-vs-DISCO comparison validation | `guidance/comparison_validation.py` + `runner/guidance_validate_comparison.py` | Implemented for plan/result gates |
| Completed-design quality gates | `guidance/quality_validation.py` + `runner/guidance_validate_quality.py` | Implemented for threshold validation |
| External ESM/RMSD/Rosetta metric merge | `guidance/external_quality.py` + `runner/guidance_merge_external_quality.py` | Partially implemented: imports external metrics and computes PG-style posthoc rank score |
| PG-vs-DISCO comparison reporting | `guidance/comparison_report.py` + `runner/guidance_report_comparison.py` | Implemented for Markdown summaries |
| PG-to-DISCO implementation parity audit | `guidance/parity.py` + `runner/guidance_audit_parity.py` | Implemented for feature-map evidence checks |

## 3. DISCO Integration Points

The main model flow is:

```text
runner/inference.py
  -> DISCO.forward()
  -> DISCO.sample_diffusion()
  -> sample_diffusion_cogen()
  -> BasicInferenceLoopStrategy.run_loop()
  -> InferenceLoopImpl.execute()
```

The sequence decision point inside `InferenceLoopImpl.execute()` is:

```python
seq_decoder_logits = self.model.get_inference_seq_decoder_logits(...)

seq_decoder_logits = self.guidance_controller.modify_logits(...)

xt_seq_new = self.sequence_sampling_strategy.step(...)
```

After the sequence update, the controller observes the step:

```python
step_diag.update(self.guidance_controller.observe_step(...))
```

This gives the controller access to the exact sequence state, current structure
tensor, decoder logits, designable-position mask, and step index.

## 4. New Files

```text
disco/model/cogen_inference/guidance/
  __init__.py
  alternating.py
  comparison.py
  comparison_report.py
  comparison_validation.py
  controller.py
  disco_analysis.py
  diagnostics.py
  evaluation_provenance.py
  artifact_provenance.py
  experiment_plan.py
  evolution.py
  external_quality.py
  fairness.py
  immunogen.py
  immunogen_validation.py
  motif.py
  parity.py
  quality_validation.py
  reprediction.py
  run_bundle.py
  scaffold_plan.py
  scoring.py
  trajectory.py
  trajectory_dump.py

configs/cogen_guidance/default.yaml
runner/guidance_audit_fairness.py
runner/guidance_audit_parity.py
runner/guidance_analyze_disco.py
runner/guidance_analyze_trajectory.py
runner/guidance_bundle_run.py
runner/guidance_compare.py
runner/guidance_evolution.py
runner/guidance_merge_external_quality.py
runner/guidance_plan_comparison.py
runner/guidance_plan_reprediction.py
runner/guidance_plan_scaffold.py
runner/guidance_report_comparison.py
runner/guidance_validate_comparison.py
runner/guidance_validate_immunogen.py
runner/guidance_validate_quality.py
tests/test_guidance.py
```

Comparison smoke artifacts may also be created under `outputs/`. The first
Mac-local PG-vs-DISCO smoke plan is:

```text
outputs/pg_disco_compare_local_smoke_2026_06_04/
```

It contains `planned_runs.csv`, validation/fairness reports, run command files,
execution logs, and `EXECUTION_REPORT.md`.

The completed Mac-local smoke folders are:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04/
outputs/pg_disco_compare_local_immunogen_smoke_2026_06_04/
```

The first folder contains Standard and Alternating runs for both systems. The
second folder contains the separate Immunogen smoke. Both folders contain
`COMPARISON_REPORT.md`, `EXECUTION_REPORT.md`, `normalized_designs.csv`,
`comparison_summary.csv`, regular validation outputs, and strict confidence
validation outputs.

`controller.py` owns orchestration: optional logit guidance, hard scoring,
combined score calculation, and best-state tracking.

`motif.py` owns motif-specific logit edits. It currently supports:

- gradient-style motif objective over a detached logits clone,
- local top-k rescue,
- designable-position masking through `can_change_idx`.

`scoring.py` owns hard exact-sequence motif metrics. It intentionally works on
strings/tokens and does not depend on model internals.

`alternating.py` owns both the hard exact-sequence metrics for two-list
alternating motif designs and the slot-based soft logit guidance adapted from
ProteinGenerator's `AlternatingMotifsBias`. The hard scorer can contribute to
the controller's combined save-best score; the logit guidance modifies
`seq_decoder_logits` before DISCO's existing sampler runs.

`immunogen.py` owns the immunogen layer: fixed fragment specs, fixed-position
masks, exact fixed-fragment preservation scoring, mutable-window motif logit
guidance, and spacing-aware motif ranking metrics.

`scaffold_plan.py` owns the sequence-level subset of ProteinGenerator
`--pdb`/`--contigs`/`--inpaint_seq`/`--inpaint_str` workflows that can be
represented in DISCO's current input JSON path. It writes
`proteinChain.sequence` strings where fixed letters stay literal and all other
positions are `-`. It also records unsupported PG structural semantics, such as
PDB coordinate template contigs and independent structure/sequence inpaint masks,
in a manifest instead of silently pretending they were ported.

`evolution.py` owns the outer-loop search logic ported from ProteinGenerator
Approach 3: genomes, mutation, crossover, fitness scoring, motif templates,
DISCO input-JSON materialization, DISCO Hydra command construction, and DISCO
sequence-output parsing.

`comparison.py` owns the PG-vs-DISCO comparison layer. It normalizes
ProteinGenerator motif/immunogen/alternating summary rows and DISCO evolution
or completed-dump rows into one schema, then aggregates by system, mode, and
variant. It keeps score provenance visible through fields such as
`confidence_metric`, `confidence_match_status`, `confidence_path`,
`sequence_path`, and `score_provenance_status`. It also fixes the important
column mismatch where ProteinGenerator Alternating summaries use
`num_unique_motifs` and `num_motif_hits` instead of the standard
`num_unique_motifs_present` and `num_motif_occurrences_total`. DISCO
evolution summaries without `present_motifs/missing_motifs` are also handled by
reading per-motif count columns such as `GHK_count`.

The normalized `comparison_score` is intentionally simple:

```text
comparison_score = 0.5 * confidence + 0.5 * motif_score
```

If an Immunogen row proves fixed fragments were not preserved, the comparison
score is set to `0.0` because that design is invalid for the mode.

`comparison_report.py` owns the Markdown reporting layer. It consumes existing
normalized/summary CSVs and an optional validation JSON report, then writes a
compact report with validation status, aggregate rows, top designs by
comparison score, provenance counts, and interpretation guardrails.

`disco_analysis.py` owns post-run analysis for completed DISCO output folders.
It parses `sequences/*.txt`, pairs each sequence with the matching
`summary_confidence_sample_*.json` where possible, and writes motif,
immunogen, alternating, and confidence metrics into one CSV-ready row per
finished sequence. This pairing is intentionally sample-aware so motif metrics
and confidence values do not come from different DISCO outputs. It also records
matched structure paths and ProteinGenerator-style quality metrics:
`longest_homopolymer`, `aa_entropy_norm`, optional seed-sequence identity,
optional changed positions versus a seed sequence, and optional PDB CA-RMSD to
a seed structure.

`evaluation_provenance.py` owns the DISCO-side equivalent of the
ProteinGenerator `--immunogen_repredict` evaluation invariant. PG's repredict
flag runs one final clean forward pass so pLDDT and coordinates match the final
sequence. DISCO stores sequence, confidence, structure, and diagnostics in
separate files, so the ported invariant is an explicit evaluation bundle:
`evaluation_status=same_sample_complete` means sequence, confidence JSON,
structure file, and diagnostics all resolve to the same output sample;
`same_sample_no_diagnostics` means sequence/confidence/structure match exactly
but diagnostics were absent.

`diagnostics.py` owns per-step guidance trace serialization. During
`runner/inference.py`, DISCO writes `diagnostics/*_guidance_diagnostics.jsonl`,
`diagnostics/*_guidance_diagnostics.csv`, and
`diagnostics/*_run_metadata.json` when diagnostic records are present. These
files contain step-wise sequence/score/noise diagnostics plus runtime seconds.
This is a lightweight sequence/score trajectory.

`trajectory_dump.py` owns optional full coordinate tensor persistence. When
`cogen_guidance.trajectory_dump.enabled=true`, `runner/inference.py` writes
`trajectories/*_x1_denoised.pt`, `trajectories/*_xt_trajectory.pt`, optional
final coordinate tensors, and `*_trajectory_manifest.json`. This remains
disabled by default because the tensors can be large.

`artifact_provenance.py` owns final/intermediate artifact binding. For each
saved DISCO sample it writes `provenance/*_artifact_provenance.json`, linking
the final sequence, final structure, diagnostics, trajectory manifest, hashes,
and any same-sample confidence artifacts that actually exist. It also writes
`provenance/*_intermediate_states.csv` with one row per saved structure-state
kind per timestep. Intermediate rows deliberately use
`structure_confidence_status=not_computed_for_intermediate_state`; they must not
be compared as pLDDT/confidence-scored structures until a separate per-state
scoring job writes matching artifacts.

`trajectory.py` owns offline analysis of those diagnostic JSONL files. It
normalizes step records into a trajectory CSV with score deltas, motif-count
deltas, sequence changes from the previous step, newly gained/lost motifs,
best-step markers, final-step markers, and sampler uncertainty fields. It also
writes a JSON summary that states whether the best scored intermediate beat the
final sample. This is the layer to use when checking whether save-best was
scientifically necessary for a run.

`run_bundle.py` owns post-run reproducibility manifests. It combines the planned
run manifest, command files, analysis manifest, summary CSV statistics,
diagnostics metadata, sequence/structure file inventories, SHA256 hashes for
bounded-size artifacts, runtime aggregates, and optional git/environment
metadata into one `run_bundle.json`. This is the audit layer between a finished
single run and the later cross-system comparison CSVs.

`experiment_plan.py` owns the comparison-run matrix. It writes per-run
commands, manifests, DISCO input JSONs, summary targets, fairness notes, and a
`guidance_compare.py` scaffold for Standard, Immunogen, and Alternating runs
across DISCO and ProteinGenerator.

`fairness.py` owns the planned-matrix fairness audit. It reads
`planned_runs.csv` and checks that every `(mode, variant)` group has the
required systems, balanced replicate counts, matched seed sets, matched
`n_cycle`, length, motifs, and designs per replicate, plus explicit positive
DISCO `sample_diffusion.N_step` and ProteinGenerator `--T` budget knobs. The
audit does not pretend those budget knobs are mathematically equivalent; it
records that distinction in the JSON report.

`comparison_validation.py` owns comparison gating. It reads `planned_runs.csv`
and verifies required systems, balanced run counts, shared parameters, planner
artifacts, summary CSV existence, sequence lengths, confidence fields,
sample-aware DISCO confidence matching, optional DISCO diagnostics, and
Immunogen fixed-fragment preservation.

`quality_validation.py` owns completed-design gates. It validates already
generated summary or normalized CSV rows against thresholds such as confidence,
motif coverage, unique motif count, fixed-fragment preservation, homopolymer
length, entropy, CA-RMSD, motif spread, diagnostics presence, structure
presence, and exact confidence provenance. This mirrors ProteinGenerator's
`validate_immunogen_summary.py` pattern but works on both DISCO analysis CSVs
and normalized PG/DISCO comparison rows.

`immunogen_validation.py` owns Immunogen category gates. It validates completed
summary rows by category: protected B-cell/fixed fragments, T-cell motif
coverage, cleavage motif coverage, and optional same-artifact provenance. This
is intentionally separate from the generic quality validator because Immunogen
failure modes are categorical: a row can have acceptable global motif coverage
while still failing the T-cell or cleavage sub-objective.

## 5. Configuration

The default config is disabled and should not affect vanilla DISCO:

```yaml
cogen_guidance:
  enabled: false
```

A minimal enabled motif run would use overrides like:

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.enabled=true \
  'cogen_guidance.motifs=[GHK,KTTKS]' \
  cogen_guidance.logit_guidance.enabled=true \
  cogen_guidance.logit_guidance.gradient_scale=1.0
```

Do not treat that command as locally cheap: DISCO inference is GPU-oriented.

## 6. Guidance Semantics

### Gradient-style motif guidance

This is analogous to PG `SequenceMotifsBias`, but applied to DISCO decoder
logits rather than PG's `seq_out` tensor. The implementation:

1. clones logits with gradients enabled,
2. converts to probabilities with a motif temperature,
3. scans all motif windows,
4. maximizes an all-score plus many-score objective,
5. normalizes the logit gradient,
6. adds it back to the real logits.

This is differentiable with respect to the logits clone, not a full end-to-end
gradient through DISCO's diffusion model.

### Local rescue

This mirrors PG's pragmatic top-k rescue. It searches for motif windows whose
required residue letters are already plausible under the model probabilities.
It then boosts those residue logits. It does not directly set sequence tokens.

### Alternating logit guidance

This ports both ProteinGenerator alternating soft potentials:

- `AlternatingLogitGuidance` is the slot-based equivalent of
  `alternating_motifs_bias`.
- `SoftTransitionAlternatingLogitGuidance` is the slot-free equivalent of
  `soft_transition_alternating_motifs_bias`.

The slot-based DISCO implementation:

1. converts sequence decoder logits to amino-acid probabilities,
2. plans broad alternating slots across the sequence,
3. rewards list1 motifs in list1 slots and list2 motifs in list2 slots,
4. adds a coverage term so both motif groups remain represented,
5. adds the asymmetric list1-tail term from ProteinGenerator,
6. normalizes the resulting logit gradient and adds it before the sampler.

Enable it independently from the plain motif guidance:

```bash
'cogen_guidance.alternating.enabled=true' \
'cogen_guidance.alternating.motif_list1=[GP,KVK]' \
'cogen_guidance.alternating.motif_list2=[DR]' \
cogen_guidance.alternating.logit_guidance_enabled=true \
cogen_guidance.alternating.logit_gradient_scale=1.0
```

The transition-only comparison variant is enabled separately:

```bash
'cogen_guidance.alternating.enabled=true' \
'cogen_guidance.alternating.motif_list1=[GP,DK]' \
'cogen_guidance.alternating.motif_list2=[DR,VW]' \
cogen_guidance.alternating.logit_guidance_enabled=false \
cogen_guidance.alternating.transition_logit_guidance_enabled=true \
cogen_guidance.alternating.transition_logit_gradient_scale=1.0
```

It rewards ordered list1->list2 and list2->list1 soft motif-start pairs,
penalizes same-list ordered pairs, and includes the same start/end/list1-tail
stabilizers used in the ProteinGenerator research comparison preset.

### Immunogen mutable-window guidance

This ports ProteinGenerator's `immunogen_mutable_motifs_bias`. It is separate
from the plain motif guidance because immunogen motifs must be encouraged only
in regions that can actually change. The DISCO version:

1. reads configured fixed fragments,
2. blocks those positions in the local `can_change_idx` mask,
3. computes motif-window scores only for starts whose whole motif window is
   mutable,
4. optionally runs local rescue with the same fixed-position mask,
5. reports fixed-fragment preservation and optional motif-spacing metrics.

Example:

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

DISCO caveat: this does not launch a second model forward pass equivalent to
ProteinGenerator's `--immunogen_repredict`. Instead, completed-run analysis now
exposes `evaluation_status` fields that prove whether the saved sequence,
structure, confidence, and diagnostics are tied to the same DISCO sample.

### Evolutionary outer loop

This ports ProteinGenerator Approach 3 as a separate runner, not as a sampler
change. The runner can:

1. create conservative DISCO guidance genomes,
2. translate each genome into `runner/inference.py` Hydra overrides,
3. write per-individual `command.txt` and `batch_manifest.json`,
4. optionally launch DISCO inference,
5. analyze DISCO sequence files and confidence JSONs,
6. rank individuals by confidence plus hard motif hits,
7. keep elites and generate a next population,
8. materialize elite motif templates as DISCO input JSONs for later
   generations.

Template semantics deliberately bridge the ProteinGenerator and DISCO input
formats. The evolutionary genome stores PG-style templates with fixed amino
acid letters and `X` for designable positions. Before a templated DISCO run is
planned, `X` is converted to DISCO's protein mask character `-`; only
`proteinChain` records whose length matches the template are kept in the
generated per-individual `input_template.json`.

Dry-run example, safe for local CPU validation:

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

This writes planned commands and manifests only. Remove `--dry-run` only when a
real DISCO inference run is intended.

### Completed DISCO dump analysis

For a finished DISCO run, use the analyzer before comparison:

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

The analyzer does not launch inference. It writes a per-design CSV plus a
manifest and records whether each confidence value was an exact sample match,
a single-confidence fallback, or missing. If diagnostics metadata exists, the
analysis CSV also includes runtime seconds, diagnostic file paths, best
diagnostic step, best diagnostic score, and the number of diagnostic records.
With `--seed-sequence`, it adds seed identity and changed positions. With
`--seed-pdb`, it adds CA-RMSD when the matched DISCO structure is a PDB file;
CIF files keep their structure path but leave RMSD blank rather than inventing
a value.

### Second-pass reprediction planning

For Immunogen-style honesty checks, materialize fixed-sequence second-pass DISCO
runs from an existing analysis CSV:

```bash
python runner/guidance_plan_reprediction.py \
  --analysis-csv /path/to/disco_immunogen_analysis.csv \
  --out /path/to/reprediction_plan \
  --motifs GHK,KTTKS \
  --seeds 0,1,2 \
  --n-cycle 4 \
  --n-step 100 \
  --fixed-fragment 1:GHK:bcell \
  --tcell-motifs AAK,LLK \
  --cleavage-motifs TT \
  --require-fixed-preserved \
  --require-same-evaluation-artifact \
  --max-designs 10
```

This writes `planned_repredictions.csv`, `reprediction_plan_manifest.json`, and
one run folder per selected design. Each run folder contains a full-sequence
DISCO `input.json`, `command.txt`, `analysis_command.txt`, and
`run_manifest.json`. The inference command sets `cogen_guidance.enabled=false`;
the point is to re-evaluate the final hard sequence, not to continue motif
optimization. This is the DISCO planning equivalent of PG
`--immunogen_repredict`, but execution remains an explicit operator step.

### Sequence-level scaffold input planning

For the portable subset of PG fixed-fragment/contig workflows, materialize a
DISCO input JSON with fixed sequence letters and designable `-` positions:

```bash
python runner/guidance_plan_scaffold.py \
  --length 12 \
  --name scaffold_design \
  --fixed-fragment 3:GHK:bcell_1 \
  --seed-sequence AAAACCCCGGGG \
  --fixed-range 9:12:seed_tail \
  --pg-seed-pdb /path/to/seed.pdb \
  --pg-contig A1-12,0 \
  --out-input-json /path/to/scaffold/input.json \
  --out-manifest-json /path/to/scaffold/manifest.json
```

This is not a PG fixed-backbone or structural contig port. It preserves
sequence-level constraints that DISCO can express in `proteinChain.sequence`.
If `--pg-seed-pdb`, structural `--pg-contig`, `--pg-inpaint-seq`, or
`--pg-inpaint-str` arguments are supplied, the manifest records them as
unsupported structural semantics. Use the resulting `input.json` with normal
DISCO inference only after accepting that limitation.

### Guidance trajectory analysis

For a finished guided DISCO run with diagnostics, analyze the intermediate
sequence/score trajectory without launching inference:

```bash
python runner/guidance_analyze_trajectory.py \
  --diagnostics-jsonl /path/to/diagnostics/sample_guidance_diagnostics.jsonl \
  --out-csv /path/to/sample_trajectory.csv \
  --out-summary-json /path/to/sample_trajectory_summary.json
```

Repeat `--diagnostics-jsonl` for multiple samples. The CSV keeps one row per
diagnostic step and exposes `combined_score_delta`,
`sequence_changes_from_previous`, `new_present_motifs`, `lost_present_motifs`,
`is_best_combined`, `is_final_step`, and `best_output_selected_record`. The JSON
summary aggregates best step, final step, best sequence, final sequence, and
whether the best intermediate scored higher than the final sequence.

### Coordinate trajectory tensor dump

For full coordinate trajectories, enable the optional tensor dump during
inference:

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  cogen_guidance.trajectory_dump.enabled=true \
  cogen_guidance.trajectory_dump.dtype=float16 \
  cogen_guidance.trajectory_dump.max_steps=20
```

The files are written under `dump_dir/trajectories/`. `max_steps=0` means no
trajectory-step truncation. `save_final_coordinate=true` also stores the final
coordinate tensor. Use this only when coordinate-level trajectory inspection is
needed; the lightweight diagnostic JSONL/CSV path is usually enough for motif
score evolution.

Final and intermediate coordinate outputs are indexed separately. The final PDB
and final sequence are scored by the normal completed-dump analyzer. Saved
intermediate tensors are only trajectory artifacts; their CSV index may repeat
sequence-level motif diagnostics for the corresponding timestep, but it does
not claim pLDDT/structure confidence unless that exact intermediate state has
also been written and scored by a downstream tool.

### Run reproducibility bundle

After a planned run and analysis have produced artifacts, create one compact
bundle:

```bash
python runner/guidance_bundle_run.py \
  --run-dir /path/to/pg_vs_disco_plan/DISCO_standard_0 \
  --dump-dir /path/to/disco_output \
  --summary-csv /path/to/disco_analysis.csv \
  --out-json /path/to/run_bundle.json \
  --issues-csv /path/to/run_bundle_issues.csv \
  --require-plan \
  --require-summary \
  --require-diagnostics
```

The bundle does not copy structures or diagnostics into JSON. It records file
paths, file sizes, bounded SHA256 hashes, summary-score ranges, provenance
counts, diagnostic runtime totals, and command/manifests needed to reproduce or
audit the run. This closes the storage/logging gap between raw DISCO outputs
and normalized PG-vs-DISCO comparison tables.

### Score and pLDDT rule

For comparison, a confidence/pLDDT value is usable only when it is tied to the
same saved protein that the sequence and motif metrics summarize.

ProteinGenerator rows in the local smokes satisfy this through the saved PDB:
the sequence and `pg_plddt_mean` are both read from that PDB. DISCO rows in the
current inference path do not yet have a native final pLDDT/summary-confidence
artifact, so they are marked `confidence_match_status=missing` and
`score_provenance_status=disco_unmatched_confidence`. Regular smoke validation
can allow this with `--no-require-confidence`; strict confidence validation is
expected to fail for DISCO until an external same-artifact metric or native
DISCO confidence output is merged.

### PG-to-DISCO implementation parity audit

The parity audit is an implementation-completion map, not a biological result
validator. It records which ProteinGenerator motif/immunogen/alternating/
logging/evaluation features have a DISCO port, which are partial, and which are
not ported. It then checks whether the listed DISCO files and named CPU tests
exist in the current checkout:

```bash
python runner/guidance_audit_parity.py \
  --disco-root /path/to/DISCO \
  --pg-root /path/to/ProteinGenerator_MOTIFS \
  --out-report-json /path/to/parity_report.json \
  --out-md /path/to/parity_report.md
```

The current local audit reports `24` ported items, `6` partial items, `2` not
ported items, and `1` not-applicable item. The partial/open items are explicit
by design: DISCO confidence matching is not identical to PG in-loop
`pred_lddt`, the evolutionary outer loop is command/template/fitness capable
but not an autonomous heavy-run executor, second-pass reprediction is planned
as fixed-sequence DISCO commands rather than executed inside the audit,
external ESM/RMSD/Rosetta metrics are imported/ranked but not run inside DISCO,
sequence-level scaffold planning exists without PG coordinate contigs, and
cleavage fold-switch sampling plus ligand-aware guidance are not part of this
DISCO motif port.

### PG-vs-DISCO experiment planner

Before launching expensive runs, write a reproducible matrix:

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

The planner writes:

- `planned_runs.csv`,
- `plan_manifest.json`,
- one `command.txt`, `analysis_command.txt`, and `run_manifest.json` per run,
- length-specific DISCO `input.json` files,
- `compare_command.txt` for the final normalization step.

### Mac-local comparison smoke status

The current local execution artifacts are deliberately tiny Mac CPU smokes:

```text
outputs/pg_disco_compare_local_smoke_fixedenv_t2_2026_06_04/
outputs/pg_disco_compare_local_immunogen_smoke_2026_06_04/
```

The first folder covers Standard and Alternating for DISCO and
ProteinGenerator. The second folder covers Immunogen for both systems. All
planned generation and analysis commands completed for these smokes. The normal
comparison validation passes with `--no-require-confidence`; strict confidence
validation intentionally fails for DISCO because no native same-sample
pLDDT/summary-confidence artifact is emitted in this path.

No biological model-quality conclusion should be inferred from these smokes.
They use seed `0`, length `16`, DISCO `N_step=2`, PG `T=2`, and `N_cycle=1`;
they are execution, logging, storage, and provenance checks.

It records that PG `--T` and DISCO `sample_diffusion.N_step` are budget knobs,
not identical diffusion-time variables. `model.N_cycle` and PG `--n_cycle` are
kept equal by default (`4`) because that is the most direct shared recycle
knob.

Missing required Immunogen inputs are represented explicitly in the plan:
DISCO Immunogen without `--fixed-fragment` becomes
`requires_immunogen_fixed_fragments`; ProteinGenerator Immunogen without seed
PDB, seed sequence, or fixed-fragment JSON becomes
`requires_pg_immunogen_seed_inputs`.

Audit the planned matrix before launching expensive jobs:

```bash
python runner/guidance_audit_fairness.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/fairness_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/fairness_issues.csv \
  --recommended-n-cycle 4
```

This is stricter than the textual `fairness_notes`: it produces machine-readable
errors for mismatched `n_cycle`, length, motifs, designs per replicate, seed
sets, replicate balance, missing systems, or missing/non-positive budget knobs.

### PG-vs-DISCO comparison validation

Run the validator before making scientific claims from a planned matrix:

```bash
python runner/guidance_validate_comparison.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/validation_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/validation_issues.csv \
  --require-diagnostics
```

For a dry planning-stage audit, allow missing completed summaries:

```bash
python runner/guidance_validate_comparison.py \
  --planned-runs-csv /path/to/pg_vs_disco_plan/planned_runs.csv \
  --out-report-json /path/to/pg_vs_disco_plan/validation_report.json \
  --out-issues-csv /path/to/pg_vs_disco_plan/validation_issues.csv \
  --allow-missing-summaries
```

The default validator expects both `DISCO` and `ProteinGenerator` in every
mode/variant group. For single-system smoke checks only, override
`--required-systems DISCO`.

### Completed-design quality validation

Run this after analysis and before interpreting rows as acceptable designs:

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
  --min-unique-motifs 1 \
  --max-homopolymer 4 \
  --min-entropy 0.50
```

Optional gates include `--max-rmsd` and `--min-spread-score`. The
`--require-same-evaluation-artifact` gate rejects rows where sequence,
structure, and confidence/pLDDT are not proven to come from the same evaluated
artifact. Any numeric gate can be skipped with `none`, for example
`--min-entropy none`.

### External ESM/RMSD/Rosetta metric merge

When external tools have produced optional metric CSVs, merge them into a DISCO
or normalized PG/DISCO summary:

```bash
python runner/guidance_merge_external_quality.py \
  --summary-csv /path/to/disco_analysis.csv \
  --external-metrics-csv /path/to/external_metrics.csv \
  --out-csv /path/to/disco_analysis_with_external_quality.csv \
  --report-json /path/to/external_quality_report.json \
  --top-csv /path/to/top_posthoc_quality.csv
```

The external CSV is joined by `design_id` by default and may contain:
`esm2_mean_token_probability`, `esm2_perplexity`, `ca_rmsd_to_reference`,
`ca_rmsd_to_seed`, `rosetta_score_per_residue`, and/or `rosetta_score`.
The module computes the documented PG-style post-hoc rank score:

```text
posthoc_quality_score =
  0.30 * confidence
  + 0.20 * ESM2 rank
  + 0.20 * RMSD rank
  + 0.20 * Rosetta rank
  + 0.10 * motif_score
```

Higher ESM-2 probability is better; lower ESM-2 perplexity, lower RMSD, and
lower Rosetta score are better. Missing external metrics receive component
score `0.0` and are recorded in `external_metric_missing`, so these rows remain
auditable instead of silently looking complete. This runner does not execute
ESM-2, ESMFold, or Rosetta.

### Immunogen category validation

Run this after `guidance_analyze_disco.py` when the analysis included
`--tcell-motifs` and/or `--cleavage-motifs`:

```bash
python runner/guidance_validate_immunogen.py \
  --summary-csv /path/to/disco_immunogen_analysis.csv \
  --out-report-json /path/to/immunogen_validation.json \
  --out-issues-csv /path/to/immunogen_validation_issues.csv \
  --tcell-motifs AAK,LLK \
  --cleavage-motifs TT \
  --require-fixed-preserved \
  --require-same-evaluation-artifact \
  --min-tcell-coverage 1.0 \
  --min-cleavage-coverage 1.0
```

The output report keeps category counts (`fixed_passed_rows`,
`tcell_passed_rows`, `cleavage_passed_rows`, `evaluation_passed_rows`) separate
from the overall pass/fail result.

### PG-vs-DISCO CSV comparison

The comparison runner consumes existing CSVs; it does not launch either model.
It writes:

- normalized per-design rows,
- aggregate rows by `(system, mode, variant)`.

Example:

```bash
python runner/guidance_compare.py \
  --source ProteinGenerator:standard:standard_gradient_coverage:/path/to/pg_motif_summary.csv \
  --source DISCO:standard:standard_gradient_coverage:/path/to/disco_analysis.csv \
  --out-normalized /path/to/normalized_designs.csv \
  --out-summary /path/to/comparison_summary.csv
```

The source format is either `system:variant:path` for backward compatibility or
`system:mode:variant:path` when the mode should travel into reports.

This is the reduction layer after validation. It now writes score-provenance
fields plus `comparison_score`, `best_confidence_design_id`, and
`best_comparison_design_id`. Real scientific conclusions still require actual
completed model runs, adequate sample sizes, and any domain-specific structural
or ligand-aware metrics that the current CSV layer does not yet compute.

### PG-vs-DISCO Markdown report

After validation and comparison reduction:

```bash
python runner/guidance_report_comparison.py \
  --normalized-csv /path/to/normalized_designs.csv \
  --summary-csv /path/to/comparison_summary.csv \
  --validation-report-json /path/to/validation_report.json \
  --out-md /path/to/comparison_report.md
```

This report is a review artifact, not a substitute for the validation gate. It
lists top designs by `comparison_score`, reports provenance statuses, and
repeats the key interpretation caveats.

### Hard scoring

Hard scoring runs on concrete token/string states. It counts exact motif hits,
including overlaps, and reports present/missing motifs. This is used for
diagnostics and save-best tracking.

## 7. Save-Best Combined

The controller can track the best observed step:

```yaml
cogen_guidance.save_best.enabled: true
cogen_guidance.save_best.select_best_output: true
```

The initial combined score is:

```text
combined =
  motif_score_weight * hard_motif_score
  + confidence_weight * sequence_confidence
```

The implementation stores sequence and structure tensors for the best observed
step. For batched or duplicated sequence inference, this is tracked per sample:
sample 0, sample 1, and so on each keep their own best sequence/structure state.
At loop finalization, when `select_best_output` is true, the loop updates each
sample row to that sample's own best state and appends a diagnostic record.
The older global `get_best_state()` path remains available as a compatibility
view of the single best state across all samples.

The diagnostic record keeps the historical scalar fields for sample 0 and adds
batch-aware audit fields:

```text
guidance_batch_size
guidance_batch_sample_scores
guidance_global_best_sample_index
guidance_global_best_step
guidance_global_best_score
guidance_best_output_samples
```

Alternating metrics can optionally add another score term:

```bash
'cogen_guidance.alternating.enabled=true' \
'cogen_guidance.alternating.motif_list1=[GP,DK]' \
'cogen_guidance.alternating.motif_list2=[DR,VW]' \
cogen_guidance.alternating.combined_weight=0.5
```

## 8. Testing

The first tests are small and CPU-only:

```text
tests/test_guidance.py
```

They cover:

- hard motif overlap counting,
- gradient guidance increasing motif-relevant logits,
- local rescue respecting fixed positions,
- controller best-step tracking, including per-sample batch save-best and
  best-output selection.
- alternating hard-score preference and pattern reporting.
- alternating logit guidance objective improvement and fixed-position masking.
- alternating logit guidance no-op behavior when motifs do not fit the sequence.
- immunogen fixed-fragment mask and preservation scoring.
- immunogen fixed-position sampler mask blocking.
- immunogen mutable-window logit guidance.
- immunogen spacing score.
- evolutionary command construction, mutation, next-generation selection, and
  DISCO sequence-output parsing.
- DISCO completed-dump parsing, exact confidence matching, fixed-fragment
  output scoring, T-cell motif scoring, and cleavage motif scoring.
- Immunogen category validation for B-cell/fixed-fragment preservation, T-cell
  coverage, cleavage coverage, and same-artifact provenance.
- Immunogen second-pass reprediction planning from completed analysis CSVs into
  fixed-sequence DISCO input JSONs and analysis commands.
- sequence-level scaffold input planning from fixed fragments, seed ranges, and
  PG scaffold provenance, with explicit unsupported-structure issues.
- DISCO completed-dump sequence-quality metrics, structure-path matching,
  optional seed-sequence distance, and optional PDB CA-RMSD to seed.
- DISCO evaluation-bundle provenance for sequence/structure/confidence/
  diagnostics same-sample matching.
- diagnostic JSONL/CSV/runtime serialization and analyzer ingestion.
- offline trajectory analysis for diagnostic JSONL files.
- optional full coordinate trajectory tensor dumping with manifests.
- post-run reproducibility bundle generation and validation.
- PG-vs-DISCO matrix planning, per-run manifests, length-specific DISCO input
  JSONs, and comparison-command scaffolding.
- PG-vs-DISCO fairness audit for matched N-cycle, lengths, motifs, designs,
  seeds, replicate balance, systems, and explicit budget knobs.
- PG-vs-DISCO validation of required systems, planned artifacts, completed
  summaries, sequence lengths, confidence provenance, diagnostics, and
  Immunogen fixed-fragment preservation.
- completed-design quality validation for confidence, motif coverage,
  low-complexity guards, fixed fragments, optional RMSD/spread, diagnostics,
  structure presence, confidence provenance, and same-artifact evaluation.
- external ESM/RMSD/Rosetta metric merge with PG-style posthoc rank scoring.
- PG-vs-DISCO row normalization, score-provenance tracking, aggregate
  comparison, and Markdown report rendering.
- PG-to-DISCO implementation parity audit covering ported, partial,
  not-ported, and not-applicable ProteinGenerator feature records.

The DISCO `.venv` used for local inference does not include `pytest`; targeted
parser/runtime checks can be verified by importing and calling the relevant
functions directly with `.venv/bin/python`. The system `pytest` on this Mac uses
Base Python and is missing DISCO dependencies such as `rdkit`.

## 9. Pending Porting Work

The following ProteinGenerator features are not yet ported:

- automatic execution of DISCO second-pass reprediction; current support writes
  fixed-sequence input JSONs and commands, then expects an explicit operator run,
- richer T-cell and cleavage motif presets/workflow metadata,
- larger non-smoke PG-vs-DISCO comparison runs with enough seeds, diffusion
  budget, and posthoc quality scoring to support model-level conclusions,
- true PG-style structural seed/contig scaffolding and cleavage fold-switch
  sampling,
- ligand-aware combined scoring.

Future work should keep these as separate packages under
`disco/model/cogen_inference/guidance/` or top-level analysis scripts rather
than expanding the DISCO loop body.
