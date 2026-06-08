import csv
import json
import tempfile

import torch
from omegaconf import OmegaConf
from pathlib import Path

from disco.data.constants import MASK_TOKEN_IDX
from disco.model.cogen_inference.guidance import (
    HardMotifScorer,
    RunBundleConfig,
    audit_fairness_csv,
    create_run_bundle,
    tokens_to_sequence,
    validate_run_bundle,
    write_fairness_report,
    write_run_bundle,
)
from disco.model.cogen_inference.guidance.alternating import (
    AlternatingLogitGuidance,
    SoftTransitionAlternatingLogitGuidance,
    combined_alternating_score,
    score_alternation,
)
from disco.model.cogen_inference.guidance.artifact_provenance import (
    write_artifact_provenance,
)
from disco.model.cogen_inference.guidance.comparison import (
    normalize_disco_row,
    normalize_pg_row,
    summarize_by_group,
    write_normalized_rows,
    write_summary_rows,
)
from disco.model.cogen_inference.guidance.comparison_report import (
    render_comparison_report,
    write_comparison_report,
)
from disco.model.cogen_inference.guidance.comparison_validation import (
    validate_comparison_plan,
    write_validation_report,
)
from disco.model.cogen_inference.guidance.controller import GuidanceController
from disco.model.cogen_inference.guidance.disco_analysis import (
    analyze_disco_dump_rows,
    sequence_entries_from_disco_sequence_text,
)
from disco.model.cogen_inference.guidance.diagnostics import (
    diagnostic_summary,
    write_diagnostic_outputs,
)
from disco.model.cogen_inference.guidance.evolution import (
    DiscoEvaluation,
    DiscoGuidanceGenome,
    apply_template_to_input_records,
    build_inference_command,
    fitness_score,
    materialize_template_input_json,
    mutate_genome,
    next_generation,
    parse_motifs as parse_evolution_motifs,
    sequence_from_disco_sequence_text,
    template_to_disco_sequence,
)
from disco.model.cogen_inference.guidance.experiment_plan import (
    ComparisonPlanConfig,
    FixedFragmentPlan,
    plan_runs,
    sequence_with_fixed_fragments,
)
from disco.model.cogen_inference.guidance.external_quality import (
    annotate_external_quality_rows,
    write_external_quality_outputs,
)
from disco.model.cogen_inference.guidance.immunogen import (
    MutableWindowMotifLogitGuidance,
    build_fixed_position_mask,
    build_fixed_sequence_mask,
    score_fixed_fragments,
    spacing_score,
    spec_from_dict,
)
from disco.model.cogen_inference.guidance.immunogen_validation import (
    ImmunogenValidationConfig,
    validate_immunogen_summary_csv,
    write_immunogen_validation_report,
)
from disco.model.cogen_inference.guidance.motif import AA_TO_IDX, MotifLogitGuidance
from disco.model.cogen_inference.guidance.parity import (
    audit_parity,
    write_parity_report,
)
from disco.model.cogen_inference.guidance.quality_validation import (
    QualityGateConfig,
    validate_summary_csv,
    write_quality_report,
)
from disco.model.cogen_inference.guidance.reprediction import (
    RepredictionPlanConfig,
    parse_seed_list,
    write_reprediction_plan,
)
from disco.model.cogen_inference.guidance.scaffold_plan import (
    audit_pg_scaffold_arguments,
    fixed_fragments_from_json,
    parse_fixed_range,
    sequence_from_scaffold_constraints,
    write_scaffold_plan,
)
from disco.model.cogen_inference.guidance.trajectory import (
    trajectory_rows,
    trajectory_summary,
    write_trajectory_outputs,
)
from disco.model.cogen_inference.guidance.trajectory_dump import (
    write_coordinate_trajectory_outputs,
)


def write_csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def pdb_ca_text(coords):
    lines = []
    for index, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            "ATOM  {atom:5d}  CA  ALA A{res:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C".format(
                atom=index,
                res=index,
                x=x,
                y=y,
                z=z,
            )
        )
    return "\n".join(lines) + "\n"


def logits_for_sequence(sequence, width=32, high=6.0, low=-6.0):
    logits = torch.full((len(sequence), width), low)
    for index, aa in enumerate(sequence):
        logits[index, AA_TO_IDX[aa]] = high
    return logits


def test_hard_motif_scorer_counts_overlaps():
    scorer = HardMotifScorer(motifs="AA,GHK", motif_all_weight=1.0, motif_many_weight=0.0)

    result = scorer.score_sequence("AAAGHK")

    assert result.counts["AA"] == 2
    assert result.counts["GHK"] == 1
    assert result.num_unique == 2
    assert result.motif_score == 1.0


def test_gradient_guidance_increases_requested_motif_logits():
    guidance = MotifLogitGuidance(
        motifs="GHK",
        gradient_enabled=True,
        gradient_scale=2.0,
        local_rescue_enabled=False,
    )
    logits = torch.zeros(6, 32)
    xt_seq = torch.full((6,), MASK_TOKEN_IDX)
    can_change = torch.ones(6, dtype=torch.bool)

    guided = guidance.apply(logits, xt_seq, can_change)

    assert guided.shape == logits.shape
    assert guided[:, :20].max() > logits[:, :20].max()
    assert guidance.last_result.gradient_applied


def test_local_rescue_respects_fixed_positions():
    guidance = MotifLogitGuidance(
        motifs="GHK",
        gradient_enabled=False,
        local_rescue_enabled=True,
        local_rescue_bonus=5.0,
        local_rescue_top_k=3,
        local_rescue_min_probability=0.01,
    )
    logits = torch.zeros(5, 32)
    xt_seq = torch.full((5,), MASK_TOKEN_IDX)
    can_change = torch.tensor([True, False, True, True, True])

    guided = guidance.apply(logits, xt_seq, can_change)

    assert torch.equal(guided[1], logits[1])


def test_controller_tracks_best_step():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "GHK",
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {"motif_all_weight": 1.0, "motif_many_weight": 0.0},
            "save_best": {
                "enabled": True,
                "select_best_output": True,
                "motif_score_weight": 1.0,
                "confidence_weight": 0.0,
            },
        }
    )
    logits = torch.zeros(5, 32)
    can_change = torch.ones(5, dtype=torch.bool)
    bad = torch.tensor([MASK_TOKEN_IDX, MASK_TOKEN_IDX, MASK_TOKEN_IDX, MASK_TOKEN_IDX, MASK_TOKEN_IDX])
    good = torch.tensor([7, 8, 11, MASK_TOKEN_IDX, MASK_TOKEN_IDX])

    controller.observe_step(bad, torch.zeros(5, 3), logits, can_change, step=0)
    controller.observe_step(good, torch.ones(5, 3), logits, can_change, step=1)

    best = controller.get_best_state()
    assert best is not None
    assert best.step == 1
    assert tokens_to_sequence(best.sequence_tokens).startswith("GHK")


def test_controller_tie_breaks_equal_score_toward_decoded_later_step():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "GHK",
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {"motif_all_weight": 1.0, "motif_many_weight": 0.0},
            "save_best": {
                "enabled": True,
                "select_best_output": True,
                "motif_score_weight": 1.0,
                "confidence_weight": 0.0,
            },
        }
    )
    logits = torch.zeros(5, 32)
    can_change = torch.ones(5, dtype=torch.bool)
    masked = torch.full((5,), MASK_TOKEN_IDX)
    decoded_no_motif = torch.zeros(5, dtype=torch.long)

    controller.observe_step(masked, torch.zeros(5, 3), logits, can_change, step=0)
    controller.observe_step(
        decoded_no_motif,
        torch.ones(5, 3),
        logits,
        can_change,
        step=1,
    )

    best = controller.get_best_state()
    assert best is not None
    assert best.step == 1
    assert tokens_to_sequence(best.sequence_tokens).count("-") == 0


def test_controller_prefers_fully_decoded_state_over_higher_masked_score():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "VW",
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {"motif_all_weight": 1.0, "motif_many_weight": 0.0},
            "save_best": {
                "enabled": True,
                "select_best_output": True,
                "motif_score_weight": 1.0,
                "confidence_weight": 0.0,
            },
        }
    )
    logits = torch.zeros(5, 32)
    can_change = torch.ones(5, dtype=torch.bool)
    masked_with_motif = torch.tensor([AA_TO_IDX["V"], AA_TO_IDX["W"], MASK_TOKEN_IDX, MASK_TOKEN_IDX, MASK_TOKEN_IDX])
    decoded_no_motif = torch.zeros(5, dtype=torch.long)

    controller.observe_step(
        masked_with_motif,
        torch.zeros(5, 3),
        logits,
        can_change,
        step=0,
    )
    controller.observe_step(
        decoded_no_motif,
        torch.ones(5, 3),
        logits,
        can_change,
        step=1,
    )

    best = controller.get_best_state()
    assert best is not None
    assert best.step == 1
    assert tokens_to_sequence(best.sequence_tokens).count("-") == 0


def test_controller_tracks_and_selects_best_step_per_batch_sample():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "GHK",
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {"motif_all_weight": 1.0, "motif_many_weight": 0.0},
            "save_best": {
                "enabled": True,
                "select_best_output": True,
                "motif_score_weight": 1.0,
                "confidence_weight": 0.0,
            },
        }
    )
    logits = torch.zeros(2, 5, 32)
    can_change = torch.ones(2, 5, dtype=torch.bool)
    bad = torch.tensor([0, 0, 0, 0, 0])
    good = torch.tensor([7, 8, 11, 0, 0])

    step0 = torch.stack([bad, good])
    struct0 = torch.stack([torch.zeros(5, 3), torch.ones(5, 3)])
    first_details = controller.observe_step(
        step0,
        struct0,
        logits,
        can_change,
        step=0,
    )

    step1 = torch.stack([good, bad])
    struct1 = torch.stack([torch.full((5, 3), 2.0), torch.full((5, 3), 3.0)])
    second_details = controller.observe_step(
        step1,
        struct1,
        logits,
        can_change,
        step=1,
    )

    best_by_sample = controller.get_best_states_by_sample()
    assert first_details["guidance_batch_size"] == 2
    assert second_details["guidance_batch_size"] == 2
    assert len(second_details["guidance_batch_sample_scores"]) == 2
    assert best_by_sample[0].step == 1
    assert best_by_sample[1].step == 0
    assert tokens_to_sequence(best_by_sample[0].sequence_tokens).startswith("GHK")
    assert tokens_to_sequence(best_by_sample[1].sequence_tokens).startswith("GHK")

    selected = controller.select_best_outputs(
        xt_seq=torch.stack([bad, bad]),
        structure=torch.full((2, 5, 3), 9.0),
    )
    assert selected is not None
    selected_seq, selected_structure, selected_details = selected
    assert tokens_to_sequence(selected_seq[0]).startswith("GHK")
    assert tokens_to_sequence(selected_seq[1]).startswith("GHK")
    assert torch.equal(selected_structure[0], struct1[0])
    assert torch.equal(selected_structure[1], struct0[1])
    assert selected_details["guidance_best_output_num_samples"] == 2


def test_alternating_score_prefers_switching_groups():
    alternating = combined_alternating_score("AAGPAADRAAGP", ["GP"], ["DR"])
    same_group = combined_alternating_score("AAGPAAGPAA", ["GP"], ["DR"])

    assert alternating > same_group


def test_alternating_metrics_report_pattern():
    metrics = score_alternation("AAGPAADRAAKVK", ["GP", "KVK"], ["DR"])

    assert metrics["motif_pattern"] == "1->2->1"
    assert metrics["same_group_transitions"] == 0


def test_alternating_logit_guidance_increases_slot_objective():
    guidance = AlternatingLogitGuidance(
        motif_list1=["GP"],
        motif_list2=["DR"],
        gradient_scale=2.0,
        max_slots=4,
    )
    logits = torch.zeros(16, 32)
    xt_seq = torch.full((16,), MASK_TOKEN_IDX)
    can_change = torch.ones(16, dtype=torch.bool)

    before, _ = guidance.objective_from_logits(logits)
    guided = guidance.apply(logits, xt_seq, can_change)
    after, _ = guidance.objective_from_logits(guided)

    assert guided.shape == logits.shape
    assert after > before
    assert guidance.last_result.gradient_applied
    assert guidance.last_result.num_slots == 4


def test_alternating_logit_guidance_respects_fixed_positions():
    guidance = AlternatingLogitGuidance(
        motif_list1=["GP"],
        motif_list2=["DR"],
        gradient_scale=2.0,
        max_slots=4,
    )
    logits = torch.zeros(8, 32)
    xt_seq = torch.full((8,), MASK_TOKEN_IDX)
    can_change = torch.tensor([True, False, True, True, True, True, True, True])

    guided = guidance.apply(logits, xt_seq, can_change)

    assert torch.equal(guided[1], logits[1])


def test_alternating_logit_guidance_noops_when_motifs_do_not_fit():
    guidance = AlternatingLogitGuidance(
        motif_list1=["GPGPGP"],
        motif_list2=["DRDRDR"],
        gradient_scale=2.0,
        max_slots=2,
    )
    logits = torch.zeros(4, 32)
    xt_seq = torch.full((4,), MASK_TOKEN_IDX)
    can_change = torch.ones(4, dtype=torch.bool)

    guided = guidance.apply(logits, xt_seq, can_change)

    assert torch.equal(guided, logits)
    assert not guidance.last_result.gradient_applied


def test_soft_transition_logit_guidance_prefers_alternating_order():
    guidance = SoftTransitionAlternatingLogitGuidance(
        motif_list1=["GP", "DK"],
        motif_list2=["DR", "VW"],
        same_group_penalty_weight=0.5,
    )
    alternating = logits_for_sequence("AAGPAAADRAAA")
    same_group = logits_for_sequence("AAGPAADKAAAA")

    alternating_score, alternating_details = guidance.objective_from_logits(
        alternating
    )
    same_group_score, same_group_details = guidance.objective_from_logits(
        same_group
    )

    assert alternating_score > same_group_score
    assert alternating_details["transition"] > same_group_details["transition"]


def test_soft_transition_logit_guidance_updates_only_changeable_positions():
    guidance = SoftTransitionAlternatingLogitGuidance(
        motif_list1=["GP"],
        motif_list2=["DR"],
        gradient_scale=2.0,
        same_group_penalty_weight=0.5,
        end_weight=0.8,
    )
    logits = torch.zeros(16, 32)
    xt_seq = torch.full((16,), MASK_TOKEN_IDX)
    can_change = torch.ones(16, dtype=torch.bool)
    can_change[1] = False

    before, _ = guidance.objective_from_logits(logits)
    guided = guidance.apply(logits, xt_seq, can_change)
    after, _ = guidance.objective_from_logits(guided)

    assert guided.shape == logits.shape
    assert after > before
    assert guidance.last_result.gradient_applied
    assert guidance.last_result.same_group_penalty > 0
    assert torch.equal(guided[1], logits[1])


def test_controller_can_apply_alternating_without_plain_motif_guidance():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": None,
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {},
            "save_best": {"enabled": False},
            "alternating": {
                "enabled": True,
                "motif_list1": ["GP"],
                "motif_list2": ["DR"],
                "expected_start_group": "list1",
                "combined_weight": 0.0,
                "logit_guidance_enabled": True,
                "logit_gradient_scale": 2.0,
                "logit_max_slots": 4,
            },
        }
    )
    logits = torch.zeros(16, 32)
    xt_seq = torch.full((16,), MASK_TOKEN_IDX)
    can_change = torch.ones(16, dtype=torch.bool)

    guided = controller.modify_logits(logits, xt_seq, can_change, step=0)

    assert not torch.equal(guided, logits)
    assert controller.last_modify_details["guidance_alternating_logit_applied"]


def test_controller_can_apply_transition_only_alternating_guidance():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": None,
            "motif_weights": None,
            "logit_guidance": {"enabled": False},
            "scoring": {},
            "save_best": {"enabled": False},
            "alternating": {
                "enabled": True,
                "motif_list1": ["GP"],
                "motif_list2": ["DR"],
                "expected_start_group": "list1",
                "combined_weight": 0.0,
                "logit_guidance_enabled": False,
                "transition_logit_guidance_enabled": True,
                "transition_logit_gradient_scale": 2.0,
                "transition_same_group_penalty_weight": 0.5,
            },
        }
    )
    logits = torch.zeros(16, 32)
    xt_seq = torch.full((16,), MASK_TOKEN_IDX)
    can_change = torch.ones(16, dtype=torch.bool)

    guided = controller.modify_logits(logits, xt_seq, can_change, step=0)

    assert not torch.equal(guided, logits)
    assert controller.last_modify_details[
        "guidance_alternating_transition_logit_applied"
    ]
    assert "guidance_alternating_logit_applied" not in controller.last_modify_details


def test_immunogen_fixed_mask_and_scoring():
    spec = spec_from_dict(
        {
            "fixed_fragments": [
                {"start_1based": 2, "sequence": "CD", "label": "bcell"}
            ]
        }
    )

    mask = build_fixed_sequence_mask("ACDE", spec)
    ok = score_fixed_fragments("ACDE", spec)
    bad = score_fixed_fragments("AWDE", spec)

    assert mask.tolist() == [False, True, True, False]
    assert ok["fixed_fragments_preserved"] is True
    assert bad["fixed_fragments_preserved"] is False
    assert bad["mismatches"][0]["label"] == "bcell"


def test_immunogen_spec_accepts_hydra_listconfig_fragments():
    cfg = OmegaConf.create(
        {
            "fixed_fragments": [
                {"start_1based": 1, "sequence": "ALL", "label": "seed"}
            ]
        }
    )

    spec = spec_from_dict(cfg)

    assert spec.fixed_fragments[0].sequence == "ALL"


def test_immunogen_fixed_position_mask_does_not_need_seed_sequence():
    spec = spec_from_dict(
        {
            "fixed_fragments": [
                {"start_1based": 2, "sequence": "CD", "label": "bcell"}
            ]
        }
    )

    mask = build_fixed_position_mask(5, spec)

    assert mask.tolist() == [False, True, True, False, False]


def test_immunogen_mutable_window_guidance_ignores_fixed_overlap():
    spec = spec_from_dict(
        {
            "fixed_fragments": [
                {"start_1based": 2, "sequence": "CD", "label": "bcell"}
            ]
        }
    )
    guidance = MutableWindowMotifLogitGuidance(
        fixed_spec=spec,
        motifs="GHK",
        gradient_enabled=True,
        gradient_scale=2.0,
        local_rescue_enabled=False,
    )
    logits = torch.zeros(5, 32)
    xt_seq = torch.full((5,), MASK_TOKEN_IDX)
    can_change = torch.ones(5, dtype=torch.bool)

    guided = guidance.apply(logits, xt_seq, can_change)

    assert torch.equal(guided, logits)
    assert not guidance.last_result.gradient_applied


def test_immunogen_mutable_window_guidance_uses_mutable_windows():
    spec = spec_from_dict(
        {
            "fixed_fragments": [
                {"start_1based": 2, "sequence": "CD", "label": "bcell"}
            ]
        }
    )
    guidance = MutableWindowMotifLogitGuidance(
        fixed_spec=spec,
        motifs="GHK",
        gradient_enabled=True,
        gradient_scale=2.0,
        local_rescue_enabled=False,
    )
    logits = torch.zeros(7, 32)
    xt_seq = torch.full((7,), MASK_TOKEN_IDX)
    can_change = torch.tensor([True, False, False, True, True, True, True])

    guided = guidance.apply(logits, xt_seq, can_change)

    assert not torch.equal(guided, logits)
    assert torch.equal(guided[1], logits[1])
    assert torch.equal(guided[2], logits[2])
    assert guidance.last_result.gradient_applied


def test_controller_blocks_immunogen_fixed_positions_in_change_mask():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "GHK",
            "logit_guidance": {"enabled": False},
            "scoring": {},
            "save_best": {"enabled": False},
            "immunogen": {
                "enabled": True,
                "fixed_fragments": [
                    {"start_1based": 2, "sequence": "CD", "label": "bcell"}
                ],
                "protect_fixed_fragments": True,
            },
        }
    )
    can_change = torch.ones(5, dtype=torch.bool)

    modified = controller.modify_change_mask(can_change)

    assert modified.tolist() == [True, False, False, True, True]
    assert can_change.tolist() == [True, True, True, True, True]


def test_controller_can_apply_immunogen_mutable_guidance():
    controller = GuidanceController(
        {
            "enabled": True,
            "motifs": "GHK",
            "logit_guidance": {"enabled": False},
            "scoring": {},
            "save_best": {"enabled": False},
            "immunogen": {
                "enabled": True,
                "fixed_fragments": [
                    {"start_1based": 2, "sequence": "CD", "label": "bcell"}
                ],
                "mutable_logit_guidance": {
                    "enabled": True,
                    "gradient_scale": 2.0,
                },
            },
        }
    )
    logits = torch.zeros(7, 32)
    xt_seq = torch.full((7,), MASK_TOKEN_IDX)
    can_change = torch.tensor([True, False, False, True, True, True, True])

    guided = controller.modify_logits(logits, xt_seq, can_change, step=0)

    assert not torch.equal(guided, logits)
    assert controller.last_modify_details[
        "guidance_immunogen_mutable_gradient_applied"
    ]


def test_immunogen_spacing_score_prefers_spread_motifs():
    spread = spacing_score("GHKAAKTTKS", ["GHK", "KTTKS"])
    clustered = spacing_score("GHKKTTKSAAA", ["GHK", "KTTKS"])

    assert spread > clustered


def test_evolution_command_contains_guidance_overrides():
    genome = DiscoGuidanceGenome(
        n_step=100,
        n_cycle=4,
        gradient_scale=2.0,
        motif_weight_all=0.4,
        motif_weight_many=0.6,
    )
    command = build_inference_command(
        python_executable="python",
        disco_root=Path("/repo/DISCO"),
        genome=genome,
        motifs=["GHK", "KTTKS"],
        input_json_path=Path("input_jsons/unconditional_config.json"),
        dump_dir=Path("out/evo"),
        seed=3,
    )

    assert "cogen_guidance.enabled=true" in command
    assert "cogen_guidance.motifs=[GHK,KTTKS]" in command
    assert "sample_diffusion.N_step=100" in command
    assert "model.N_cycle=4" in command
    assert "seeds=[3]" in command


def test_evolution_mutation_respects_fixed_n_cycle():
    genome = DiscoGuidanceGenome(n_cycle=4)

    for _ in range(20):
        mutated = mutate_genome(
            genome,
            mutation_rate=1.0,
            include_local_rescue=True,
            fixed_n_cycle=4,
        )
        assert mutated.n_cycle == 4


def test_evolution_next_generation_preserves_population_size():
    genome = DiscoGuidanceGenome()
    evaluations = [
        DiscoEvaluation(
            generation=0,
            individual=0,
            genome=genome,
            run_dir=Path("/tmp/a"),
            fitness=0.8,
            confidence=0.7,
            unique_motifs=2,
            total_motifs=3,
            sequence="GHKAAKTTKS",
            counts={"GHK": 1, "KTTKS": 1},
            command=[],
        ),
        DiscoEvaluation(
            generation=0,
            individual=1,
            genome=DiscoGuidanceGenome(gradient_scale=2.0),
            run_dir=Path("/tmp/b"),
            fitness=0.6,
            confidence=0.8,
            unique_motifs=1,
            total_motifs=1,
            sequence="GHKAAAAAAA",
            counts={"GHK": 1, "KTTKS": 0},
            command=[],
        ),
    ]

    population = next_generation(
        evaluations=evaluations,
        population_size=6,
        elite_fraction=0.2,
        mutation_rate=0.5,
        motifs=["GHK", "KTTKS"],
        template_from_elites=True,
        include_local_rescue=True,
        fixed_n_cycle=4,
    )

    assert len(population) == 6
    assert {item.n_cycle for item in population} == {4}
    assert population[0].template_sequence == "GHKXXKTTKS"


def test_evolution_template_to_disco_sequence_converts_x_to_mask():
    assert template_to_disco_sequence("GHKXXKTTKS") == "GHK--KTTKS"
    assert template_to_disco_sequence("GHK--KTTKS") == "GHK--KTTKS"


def test_evolution_apply_template_filters_matching_lengths_and_preserves_context():
    records = [
        {
            "name": "short",
            "sequences": [
                {"proteinChain": {"sequence": "-----"}},
                {"ligand": {"ccdCodes": ["ATP"]}},
            ],
        },
        {
            "name": "long",
            "sequences": [
                {"proteinChain": {"sequence": "--------"}},
                {"ligand": {"ccdCodes": ["HEM"]}},
            ],
        },
    ]

    updated = apply_template_to_input_records(records, "GHKXX")

    assert len(updated) == 1
    assert updated[0]["name"] == "short_template"
    assert updated[0]["sequences"][0]["proteinChain"]["sequence"] == "GHK--"
    assert updated[0]["sequences"][1]["ligand"]["ccdCodes"] == ["ATP"]
    assert records[0]["sequences"][0]["proteinChain"]["sequence"] == "-----"


def test_evolution_materializes_template_input_json():
    records = [
        {
            "name": "design70",
            "sequences": [{"proteinChain": {"sequence": "---------"}}],
        },
        {
            "name": "design100",
            "sequences": [{"proteinChain": {"sequence": "------------"}}],
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.json"
        output_path = Path(tmpdir) / "templated" / "input.json"
        base_path.write_text(json.dumps(records) + "\n")

        materialized = materialize_template_input_json(
            base_input_json_path=base_path,
            output_json_path=output_path,
            template_sequence="GHKXXKTTK",
        )

        assert materialized == output_path
        updated = json.loads(output_path.read_text())
        assert len(updated) == 1
        assert updated[0]["name"] == "design70_template"
        assert updated[0]["sequences"][0]["proteinChain"]["sequence"] == "GHK--KTTK"


def test_evolution_fitness_modes_and_sequence_parser():
    motifs = parse_evolution_motifs("GHK,KTTKS")
    balance = fitness_score(
        confidence=0.8,
        unique_motifs=2,
        total_motifs=2,
        required_motifs=len(motifs),
        target_total_motifs=6,
        mode="balance",
    )
    max_motifs = fitness_score(
        confidence=0.5,
        unique_motifs=2,
        total_motifs=8,
        required_motifs=len(motifs),
        target_total_motifs=6,
        mode="max_motifs",
    )
    text = ">cogen_seq 0\nCHAIN A type=protein mode=design seq=GHKAA\n"

    assert max_motifs > balance
    assert sequence_from_disco_sequence_text(text) == "GHKAA"


def test_disco_analysis_parses_multiple_sequence_entries():
    text = (
        ">cogen_seq 0\n"
        "CHAIN A type=protein mode=design seq=GHKAA\n"
        "CHAIN B type=dna mode=fixed seq=ACGT\n"
        ">cogen_seq 1\n"
        "MFPKTTKS\n"
        "ligand_smiles CCO\n"
    )

    entries = sequence_entries_from_disco_sequence_text(text)

    assert entries == [(0, "GHKAA"), (1, "MFPKTTKS")]


def test_disco_analysis_parses_masked_sequence_entries():
    text = ">cogen_seq 0\n-SPHHD---VH-C---\n"

    entries = sequence_entries_from_disco_sequence_text(text)

    assert entries == [(0, "-SPHHD---VH-C---")]


def test_disco_analysis_pairs_sequence_with_matching_confidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_dir = Path(tmpdir)
        seq_dir = dump_dir / "sequences"
        pdb_dir = dump_dir / "pdbs"
        seq_dir.mkdir()
        pdb_dir.mkdir()
        (seq_dir / "design_sample_0.txt").write_text(
            ">cogen_seq 0\n"
            "CHAIN A type=protein mode=design seq=GHKAAKTTKS\n"
        )
        (pdb_dir / "design_sample_0_summary_confidence_sample_0.json").write_text(
            json.dumps({"ranking_score": 75.0}) + "\n"
        )
        (pdb_dir / "design_sample_0_sample_0.pdb").write_text(
            pdb_ca_text([(0.0, 0.0, 0.0)])
        )
        write_diagnostic_outputs(
            diagnostics_dir=dump_dir / "diagnostics",
            sample_id="design_sample_0",
            records=[
                {
                    "step": 0,
                    "guidance_sequence": "GHK-----",
                    "guidance_combined_score": 0.1,
                },
                {
                    "step": 1,
                    "guidance_sequence": "GHKAAKTTKS",
                    "guidance_combined_score": 0.9,
                },
            ],
            runtime_seconds=1.25,
            sample_name="design",
            seed=0,
        )

        rows = analyze_disco_dump_rows(
            dump_dir=dump_dir,
            motifs=["GHK", "KTTKS"],
            fixed_spec=spec_from_dict(
                {
                    "fixed_fragments": [
                        {"start_1based": 1, "sequence": "GHK", "label": "bcell"}
                    ]
                }
            ),
            tcell_motifs=["AAK"],
            cleavage_motifs=["TT"],
        )
        assert Path(rows[0]["diagnostics_jsonl"]).exists()

    assert len(rows) == 1
    assert rows[0]["design_id"] == "design_sample_0"
    assert rows[0]["confidence"] == "0.750000"
    assert rows[0]["confidence_match_status"] == "exact_sample"
    assert rows[0]["structure_match_status"] == "exact_sequence_index"
    assert rows[0]["evaluation_status"] == "same_sample_complete"
    assert rows[0]["evaluation_diagnostics_status"] == "exact_base"
    assert rows[0]["unique_motifs"] == 2
    assert rows[0]["GHK_count"] == 1
    assert rows[0]["fixed_fragments_preserved"] is True
    assert rows[0]["tcell_unique_motifs"] == 1
    assert rows[0]["cleavage_unique_motifs"] == 1
    assert rows[0]["runtime_seconds"] == 1.25
    assert rows[0]["num_diagnostic_records"] == 2
    assert rows[0]["diagnostic_best_step"] == 1


def test_disco_analysis_adds_quality_and_structure_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_dir = Path(tmpdir) / "dump"
        seq_dir = dump_dir / "sequences"
        pdb_dir = dump_dir / "pdbs"
        seq_dir.mkdir(parents=True)
        pdb_dir.mkdir(parents=True)
        (seq_dir / "quality_sample_0.txt").write_text(
            ">cogen_seq 0\n"
            "CHAIN A type=protein mode=design seq=AAAC\n"
        )
        (pdb_dir / "quality_sample_0_sample_0.pdb").write_text(
            pdb_ca_text(
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0),
                ]
            )
        )
        (pdb_dir / "quality_sample_0_summary_confidence_sample_0.json").write_text(
            json.dumps({"ranking_score": 70.0}) + "\n"
        )
        seed_pdb = Path(tmpdir) / "seed.pdb"
        seed_pdb.write_text(
            pdb_ca_text(
                [
                    (10.0, 0.0, 0.0),
                    (11.0, 0.0, 0.0),
                    (12.0, 0.0, 0.0),
                    (13.0, 0.0, 0.0),
                ]
            )
        )

        rows = analyze_disco_dump_rows(
            dump_dir=dump_dir,
            motifs=["AAA"],
            seed_sequence="AAAT",
            seed_pdb=seed_pdb,
        )

    assert rows[0]["structure_match_status"] == "exact_sequence_index"
    assert rows[0]["evaluation_status"] == "same_sample_no_diagnostics"
    assert rows[0]["structure_path"].endswith("quality_sample_0_sample_0.pdb")
    assert rows[0]["longest_homopolymer"] == 3
    assert rows[0]["aa_entropy_norm"] != ""
    assert rows[0]["seed_sequence_identity"] == "0.750"
    assert rows[0]["num_sequence_changes_vs_seed"] == 1
    assert rows[0]["changed_positions_1based"] == "4"
    assert rows[0]["ca_rmsd_to_seed"] == "0.000"


def test_diagnostics_summary_selects_best_guidance_step():
    summary = diagnostic_summary(
        [
            {
                "step": 5,
                "guidance_sequence": "AAAA",
                "guidance_combined_score": 0.2,
            },
            {
                "step": 4,
                "guidance_sequence": "GHKAA",
                "guidance_combined_score": 0.8,
            },
        ]
    )

    assert summary["num_diagnostic_records"] == 2
    assert summary["best_step"] == 4
    assert summary["best_sequence"] == "GHKAA"


def test_write_diagnostic_outputs_serializes_tensor_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = write_diagnostic_outputs(
            diagnostics_dir=Path(tmpdir),
            sample_id="sample_0",
            records=[
                {
                    "step": torch.tensor(1),
                    "guidance_sequence": "GHK",
                    "guidance_combined_score": torch.tensor(0.7),
                    "nested": {"value": torch.tensor([1, 2])},
                }
            ],
            runtime_seconds=0.5,
            sample_name="sample",
            seed=3,
        )

        assert Path(result["diagnostics_jsonl"]).exists()
        assert Path(result["diagnostics_csv"]).exists()
        manifest = json.loads(Path(result["diagnostics_manifest"]).read_text())
        assert manifest["runtime_seconds"] == 0.5
        assert manifest["best_step"] == 1


def test_experiment_plan_writes_fixed_fragment_sequence():
    sequence = sequence_with_fixed_fragments(
        length=8,
        fixed_fragments=[FixedFragmentPlan(2, "GHK", "bcell")],
    )

    assert sequence == "-GHK----"


def test_experiment_plan_materializes_core_matrix():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pg_seed_pdb = root / "seed.pdb"
        pg_seed_pdb.write_text("HEADER seed\n")
        pg_fixed_json = root / "fixed.json"
        pg_fixed_json.write_text(
            json.dumps(
                {
                    "fixed_fragments": [
                        {"start_1based": 1, "sequence": "GHK", "label": "bcell"}
                    ]
                }
            )
            + "\n"
        )
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard", "immunogen", "alternating"],
            systems=["DISCO", "ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            n_cycle=4,
            standard_length=8,
            immunogen_length=8,
            alternating_length=8,
            disco_n_step=50,
            pg_standard_T=60,
            pg_short_T=4,
            standard_motifs=["GHK", "KTTKS"],
            immunogen_motifs=["AAK"],
            alternating_list1=["GP"],
            alternating_list2=["DR"],
            fixed_fragments=[FixedFragmentPlan(1, "GHK", "bcell")],
            pg_seed_pdb=pg_seed_pdb,
            pg_seed_sequence="GHKAAAAA",
            pg_fixed_fragments_json=pg_fixed_json,
            disco_python="python",
            pg_python="python",
        )

        runs = plan_runs(config)

        assert len(runs) == 6
        assert {run.status for run in runs} == {"planned"}
        assert (config.out_dir / "planned_runs.csv").exists()
        assert (config.out_dir / "compare_command.txt").exists()

        disco_immunogen = [
            run for run in runs
            if run.system == "DISCO" and run.mode == "immunogen"
        ][0]
        input_json = json.loads(disco_immunogen.input_json_path.read_text())
        assert input_json[0]["sequences"][0]["proteinChain"]["sequence"] == "GHK-----"
        assert "model.N_cycle=4" in disco_immunogen.command
        assert "sample_diffusion.N_step=50" in disco_immunogen.command
        assert any(
            item.startswith("cogen_guidance.immunogen.fixed_fragments=")
            for item in disco_immunogen.command
        )

        pg_standard = [
            run for run in runs
            if run.system == "ProteinGenerator" and run.mode == "standard"
        ][0]
        assert "--n_cycle" in pg_standard.command
        assert "4" in pg_standard.command
        assert "--T" in pg_standard.command
        assert "60" in pg_standard.command


def test_experiment_plan_adds_pg_checkpoint_to_pg_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = root / "seqdiff.pt"
        checkpoint.write_text("placeholder")
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            pg_checkpoint=checkpoint,
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert "--checkpoint" in runs[0].command
        assert str(checkpoint) in runs[0].command


def test_experiment_plan_adds_pythonpath_to_pg_alternating_analysis():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=root / "ProteinGenerator_MOTIFS",
            modes=["alternating"],
            systems=["ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            alternating_list1=["GP"],
            alternating_list2=["DR"],
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert runs[0].analysis_command[:2] == [
            "env",
            f"PYTHONPATH={root / 'ProteinGenerator_MOTIFS' / 'code'}",
        ]
        assert "protein_generator.alternating_motifs.analyze" in runs[0].analysis_command


def test_experiment_plan_enables_disco_trajectory_dump_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert "cogen_guidance.trajectory_dump.enabled=true" in runs[0].command
        assert "cogen_guidance.trajectory_dump.save_final_coordinate=true" in runs[0].command
        assert "cogen_guidance.trajectory_dump.dtype=float16" in runs[0].command


def test_experiment_plan_can_build_direct_pg_immunogen_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        fixed_json = root / "fixed.json"
        seed_pdb = root / "seed.pdb"
        seed_trb = root / "seed.trb"
        fixed_json.write_text(json.dumps({"fixed_fragments": []}) + "\n")
        seed_pdb.write_text("ATOM\n")
        seed_trb.write_text("TRB\n")
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["immunogen"],
            systems=["ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            fixed_fragments=[FixedFragmentPlan(1, "ALL", "seed")],
            pg_seed_pdb=seed_pdb,
            pg_seed_sequence="ALLRAA",
            pg_fixed_fragments_json=fixed_json,
            pg_immunogen_direct=True,
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert "docker" not in runs[0].command
        assert "--sampler" in runs[0].command
        assert "immunogen_repredict_coverage" in runs[0].command
        assert "--trb" in runs[0].command
        assert str(seed_trb) in runs[0].command
        assert "--immunogen_repredict" in runs[0].command
        assert str(fixed_json) in runs[0].analysis_command


def test_experiment_plan_materializes_transition_only_alternating_variant():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["alternating"],
            systems=["DISCO", "ProteinGenerator"],
            variant_set="exploratory",
            seeds=[0],
            alternating_length=12,
            alternating_list1=["GP"],
            alternating_list2=["DR"],
        )

        runs = plan_runs(config)
        transition_runs = [
            run for run in runs
            if run.variant == "transition_only_tail_guard"
        ]
        disco_transition = [
            run for run in transition_runs if run.system == "DISCO"
        ][0]
        pg_transition = [
            run for run in transition_runs if run.system == "ProteinGenerator"
        ][0]

        assert len(transition_runs) == 2
        assert (
            "cogen_guidance.alternating.transition_logit_guidance_enabled=true"
            in disco_transition.command
        )
        assert (
            "cogen_guidance.alternating.logit_guidance_enabled=false"
            in disco_transition.command
        )
        assert "soft_transition_alternating_motifs_bias" in pg_transition.command


def test_experiment_plan_marks_missing_pg_immunogen_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["immunogen"],
            systems=["ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            fixed_fragments=[FixedFragmentPlan(1, "GHK", "bcell")],
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert runs[0].status == "requires_pg_immunogen_seed_inputs"
        assert runs[0].command == []


def test_experiment_plan_marks_missing_disco_immunogen_fragments():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["immunogen"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
        )

        runs = plan_runs(config)

        assert len(runs) == 1
        assert runs[0].status == "requires_immunogen_fixed_fragments"
        assert runs[0].command == []
        assert runs[0].input_json_path is None


def test_fairness_audit_accepts_balanced_standard_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO", "ProteinGenerator"],
            variant_set="core",
            seeds=[0, 1],
            n_cycle=4,
            standard_length=10,
            standard_motifs=["GHK"],
            designs_per_replicate=2,
        )
        plan_runs(config)

        issues, summary = audit_fairness_csv(config.out_dir / "planned_runs.csv")
        report_path = config.out_dir / "fairness_report.json"
        issues_path = config.out_dir / "fairness_issues.csv"
        write_fairness_report(
            report_json_path=report_path,
            issues_csv_path=issues_path,
            issues=issues,
            summary=summary,
        )

        assert summary["status"] == "pass"
        assert summary["error_count"] == 0
        assert summary["warning_count"] == 0
        assert summary["num_groups"] == 1
        assert "DISCO" in summary["groups"][0]["systems"]
        assert "ProteinGenerator" in summary["groups"][0]["systems"]
        assert json.loads(report_path.read_text())["status"] == "pass"
        assert issues_path.exists()


def test_fairness_audit_rejects_unmatched_group_parameters():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO", "ProteinGenerator"],
            variant_set="core",
            seeds=[0],
            n_cycle=4,
            standard_length=10,
            standard_motifs=["GHK"],
        )
        plan_runs(config)
        planned_csv = config.out_dir / "planned_runs.csv"
        rows = list(csv.DictReader(planned_csv.open()))
        fieldnames = list(rows[0].keys())
        for row in rows:
            if row["system"] == "DISCO":
                parameters = json.loads(row["parameters_json"])
                parameters["length"] = 11
                row["parameters_json"] = json.dumps(parameters)
        with planned_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        issues, summary = audit_fairness_csv(planned_csv)
        checks = {issue.check for issue in issues}

        assert summary["status"] == "fail"
        assert "length_matched" in checks


def test_comparison_normalizes_pg_and_disco_rows():
    pg = normalize_pg_row(
        {
            "design_id": "design_000",
            "sequence": "GHKAA",
            "pg_plddt_mean": "0.750",
            "num_unique_motifs_present": "1",
            "num_motif_occurrences_total": "1",
        },
        variant="standard",
        source_path="/pg/summary.csv",
    )
    disco = normalize_disco_row(
        {
            "individual": "0",
            "sequence": "GHKAAKTTKS",
            "confidence": "0.650",
            "unique_motifs": "2",
            "total_motifs": "2",
            "fixed_fragments_preserved": "True",
        },
        variant="standard",
        source_path="/disco/evolution_summary.csv",
    )

    assert pg.system == "ProteinGenerator"
    assert disco.system == "DISCO"
    assert pg.confidence == 0.75
    assert disco.unique_motifs == 2
    assert disco.fixed_preserved is True
    assert pg.score_provenance_status == "pg_sequence_and_plddt_from_same_saved_pdb"
    assert disco.score_provenance_status == "disco_confidence_provenance_unknown"
    assert pg.comparison_score > 0
    assert disco.comparison_score > 0


def test_comparison_normalizes_pg_alternating_columns():
    row = normalize_pg_row(
        {
            "design_id": "alt_000",
            "sequence": "GPAADRAA",
            "pg_plddt_mean": "0.710",
            "num_unique_motifs": "2",
            "num_motif_hits": "2",
            "unique_motifs": "DR,GP",
            "motif_pattern": "1->2",
        },
        mode="alternating",
        variant="alternating_slot_tail_guard",
        source_path="/pg/alternating_summary.csv",
    )

    assert row.mode == "alternating"
    assert row.unique_motifs == 2
    assert row.total_motifs == 2
    assert row.motif_coverage == 1.0
    assert row.comparison_score > 0.8


def test_comparison_normalizes_disco_evolution_count_fields():
    row = normalize_disco_row(
        {
            "individual": "0",
            "sequence": "GHKAAKTTKS",
            "confidence": "0.7",
            "GHK_count": "1",
            "KTTKS_count": "1",
        },
        mode="standard",
        variant="evolution",
        source_path="/disco/evolution_summary.csv",
    )

    assert row.unique_motifs == 2
    assert row.total_motifs == 2
    assert row.motif_coverage == 1.0
    assert row.comparison_score == 0.85


def test_comparison_normalizes_disco_same_artifact_provenance():
    row = normalize_disco_row(
        {
            "design_id": "design_sample_0",
            "sequence": "GHKAA",
            "confidence": "0.75",
            "confidence_match_status": "exact_sample",
            "evaluation_status": "same_sample_complete",
            "unique_motifs": "1",
            "total_motifs": "1",
            "present_motifs": "GHK",
            "missing_motifs": "",
        },
        variant="standard",
        source_path="/disco/analysis.csv",
    )

    assert (
        row.score_provenance_status
        == "disco_exact_sequence_structure_confidence_match"
    )


def test_comparison_summarizes_by_system_and_variant():
    rows = [
        normalize_pg_row(
            {
                "design_id": "a",
                "sequence": "GHK",
                "pg_plddt_mean": "0.8",
                "num_unique_motifs_present": "1",
                "num_motif_occurrences_total": "1",
            },
            variant="std",
            source_path="/pg.csv",
        ),
        normalize_pg_row(
            {
                "design_id": "b",
                "sequence": "GHKGHK",
                "pg_plddt_mean": "0.6",
                "num_unique_motifs_present": "1",
                "num_motif_occurrences_total": "2",
            },
            variant="std",
            source_path="/pg.csv",
        ),
        normalize_disco_row(
            {
                "individual": "0",
                "sequence": "GHKAAKTTKS",
                "confidence": "0.7",
                "unique_motifs": "2",
                "total_motifs": "2",
            },
            variant="std",
            source_path="/disco.csv",
        ),
    ]

    summary = summarize_by_group(rows)
    pg_summary = [row for row in summary if row["system"] == "ProteinGenerator"][0]
    disco_summary = [row for row in summary if row["system"] == "DISCO"][0]

    assert pg_summary["num_designs"] == 2
    assert pg_summary["mean_total_motifs"] == 1.5
    assert pg_summary["best_comparison_design_id"] == "a"
    assert disco_summary["best_unique_motifs"] == 2
    assert disco_summary["confidence_exact_fraction"] == 0


def test_comparison_report_renders_validation_and_top_designs():
    rows = [
        normalize_pg_row(
            {
                "design_id": "pg_best",
                "sequence": "GHKAA",
                "pg_plddt_mean": "0.8",
                "num_unique_motifs_present": "1",
                "num_motif_occurrences_total": "1",
                "motifs_present": "GHK",
                "motifs_missing": "",
            },
            mode="standard",
            variant="std",
            source_path="/pg.csv",
        ),
        normalize_disco_row(
            {
                "design_id": "disco_best",
                "sequence": "GHKAAKTTKS",
                "confidence": "0.7",
                "confidence_metric": "ranking_score",
                "confidence_match_status": "exact_sample",
                "unique_motifs": "2",
                "total_motifs": "2",
                "present_motifs": "GHK,KTTKS",
            },
            mode="standard",
            variant="std",
            source_path="/disco.csv",
        ),
    ]
    summary = summarize_by_group(rows)
    rendered = render_comparison_report(
        normalized_rows=[row.as_dict() for row in rows],
        summary_rows=summary,
        validation_report={
            "status": "pass",
            "error_count": 0,
            "warning_count": 0,
            "completed_summary_count": 2,
            "expected_summary_count": 2,
        },
    )

    assert "Status: `pass`" in rendered
    assert "disco_best" in rendered
    assert "score_provenance_status" in rendered


def test_write_comparison_report_from_csvs():
    rows = [
        normalize_pg_row(
            {
                "design_id": "pg_best",
                "sequence": "GHKAA",
                "pg_plddt_mean": "0.8",
                "num_unique_motifs_present": "1",
                "num_motif_occurrences_total": "1",
                "motifs_present": "GHK",
            },
            mode="standard",
            variant="std",
            source_path="/pg.csv",
        )
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        normalized_path = tmp / "normalized.csv"
        summary_path = tmp / "summary.csv"
        report_path = tmp / "report.md"
        validation_path = tmp / "validation.json"
        write_normalized_rows(normalized_path, rows)
        write_summary_rows(summary_path, summarize_by_group(rows))
        validation_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "error_count": 0,
                    "warning_count": 0,
                    "completed_summary_count": 1,
                    "expected_summary_count": 1,
                }
            )
            + "\n"
        )

        write_comparison_report(
            normalized_csv=normalized_path,
            summary_csv=summary_path,
            out_md=report_path,
            validation_report_json=validation_path,
            title="Synthetic Report",
        )

        text = report_path.read_text()
        assert "# Synthetic Report" in text
        assert "pg_best" in text


def test_quality_validation_accepts_clean_disco_analysis_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        diagnostics_path = tmp / "trace.jsonl"
        structure_path = tmp / "design.pdb"
        diagnostics_path.write_text(json.dumps({"step": 0}) + "\n")
        structure_path.write_text(pdb_ca_text([(0, 0, 0)]))
        summary_path = tmp / "analysis.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "clean",
                    "sequence": "GHKAAKTTKS",
                    "confidence": "0.75",
                    "confidence_match_status": "exact_sample",
                    "present_motifs": "GHK,KTTKS",
                    "missing_motifs": "",
                    "unique_motifs": "2",
                    "total_motifs": "2",
                    "fixed_fragments_preserved": "True",
                    "num_fixed_mismatches": "0",
                    "longest_homopolymer": "2",
                    "aa_entropy_norm": "0.75",
                    "ca_rmsd_to_seed": "0.5",
                    "motif_spread_score": "0.8",
                    "diagnostics_jsonl": str(diagnostics_path),
                    "structure_path": str(structure_path),
                    "evaluation_status": "same_sample_complete",
                }
            ],
        )

        issues, summary = validate_summary_csv(
            summary_csv=summary_path,
            config=QualityGateConfig(
                require_fixed_preserved=True,
                require_exact_confidence=True,
                require_diagnostics=True,
                require_structure=True,
                require_same_evaluation_artifact=True,
                max_rmsd=1.0,
                min_spread_score=0.5,
            ),
        )

        assert issues == []
    assert summary["status"] == "pass"
    assert summary["passed_rows"] == 1


def test_quality_validation_rejects_mixed_evaluation_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "analysis.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "mixed",
                    "sequence": "GHKAA",
                    "confidence": "0.75",
                    "confidence_match_status": "single_confidence_fallback",
                    "present_motifs": "GHK",
                    "missing_motifs": "",
                    "unique_motifs": "1",
                    "total_motifs": "1",
                    "longest_homopolymer": "2",
                    "aa_entropy_norm": "0.75",
                    "evaluation_status": "sample_mismatch_or_fallback",
                }
            ],
        )

        issues, summary = validate_summary_csv(
            summary_csv=summary_path,
            config=QualityGateConfig(
                require_same_evaluation_artifact=True,
                require_exact_confidence=False,
                require_structure=False,
                min_confidence=0.0,
            ),
        )

        assert summary["status"] == "fail"
        assert {issue.check for issue in issues} == {"same_evaluation_artifact"}


def test_immunogen_validation_accepts_category_complete_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "immunogen.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "immunogen_clean",
                    "sequence": "GHKAAKTTKS",
                    "fixed_fragments_preserved": "True",
                    "num_fixed_mismatches": "0",
                    "tcell_unique_motifs": "2",
                    "tcell_total_motifs": "2",
                    "tcell_missing_motifs": "",
                    "cleavage_unique_motifs": "1",
                    "cleavage_total_motifs": "1",
                    "cleavage_missing_motifs": "",
                    "evaluation_status": "same_sample_complete",
                }
            ],
        )

        issues, summary = validate_immunogen_summary_csv(
            summary_csv=summary_path,
            config=ImmunogenValidationConfig(
                tcell_motifs=["GHK", "KTTKS"],
                cleavage_motifs=["AAK"],
                require_same_evaluation_artifact=True,
            ),
        )

        assert issues == []
        assert summary["status"] == "pass"
        assert summary["fixed_passed_rows"] == 1
        assert summary["tcell_passed_rows"] == 1
        assert summary["cleavage_passed_rows"] == 1


def test_immunogen_validation_reports_category_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "immunogen.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "immunogen_bad",
                    "sequence": "AAAAA",
                    "fixed_fragments_preserved": "False",
                    "num_fixed_mismatches": "1",
                    "tcell_unique_motifs": "1",
                    "tcell_total_motifs": "1",
                    "tcell_missing_motifs": "KTTKS",
                    "cleavage_unique_motifs": "0",
                    "cleavage_total_motifs": "0",
                    "cleavage_missing_motifs": "AAK",
                    "evaluation_status": "sample_mismatch_or_fallback",
                }
            ],
        )

        issues, summary = validate_immunogen_summary_csv(
            summary_csv=summary_path,
            config=ImmunogenValidationConfig(
                tcell_motifs=["GHK", "KTTKS"],
                cleavage_motifs=["AAK"],
                require_same_evaluation_artifact=True,
            ),
        )

        checks = {item.check for item in issues}
        categories = {item.category for item in issues}
        assert summary["status"] == "fail"
        assert "fixed_fragments_preserved" in checks
        assert "fixed_fragment_mismatches" in checks
        assert "tcell_coverage" in checks
        assert "cleavage_coverage" in checks
        assert "same_evaluation_artifact" in checks
        assert categories == {"bcell_fixed", "tcell", "cleavage", "evaluation"}


def test_immunogen_validation_report_writes_json_and_issue_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "immunogen.csv"
        report_path = tmp / "report.json"
        issues_path = tmp / "issues.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "immunogen_bad",
                    "fixed_fragments_preserved": "False",
                    "num_fixed_mismatches": "1",
                }
            ],
        )
        issues, summary = validate_immunogen_summary_csv(
            summary_csv=summary_path,
            config=ImmunogenValidationConfig(),
        )

        write_immunogen_validation_report(
            report_json_path=report_path,
            issues_csv_path=issues_path,
            issues=issues,
            summary=summary,
        )

        assert json.loads(report_path.read_text())["status"] == "fail"
        assert "bcell_fixed" in issues_path.read_text()


def test_quality_validation_reports_threshold_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "bad.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "bad",
                    "sequence": "AAAAAA",
                    "confidence": "0.4",
                    "confidence_match_status": "missing",
                    "present_motifs": "GHK",
                    "missing_motifs": "KTTKS",
                    "unique_motifs": "1",
                    "fixed_fragments_preserved": "False",
                    "num_fixed_mismatches": "1",
                    "longest_homopolymer": "6",
                    "aa_entropy_norm": "0.1",
                    "ca_rmsd_to_seed": "5.0",
                    "motif_spread_score": "0.1",
                }
            ],
        )

        issues, summary = validate_summary_csv(
            summary_csv=summary_path,
            config=QualityGateConfig(
                min_confidence=0.6,
                min_motif_coverage=1.0,
                min_unique_motifs=2,
                max_homopolymer=4,
                min_entropy=0.5,
                max_rmsd=2.0,
                min_spread_score=0.5,
                require_fixed_preserved=True,
                require_exact_confidence=True,
            ),
        )

        checks = {issue.check for issue in issues}
        assert summary["status"] == "fail"
        assert "min_confidence" in checks
        assert "exact_confidence_provenance" in checks
        assert "min_motif_coverage" in checks
        assert "min_unique_motifs" in checks
        assert "fixed_fragments_preserved" in checks
        assert "fixed_fragment_mismatches" in checks
        assert "max_homopolymer" in checks
        assert "min_entropy" in checks
        assert "max_rmsd" in checks
        assert "min_spread_score" in checks


def test_quality_validation_report_writes_json_and_issue_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "bad.csv"
        write_csv_rows(
            summary_path,
            [
                {
                    "design_id": "bad",
                    "sequence": "AAAA",
                    "confidence": "0.1",
                    "present_motifs": "",
                    "missing_motifs": "GHK",
                    "longest_homopolymer": "4",
                    "aa_entropy_norm": "0.1",
                }
            ],
        )
        issues, summary = validate_summary_csv(
            summary_csv=summary_path,
            config=QualityGateConfig(),
        )
        report_path = tmp / "quality.json"
        issues_path = tmp / "quality_issues.csv"

        write_quality_report(
            report_json_path=report_path,
            issues_csv_path=issues_path,
            issues=issues,
            summary=summary,
        )

        assert json.loads(report_path.read_text())["status"] == "fail"
        assert "min_confidence" in issues_path.read_text()


def test_external_quality_score_uses_pg_rank_style_components():
    rows = annotate_external_quality_rows(
        [
            {
                "design_id": "A",
                "confidence": "0.80",
                "motif_score": "0.50",
                "esm2_mean_token_probability": "0.90",
                "ca_rmsd_to_reference": "2.0",
                "rosetta_score_per_residue": "-3.0",
            },
            {
                "design_id": "B",
                "confidence": "0.60",
                "motif_score": "1.00",
                "esm2_mean_token_probability": "0.50",
                "ca_rmsd_to_reference": "10.0",
                "rosetta_score_per_residue": "-1.0",
            },
        ]
    )
    by_id = {row["design_id"]: row for row in rows}

    assert by_id["A"]["external_metric_status"] == "complete"
    assert by_id["B"]["external_metric_status"] == "complete"
    assert by_id["A"]["esm2_rank_score"] == "1.000000"
    assert by_id["A"]["structure_consistency_score"] == "1.000000"
    assert by_id["A"]["rosetta_rank_score"] == "1.000000"
    assert by_id["A"]["posthoc_quality_score"] == "0.890000"
    assert by_id["B"]["posthoc_quality_score"] == "0.580000"


def test_external_quality_outputs_merge_metrics_and_report_missing_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary_csv = tmp / "summary.csv"
        external_csv = tmp / "external.csv"
        out_csv = tmp / "augmented.csv"
        report_json = tmp / "report.json"
        top_csv = tmp / "top.csv"
        write_csv_rows(
            summary_csv,
            [
                {
                    "design_id": "has_external",
                    "confidence": "0.70",
                    "motif_score": "1.0",
                },
                {
                    "design_id": "summary_only",
                    "confidence": "0.90",
                    "motif_score": "0.5",
                },
            ],
        )
        write_csv_rows(
            external_csv,
            [
                {
                    "design_id": "has_external",
                    "esm2_perplexity": "4.0",
                    "ca_rmsd_to_reference": "3.0",
                }
            ],
        )

        report = write_external_quality_outputs(
            summary_csv=summary_csv,
            external_metrics_csv=external_csv,
            out_csv=out_csv,
            report_json=report_json,
            top_csv=top_csv,
        )
        rows = list(csv.DictReader(out_csv.open()))
        by_id = {row["design_id"]: row for row in rows}

        assert report["num_rows"] == 2
        assert report["status_counts"]["partial"] == 1
        assert report["status_counts"]["no_external_metrics"] == 1
        assert by_id["has_external"]["esm2_rank_source"] == "esm2_perplexity"
        assert by_id["has_external"]["external_metric_missing"] == "rosetta"
        assert by_id["summary_only"]["external_metric_missing"] == "esm2,rmsd,rosetta"
        assert "posthoc_quality_score" in top_csv.read_text()


def test_comparison_validation_accepts_completed_disco_summary_with_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = ComparisonPlanConfig(
            out_dir=root / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
            standard_length=10,
            standard_motifs=["GHK"],
        )
        runs = plan_runs(config)
        run = runs[0]
        diagnostics_path = run.run_dir / "diagnostics" / "trace.jsonl"
        diagnostics_path.parent.mkdir(parents=True)
        diagnostics_path.write_text(json.dumps({"step": 0}) + "\n")
        write_csv_rows(
            run.summary_csv_path,
            [
                {
                    "design_id": "design_sample_0",
                    "sequence": "GHKAAAAAAA",
                    "confidence": "0.75",
                    "confidence_match_status": "exact_sample",
                    "unique_motifs": "1",
                    "total_motifs": "1",
                    "diagnostics_jsonl": str(diagnostics_path),
                }
            ],
        )

        issues, summary = validate_comparison_plan(
            planned_runs_csv=config.out_dir / "planned_runs.csv",
            require_diagnostics=True,
            required_systems=["DISCO"],
        )

        assert summary["status"] == "pass"
        assert summary["completed_summary_count"] == 1
        assert [item for item in issues if item.severity == "error"] == []


def test_comparison_validation_allows_missing_summaries_for_planned_matrix():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
            standard_length=10,
            standard_motifs=["GHK"],
        )
        plan_runs(config)

        issues, summary = validate_comparison_plan(
            planned_runs_csv=config.out_dir / "planned_runs.csv",
            allow_missing_summaries=True,
            required_systems=["DISCO"],
        )

        assert summary["status"] == "pass"
        assert summary["warning_count"] == 1
        assert issues[0].check == "summary_exists"


def test_comparison_validation_requires_both_systems_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
            standard_length=10,
            standard_motifs=["GHK"],
        )
        plan_runs(config)

        issues, summary = validate_comparison_plan(
            planned_runs_csv=config.out_dir / "planned_runs.csv",
            allow_missing_summaries=True,
        )

        checks = {item.check for item in issues}
        assert summary["status"] == "fail"
        assert "required_systems" in checks


def test_comparison_validation_rejects_unmatched_disco_confidence_and_length():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
            standard_length=10,
            standard_motifs=["GHK"],
        )
        run = plan_runs(config)[0]
        write_csv_rows(
            run.summary_csv_path,
            [
                {
                    "design_id": "design_sample_0",
                    "sequence": "GHK",
                    "confidence": "0.75",
                    "confidence_match_status": "missing",
                }
            ],
        )

        issues, summary = validate_comparison_plan(
            planned_runs_csv=config.out_dir / "planned_runs.csv",
            required_systems=["DISCO"],
        )

        checks = {item.check for item in issues}
        assert summary["status"] == "fail"
        assert "sequence_length" in checks
        assert "disco_confidence_match" in checks


def test_comparison_validation_report_writes_json_and_issue_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ComparisonPlanConfig(
            out_dir=Path(tmpdir) / "plan",
            disco_root=Path("/repo/DISCO"),
            pg_root=Path("/repo/ProteinGenerator_MOTIFS"),
            modes=["standard"],
            systems=["DISCO"],
            variant_set="core",
            seeds=[0],
            standard_length=10,
            standard_motifs=["GHK"],
        )
        plan_runs(config)
        issues, summary = validate_comparison_plan(
            planned_runs_csv=config.out_dir / "planned_runs.csv",
            allow_missing_summaries=True,
            required_systems=["DISCO"],
        )
        report_path = Path(tmpdir) / "validation.json"
        issues_path = Path(tmpdir) / "issues.csv"

        write_validation_report(
            report_json_path=report_path,
            issues_csv_path=issues_path,
            issues=issues,
            summary=summary,
        )

        assert json.loads(report_path.read_text())["status"] == "pass"
        assert "summary_exists" in issues_path.read_text()


def test_trajectory_rows_track_deltas_and_best_step():
    records = [
        {
            "step": 3,
            "guidance_sequence": "AAAAAA",
            "guidance_combined_score": 0.1,
            "guidance_motif_score": 0.0,
            "guidance_num_unique": 0,
            "guidance_num_total": 0,
            "guidance_present": "",
            "guidance_missing": "GHK",
            "num_newly_unmasked": 1,
        },
        {
            "step": 2,
            "guidance_sequence": "AAGHKA",
            "guidance_combined_score": 1.4,
            "guidance_motif_score": 1.0,
            "guidance_num_unique": 1,
            "guidance_num_total": 1,
            "guidance_present": "GHK",
            "guidance_missing": "",
            "guidance_best_updated": True,
            "num_newly_unmasked": 3,
        },
        {
            "step": 1,
            "guidance_sequence": "AAGHAA",
            "guidance_combined_score": 0.8,
            "guidance_motif_score": 0.5,
            "guidance_num_unique": 0,
            "guidance_num_total": 0,
            "guidance_present": "",
            "guidance_missing": "GHK",
            "guidance_best_output_selected": True,
        },
    ]

    rows = trajectory_rows(records, sample_id="sample_a")
    summary = trajectory_summary(rows, sample_id="sample_a")

    assert rows[0]["combined_score_delta"] == ""
    assert rows[1]["combined_score_delta"] == 1.3
    assert rows[1]["sequence_changes_from_previous"] == 3
    assert rows[1]["new_present_motifs"] == "GHK"
    assert rows[1]["is_best_combined"] is True
    assert rows[2]["lost_present_motifs"] == "GHK"
    assert rows[2]["best_step_so_far"] == 2
    assert summary["best_step"] == 2
    assert summary["final_step"] == 1
    assert summary["best_beats_final"] is True
    assert summary["best_output_selected"] is True


def test_write_trajectory_outputs_writes_csv_and_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        diagnostics_path = tmp / "sample_b_guidance_diagnostics.jsonl"
        out_csv = tmp / "trajectory.csv"
        out_json = tmp / "trajectory_summary.json"
        write_jsonl_rows(
            diagnostics_path,
            [
                {
                    "step": 4,
                    "guidance_sequence": "AAAA",
                    "guidance_combined_score": 0.2,
                    "guidance_num_unique": 0,
                    "guidance_num_total": 0,
                    "guidance_present": "",
                    "guidance_missing": "LL",
                    "entropy_mean": 2.1,
                },
                {
                    "step": 3,
                    "guidance_sequence": "AALL",
                    "guidance_combined_score": 1.2,
                    "guidance_num_unique": 1,
                    "guidance_num_total": 1,
                    "guidance_present": "LL",
                    "guidance_missing": "",
                    "entropy_mean": 1.7,
                },
            ],
        )

        summary = write_trajectory_outputs(
            diagnostics_jsonl_paths=[diagnostics_path],
            out_csv=out_csv,
            out_summary_json=out_json,
        )
        csv_rows = list(csv.DictReader(out_csv.open()))
        saved_summary = json.loads(out_json.read_text())

        assert summary["aggregate"]["total_trajectory_steps"] == 2
        assert csv_rows[0]["sample_id"] == "sample_b"
        assert csv_rows[1]["present_motifs"] == "LL"
        assert csv_rows[1]["is_best_combined"] == "True"
        assert saved_summary["samples"][0]["best_sequence"] == "AALL"


def test_write_coordinate_trajectory_outputs_saves_tensors_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prediction = {
            "x1_denoised": torch.arange(3 * 2 * 4 * 3, dtype=torch.float32).reshape(
                3,
                2,
                4,
                3,
            ),
            "xt_trajectory": torch.ones(3, 2, 4, 3),
            "trajectory_noise_levels": torch.tensor([1.0, 0.5, 0.1]),
            "coordinate": torch.zeros(2, 4, 3),
        }

        result = write_coordinate_trajectory_outputs(
            trajectory_dir=tmp / "trajectories",
            sample_id="sample_a",
            prediction=prediction,
            save_final_coordinate=True,
            dtype_name="float16",
            max_steps=2,
        )
        manifest = json.loads(Path(result["trajectory_manifest"]).read_text())
        saved = {
            item["key"]: item
            for item in manifest["saved_tensors"]
        }

        assert result["trajectory_num_saved_tensors"] == 3
        assert manifest["trajectory_noise_levels"] == [1.0, 0.5]
        assert manifest["truncated_keys"] == ["x1_denoised", "xt_trajectory"]
        assert saved["x1_denoised"]["shape"] == [2, 2, 4, 3]
        assert saved["x1_denoised"]["dtype"] == "float16"
        assert Path(saved["x1_denoised"]["path"]).exists()
        assert torch.load(saved["x1_denoised"]["path"]).shape == (2, 2, 4, 3)


def test_run_bundle_collects_summary_diagnostics_and_hashes():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_dir = tmp / "run"
        dump_dir = tmp / "dump"
        diagnostics_dir = dump_dir / "diagnostics"
        sequences_dir = dump_dir / "sequences"
        pdbs_dir = dump_dir / "pdbs"
        trajectories_dir = dump_dir / "trajectories"
        run_dir.mkdir()
        diagnostics_dir.mkdir(parents=True)
        sequences_dir.mkdir(parents=True)
        pdbs_dir.mkdir(parents=True)
        trajectories_dir.mkdir(parents=True)

        (run_dir / "command.txt").write_text("python runner/inference.py\n")
        (run_dir / "analysis_command.txt").write_text(
            "python runner/guidance_analyze_disco.py\n"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "system": "DISCO",
                    "mode": "standard",
                    "variant": "standard_gradient_coverage",
                    "parameters": {"n_cycle": 4, "motifs": ["GHK"]},
                }
            )
            + "\n"
        )
        summary_csv = run_dir / "summary.csv"
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "design_id",
                    "system",
                    "mode",
                    "variant",
                    "confidence",
                    "motif_score",
                    "comparison_score",
                    "score_provenance_status",
                    "evaluation_status",
                    "confidence_match_status",
                    "runtime_seconds",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "design_id": "sample_0",
                    "system": "DISCO",
                    "mode": "standard",
                    "variant": "standard_gradient_coverage",
                    "confidence": "0.82",
                    "motif_score": "1.0",
                    "comparison_score": "0.91",
                    "score_provenance_status": (
                        "disco_exact_sequence_structure_confidence_match"
                    ),
                    "evaluation_status": "same_sample_complete",
                    "confidence_match_status": "exact_sample",
                    "runtime_seconds": "12.5",
                }
            )
        summary_csv.with_suffix(".manifest.json").write_text(
            json.dumps({"script_name": "guidance_analyze_disco.py", "num_rows": 1})
            + "\n"
        )
        (diagnostics_dir / "sample_0_run_metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": "sample_0",
                    "runtime_seconds": 12.5,
                    "num_diagnostic_records": 2,
                    "best_output_selected": True,
                    "diagnostics_jsonl": str(
                        diagnostics_dir / "sample_0_guidance_diagnostics.jsonl"
                    ),
                    "diagnostics_csv": str(
                        diagnostics_dir / "sample_0_guidance_diagnostics.csv"
                    ),
                }
            )
            + "\n"
        )
        (diagnostics_dir / "sample_0_guidance_diagnostics.jsonl").write_text(
            '{"step": 1, "guidance_combined_score": 1.0}\n'
        )
        (diagnostics_dir / "sample_0_guidance_diagnostics.csv").write_text(
            "step,guidance_combined_score\n1,1.0\n"
        )
        (sequences_dir / "sample_0.txt").write_text(">cogen_seq 0\nGHKAA\n")
        (pdbs_dir / "sample_0.pdb").write_text("HEADER SAMPLE\n")
        torch.save(torch.zeros(2, 3), trajectories_dir / "sample_0_x1_denoised.pt")
        (trajectories_dir / "sample_0_trajectory_manifest.json").write_text(
            json.dumps({"sample_id": "sample_0"}) + "\n"
        )

        bundle = write_run_bundle(
            RunBundleConfig(
                run_dir=run_dir,
                out_json=run_dir / "run_bundle.json",
                system="DISCO",
                mode="standard",
                variant="standard_gradient_coverage",
                replicate="0",
                seed="0",
                dump_dir=dump_dir,
                summary_csv=summary_csv,
                include_git=False,
            )
        )

        assert bundle["summary_csv"]["num_rows"] == 1
        assert any(
            column["column"] == "comparison_score" and column["max"] == 0.91
            for column in bundle["summary_csv"]["numeric_columns"]
        )
        assert bundle["diagnostics"]["num_metadata_records"] == 1
        assert bundle["diagnostics"]["total_runtime_seconds"] == 12.5
        assert bundle["artifact_files"]["summary_csv"]["sha256_status"] == "computed"
        assert bundle["artifact_files"]["sequences"]["num_files"] == 1
        assert bundle["artifact_files"]["structures"]["num_files"] == 1
        assert bundle["artifact_files"]["trajectories"]["num_files"] == 2
        assert validate_run_bundle(
            bundle,
            require_summary=True,
            require_diagnostics=True,
            require_plan=True,
        ) == []


def test_run_bundle_validation_reports_missing_required_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_dir = tmp / "run"
        run_dir.mkdir()

        bundle = create_run_bundle(
            RunBundleConfig(
                run_dir=run_dir,
                out_json=run_dir / "run_bundle.json",
                include_git=False,
            )
        )
        issues = validate_run_bundle(
            bundle,
            require_summary=True,
            require_diagnostics=True,
            require_plan=True,
        )
        checks = {issue.check for issue in issues}

        assert "planned_manifest" in checks
        assert "summary_csv_exists" in checks
        assert "diagnostics_dir_exists" in checks


def test_reprediction_plan_writes_fixed_sequence_input_and_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        analysis_csv = tmp / "analysis.csv"
        write_csv_rows(
            analysis_csv,
            [
                {
                    "design_id": "candidate A",
                    "sequence": "GHKAAKTT",
                    "confidence": "0.83",
                    "motif_score": "1.0",
                    "comparison_score": "0.91",
                    "fixed_fragments_preserved": "true",
                    "evaluation_status": "same_sample_complete",
                    "confidence_match_status": "exact_sample",
                }
            ],
        )

        manifest = write_reprediction_plan(
            RepredictionPlanConfig(
                analysis_csv=analysis_csv,
                out_dir=tmp / "repredict",
                disco_root=Path(__file__).resolve().parents[1],
                motifs=["GHK", "KTT"],
                seeds=[7],
                n_cycle=4,
                n_step=12,
                python=".venv/bin/python",
                require_fixed_preserved=True,
                require_same_evaluation_artifact=True,
                fixed_fragments=[FixedFragmentPlan(1, "GHK", "bcell")],
                tcell_motifs=["AAK"],
                cleavage_motifs=["TT"],
            )
        )

        assert manifest["num_planned"] == 1
        plan_csv = Path(manifest["planned_csv"])
        rows = list(csv.DictReader(plan_csv.open()))
        assert rows[0]["status"] == "planned"
        assert "cogen_guidance.enabled=false" in rows[0]["command"]
        assert "--fixed-fragment 1:GHK:bcell" in rows[0]["analysis_command"]
        input_json = json.loads(Path(rows[0]["input_json_path"]).read_text())
        sequence = input_json[0]["sequences"][0]["proteinChain"]["sequence"]
        assert sequence == "GHKAAKTT"
        assert "-" not in sequence


def test_reprediction_plan_filters_and_ranks_designs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        analysis_csv = tmp / "analysis.csv"
        write_csv_rows(
            analysis_csv,
            [
                {
                    "design_id": "low",
                    "sequence": "GHKAAAAA",
                    "confidence": "0.80",
                    "motif_score": "0.40",
                    "comparison_score": "0.50",
                    "fixed_fragments_preserved": "true",
                    "evaluation_status": "same_sample_complete",
                },
                {
                    "design_id": "best",
                    "sequence": "GHKTTAAA",
                    "confidence": "0.90",
                    "motif_score": "1.0",
                    "comparison_score": "0.95",
                    "fixed_fragments_preserved": "true",
                    "evaluation_status": "same_sample_complete",
                },
                {
                    "design_id": "bad_fixed",
                    "sequence": "GHKTTCCC",
                    "confidence": "0.92",
                    "motif_score": "1.0",
                    "comparison_score": "0.99",
                    "fixed_fragments_preserved": "false",
                    "evaluation_status": "same_sample_complete",
                },
            ],
        )

        manifest = write_reprediction_plan(
            RepredictionPlanConfig(
                analysis_csv=analysis_csv,
                out_dir=tmp / "repredict",
                disco_root=Path(__file__).resolve().parents[1],
                motifs=["GHK", "TT"],
                seeds=parse_seed_list("1,2"),
                max_designs=1,
                min_motif_score=0.8,
                require_fixed_preserved=True,
            )
        )

        rows = list(csv.DictReader(Path(manifest["planned_csv"]).open()))
        by_id = {row["design_id"]: row for row in rows}

        assert manifest["num_planned"] == 1
        assert by_id["best"]["status"] == "planned"
        assert by_id["low"]["reason"] == "below_min_motif_score"
        assert by_id["bad_fixed"]["reason"] == "fixed_fragments_not_preserved"


def test_scaffold_plan_writes_fixed_fragment_input_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixed_json = tmp / "fixed_fragments.json"
        fixed_json.write_text(
            json.dumps(
                {
                    "fixed_fragments": [
                        {"start_1based": 3, "sequence": "GHK", "label": "bcell"}
                    ]
                }
            )
            + "\n"
        )

        result = write_scaffold_plan(
            input_json_path=tmp / "input.json",
            manifest_json_path=tmp / "manifest.json",
            length=10,
            name="scaffold_test",
            fixed_fragments=fixed_fragments_from_json(fixed_json),
        )

        assert result.status == "ready_sequence_scaffold"
        assert result.sequence == "--GHK-----"
        input_json = json.loads((tmp / "input.json").read_text())
        assert input_json[0]["sequences"][0]["proteinChain"]["sequence"] == "--GHK-----"
        manifest = json.loads((tmp / "manifest.json").read_text())
        assert manifest["num_fixed_positions"] == 3
        assert manifest["num_designable_positions"] == 7


def test_scaffold_plan_fixed_range_uses_seed_sequence():
    sequence = sequence_from_scaffold_constraints(
        length=12,
        seed_sequence="AAAACCCCGGGG",
        fixed_fragments=[FixedFragmentPlan(2, "AA", "literal")],
        fixed_ranges=[parse_fixed_range("9:12:tail")],
    )

    assert sequence == "-AA-----GGGG"


def test_scaffold_plan_reports_structural_contig_unsupported():
    issues = audit_pg_scaffold_arguments(
        pg_seed_pdb=Path("seed.pdb"),
        pg_contigs=["A1-20,0"],
        pg_inpaint_seq=["A1-20"],
        pg_inpaint_str=[],
    )

    codes = {issue.code for issue in issues}
    assert "pg_seed_pdb_structure_template_not_ported" in codes
    assert "pg_contig_structure_mask_not_ported" in codes
    assert any(issue.severity == "unsupported" for issue in issues)


def test_parity_audit_reports_core_ported_and_open_features():
    disco_root = Path(__file__).resolve().parents[1]

    records, summary = audit_parity(disco_root=disco_root)
    by_id = {record["feature_id"]: record for record in records}

    assert summary["num_features"] >= 25
    assert summary["missing_disco_evidence_count"] == 0
    assert summary["missing_test_evidence_count"] == 0
    assert by_id["generic_hard_motif_scoring"]["status"] == "ported"
    assert by_id["generic_hard_motif_scoring"]["disco_evidence_ok"]
    assert by_id["sequence_motifs_logit_guidance"]["status"] == "ported"
    assert by_id["immunogen_mutable_window_guidance"]["status"] == "ported"
    assert by_id["evolutionary_outer_loop"]["status"] == "partial"
    assert by_id["structural_seed_contig_scaffolding"]["status"] == "partial"
    assert by_id["ligand_aware_guidance"]["status"] == "not_ported"


def test_write_parity_report_outputs_json_and_markdown():
    disco_root = Path(__file__).resolve().parents[1]
    records, summary = audit_parity(disco_root=disco_root)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        report_json = tmp / "parity.json"
        report_md = tmp / "parity.md"

        write_parity_report(
            report_json_path=report_json,
            report_md_path=report_md,
            records=records,
            summary=summary,
        )

        payload = json.loads(report_json.read_text())
        markdown = report_md.read_text()

        assert payload["summary"]["num_features"] == summary["num_features"]
        assert any(
            item["feature_id"] == "save_best_combined"
            for item in payload["features"]
        )
        assert "# DISCO ProteinGenerator Parity Audit" in markdown
        assert "sequence_motifs_logit_guidance" in markdown


def test_artifact_provenance_writes_per_saved_intermediate_state_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        structure = tmp / "pdbs" / "sample_0.pdb"
        sequence = tmp / "sequences" / "sample_0.txt"
        confidence = tmp / "pdbs" / "sample_0_summary_confidence_sample_0.json"
        diagnostics_manifest = tmp / "diagnostics" / "sample_0_run_metadata.json"
        trajectory_manifest = tmp / "trajectories" / "sample_0_trajectory_manifest.json"
        for path in [structure, sequence, confidence, diagnostics_manifest, trajectory_manifest]:
            path.parent.mkdir(parents=True, exist_ok=True)
        structure.write_text("ATOM\n")
        sequence.write_text(">cogen_seq 0\nGHKAAA\n")
        confidence.write_text(json.dumps({"ranking_score": 0.75}) + "\n")
        diagnostics_manifest.write_text(
            json.dumps(
                {
                    "best_step": 1,
                    "best_score": 0.8,
                    "final_step": 0,
                    "final_sequence": "GHKAAA",
                }
            )
            + "\n"
        )
        trajectory_manifest.write_text(
            json.dumps(
                {
                    "saved_tensors": [
                        {"key": "x1_denoised", "path": str(tmp / "x1.pt")},
                        {"key": "xt_trajectory", "path": str(tmp / "xt.pt")},
                    ]
                }
            )
            + "\n"
        )

        result = write_artifact_provenance(
            provenance_dir=tmp / "provenance",
            sample_id="sample_0",
            sample_name="sample",
            seed=0,
            structure_path=structure,
            sequence_path=sequence,
            confidence_paths=[str(confidence)],
            diagnostics_manifest=diagnostics_manifest,
            diagnostic_records=[
                {
                    "step": 1,
                    "guidance_sequence": "GHKAAA",
                    "guidance_motif_score": 1.0,
                    "guidance_combined_score": 0.8,
                },
                {
                    "step": 0,
                    "guidance_sequence": "GHKAAA",
                    "guidance_motif_score": 1.0,
                    "guidance_combined_score": 0.8,
                },
            ],
            trajectory_manifest=trajectory_manifest,
        )

        rows = list(csv.DictReader(Path(result["intermediate_state_index"]).open()))
        manifest = json.loads(Path(result["artifact_provenance_manifest"]).read_text())

        assert result["num_intermediate_states"] == 4
        assert {row["structure_state_kind"] for row in rows} == {
            "x1_denoised",
            "xt_trajectory",
        }
        assert all(
            row["structure_confidence_status"]
            == "not_computed_for_intermediate_state"
            for row in rows
        )
        assert manifest["final"]["confidence_metric"] == "ranking_score"
        assert manifest["final"]["confidence_value"] == 0.75
