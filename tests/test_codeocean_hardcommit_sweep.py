from pathlib import Path
from types import SimpleNamespace

from runner.guidance_codeocean_hardcommit_sweep import (
    DEFAULT_MOTIFS,
    build_inference_command,
    choose_motif_subset,
    hydra_list,
    ideal_hard_commit_starts_1based,
    parse_int_list,
    plan_runs,
)


def test_parse_int_list_supports_ranges_and_commas():
    assert parse_int_list("0-2,5,8-6") == [0, 1, 2, 5, 8, 7, 6]


def test_motif_subsets_have_requested_size_and_full_coverage():
    motifs = list(DEFAULT_MOTIFS)
    for target_count in [6, 8, 10, 12, 15]:
        observed = set()
        for seed_index in range(20):
            subset = choose_motif_subset(motifs, target_count, seed_index)
            assert len(subset) == target_count
            assert len(set(subset)) == target_count
            observed.update(subset)
        assert observed == set(motifs)


def test_hard_commit_position_plan_has_no_overlaps():
    motifs = choose_motif_subset(list(DEFAULT_MOTIFS), 15, 0)
    starts = ideal_hard_commit_starts_1based(motifs, 80)
    claimed = set()
    for motif, start_1based in zip(motifs, starts, strict=True):
        positions = set(range(start_1based - 1, start_1based - 1 + len(motif)))
        assert positions.isdisjoint(claimed)
        claimed.update(positions)
    assert min(claimed) >= 0
    assert max(claimed) < 80


def test_command_uses_exact_subset_and_hard_commit_overrides():
    motifs = choose_motif_subset(list(DEFAULT_MOTIFS), 6, 0)
    command = build_inference_command(
        python_executable="/venv/bin/python",
        disco_root=Path("/repo/DISCO"),
        experiment="designable",
        input_json_path=Path("/tmp/input.json"),
        dump_dir=Path("/tmp/dump"),
        seed=7,
        motifs=motifs,
        length=80,
        n_step=50,
        n_cycle=4,
        position_offset=3,
    )
    text = " ".join(command)
    assert "seeds=[7]" in text
    assert "sample_diffusion.N_step=50" in text
    assert "model.N_cycle=4" in text
    assert f"cogen_guidance.motifs={hydra_list(motifs)}" in text
    assert "cogen_guidance.logit_guidance.hard_commit_enabled=true" in text
    assert "cogen_guidance.logit_guidance.hard_commit_position_offset=3" in text
    assert "cogen_guidance.logit_guidance.hard_commit_max_motifs_per_step=6" in text
    assert "cogen_guidance.save_best.select_best_output=true" in text


def test_plan_runs_creates_one_run_per_target_seed(tmp_path):
    args = SimpleNamespace(
        out=tmp_path,
        disco_root=Path("/repo/DISCO"),
        python_executable="/venv/bin/python",
        motifs=",".join(DEFAULT_MOTIFS),
        target_counts="6,8",
        seeds="0-1",
        length=80,
        n_step=50,
        n_cycle=4,
        experiment="designable",
        placement_mode="even_shifted",
        shift_offsets="0,-3,3",
        no_trajectory_dump=False,
        trajectory_dtype="float32",
        include_soft_aggressive=False,
        soft_target_count=10,
        soft_seeds="0-1",
        soft_gradient_scale=50.0,
        soft_local_rescue_bonus=10.0,
        soft_motif_temperature=0.35,
    )
    runs = plan_runs(args)
    assert len(runs) == 4
    assert (tmp_path / "inputs" / "unconditional_l80.json").exists()
    assert {run.target_count for run in runs} == {6, 8}
    assert {run.seed for run in runs} == {0, 1}
    assert {run.position_offset for run in runs} == {0, -3}


def test_plan_runs_adds_soft_aggressive_control_block(tmp_path):
    args = SimpleNamespace(
        out=tmp_path,
        disco_root=Path("/repo/DISCO"),
        python_executable="/venv/bin/python",
        motifs=",".join(DEFAULT_MOTIFS),
        target_counts="6",
        seeds="0-1",
        length=80,
        n_step=50,
        n_cycle=4,
        experiment="designable",
        placement_mode="even_shifted",
        shift_offsets="0,-3,3",
        no_trajectory_dump=False,
        trajectory_dtype="float32",
        include_soft_aggressive=True,
        soft_target_count=10,
        soft_seeds="0-1",
        soft_gradient_scale=50.0,
        soft_local_rescue_bonus=10.0,
        soft_motif_temperature=0.35,
    )
    runs = plan_runs(args)
    hard_runs = [run for run in runs if run.guidance_mode == "hard_commit"]
    soft_runs = [run for run in runs if run.guidance_mode == "soft_aggressive"]
    assert len(hard_runs) == 2
    assert len(soft_runs) == 2
    assert all(run.target_count == 10 for run in soft_runs)
    soft_command = " ".join(soft_runs[0].command)
    assert "cogen_guidance.logit_guidance.enabled=true" in soft_command
    assert "cogen_guidance.logit_guidance.gradient_scale=50.0" in soft_command
    assert "cogen_guidance.logit_guidance.local_rescue_enabled=true" in soft_command
    assert "cogen_guidance.logit_guidance.hard_commit_enabled=false" in soft_command
