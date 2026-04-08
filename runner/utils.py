# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from pathlib import Path

import biotite.structure as struc
import rich
import rich.syntax
import rich.tree
import torch

from disco.data.constants import DNA_RES_IDX_TO_RESNAME, RNA_RES_IDX_TO_RESNAME
from omegaconf import DictConfig, OmegaConf

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = [],
    resolve: bool = False,
    log_dir: str | None = None,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else print(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if log_dir is not None:
        with open(Path(log_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def get_biomol_output_lines(
    ftr_dict: dict,
    aa_seq: str,
    conditional_biomol_eval: bool,
    prot_can_change_res_idx: torch.Tensor,
) -> list[str]:
    """Formats predicted sequences by chain and biomolecule type.

    For each non-ligand chain in the atom array, produces a line describing
    the chain letter, biomolecule type (protein/dna/rna), design mode
    (design/fixed), and the predicted sequence.

    Args:
        ftr_dict: Feature dictionary containing ``"atom_array"``,
            ``"atom_to_token_idx"``, and ``"token_array"``.
        aa_seq: Predicted amino acid sequence string for all protein
            residues (concatenated across protein chains).
        conditional_biomol_eval: If False, returns ``aa_seq`` directly
            without per-chain formatting.
        prot_can_change_res_idx: Boolean tensor indicating which protein
            residue positions were designable (masked).

    Returns:
        List of formatted strings, one per chain, or the raw ``aa_seq``
        if ``conditional_biomol_eval`` is False.
    """
    if not conditional_biomol_eval:
        return aa_seq

    output_strs = []
    chain_starts = struc.get_chain_starts(
        ftr_dict["atom_array"], add_exclusive_stop=True
    )
    chain_idx = prot_idx_start = 0
    for start, end in zip(chain_starts[:-1], chain_starts[1:], strict=False):
        chain_atoms = ftr_dict["atom_array"][start:end]
        is_ligand = ~(
            chain_atoms.is_rna | chain_atoms.is_dna | chain_atoms.is_protein
        ).all()

        if is_ligand:
            continue

        all_tok_idxs = ftr_dict["atom_to_token_idx"][start:end]
        dupes_mask = torch.cat(
            [
                torch.tensor([True], device=all_tok_idxs.device),
                all_tok_idxs[1:] != all_tok_idxs[:-1],
            ]
        )

        tok_idxs = all_tok_idxs[dupes_mask]
        tokens = ftr_dict["token_array"][tok_idxs]

        biomol_type_str = ""
        design_mode = "fixed"
        if (chain_atoms.is_rna | chain_atoms.is_dna).all():
            is_dna = chain_atoms.is_dna.all()

            res_map = DNA_RES_IDX_TO_RESNAME if is_dna else RNA_RES_IDX_TO_RESNAME
            biomol_str = "".join([res_map[tok.value] for tok in tokens])
            biomol_type_str = "dna" if is_dna else "rna"

        # Else clause is always protein
        else:
            assert chain_atoms.is_protein.all()

            end_idx = prot_idx_start + len(tokens)
            biomol_str = aa_seq[prot_idx_start:end_idx]

            biomol_type_str = "protein"
            if prot_can_change_res_idx[prot_idx_start:end_idx].all():
                design_mode = "design"

            prot_idx_start = end_idx

        output_strs.append(
            f"CHAIN {_ALPHABET[chain_idx]} type={biomol_type_str} mode={design_mode} seq={biomol_str}\n"
        )

        chain_idx += 1

    return output_strs
