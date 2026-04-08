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

from typing import Any

from disco.utils.seq import (
    amino_acid_constants,
    dna_constants,
    ligand_constants,
    rna_constants,
)


def exists(val: Any) -> bool:
    """Check if a value exists.

    :param val: The value to check.
    :return: `True` if the value exists, otherwise `False`.
    """
    return val is not None


def get_residue_constants(
    res_chem_type: str | None = None, res_chem_index: Any | None = None
) -> Any:
    """Returns the residue constants module for a given residue chemical type.

    Exactly one of ``res_chem_type`` or ``res_chem_index`` must be provided.

    Args:
        res_chem_type: String identifying the residue chemistry (e.g.,
            ``"peptide"``, ``"rna"``, ``"dna"``). Matching is
            case-insensitive and substring-based.
        res_chem_index: Integer index of the residue chemistry type
            (0=peptide, 1=RNA, 2=DNA, other=ligand).

    Returns:
        The corresponding constants module (``amino_acid_constants``,
        ``rna_constants``, ``dna_constants``, or ``ligand_constants``).
    """
    assert exists(res_chem_type) or exists(
        res_chem_index
    ), "Either `res_chem_type` or `res_chem_index` must be provided."
    if (exists(res_chem_type) and "peptide" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 0
    ):
        residue_constants = amino_acid_constants
    elif (exists(res_chem_type) and "rna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 1
    ):
        residue_constants = rna_constants
    elif (exists(res_chem_type) and "dna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 2
    ):
        residue_constants = dna_constants
    else:
        residue_constants = ligand_constants
    return residue_constants
