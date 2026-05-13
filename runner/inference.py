# Copyright 2024 ByteDance and/or its affiliates.
# This file was modified in 2026 by Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
import warnings

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from os.path import exists as opexists, join as opjoin
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch

from disco.data.constants import MASK_TOKEN_IDX, PRO_RES_IDX_TO_RESNAME_ONE
from disco.data.infer_data_pipeline import get_inference_dataloader
from disco.data.json_to_feature import SampleDictToFeatures
from disco.model.disco import DISCO

# eval model
from disco.utils.seed import seed_everything
from disco.utils.torch_utils import to_device
from huggingface_hub import hf_hub_download
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from runner.dumper import DataDumper
from runner.utils import get_biomol_output_lines, print_config_tree

OmegaConf.register_new_resolver("gt", lambda a, b: a > b)

rootutils.setup_root(__file__, indicator=".project-root")

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing.*",
    category=UserWarning,
    module="openfold.model.triangular_multiplicative_update",
)


# Matches per-layer ESM/DPLM rotary inv_freq keys from old checkpoints, e.g.
# "lm_module.lm_model.model.esm.encoder.layer.23.attention.self.rotary_embeddings.inv_freq".
_ESM_PER_LAYER_INV_FREQ_RE = re.compile(
    r"^(?P<prefix>.+\.esm\.)encoder\.layer\.(?P<layer>\d+)"
    r"\.attention\.self\.rotary_embeddings\.inv_freq$"
)

# Matches ESM embedding position_embeddings keys, e.g.
# "lm_module.lm_model.model.esm.embeddings.position_embeddings.weight".
_ESM_POSITION_EMBEDDINGS_RE = re.compile(
    r"^.+\.esm\.embeddings\.position_embeddings\.weight$"
)


def _remap_esm_rotary_inv_freq(
    checkpoint_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Promotes per-layer ESM rotary inv_freq buffers to the shared key.

    Newer transformers releases (the EsmModel rewrite introduced around
    transformers 5.x) store a single shared rotary inv_freq buffer at
    ``<prefix>rotary_embeddings.inv_freq`` instead of one per attention
    layer. Old DISCO checkpoints carry the per-layer copies, which are
    identical across layers, so the first one can safely be promoted.

    The remap is only applied when the current model actually expects the
    new shared key, which makes the fix a no-op on older transformers
    versions where the per-layer buffers still exist.
    """
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict(checkpoint_state)

    grouped: dict[str, list[tuple[int, str]]] = {}
    for key in checkpoint_state:
        match = _ESM_PER_LAYER_INV_FREQ_RE.match(key)
        if match:
            grouped.setdefault(match.group("prefix"), []).append(
                (int(match.group("layer")), key)
            )

    for prefix, layer_keys in grouped.items():
        new_key = f"{prefix}rotary_embeddings.inv_freq"
        if new_key not in current_state:
            # Current model still uses the per-layer layout; leave untouched.
            continue
        if new_key in remapped:
            continue

        layer_keys.sort(key=lambda lk: lk[0])
        _, first_layer_key = layer_keys[0]

        first_inv_freq = checkpoint_state[first_layer_key]
        for _, other_key in layer_keys[1:]:
            other_inv_freq = checkpoint_state[other_key]
            if other_inv_freq.shape != first_inv_freq.shape or not torch.equal(
                other_inv_freq, first_inv_freq
            ):
                raise RuntimeError(
                    f"Rotary inv_freq mismatch under prefix '{prefix}': "
                    f"'{other_key}' differs from '{first_layer_key}'; cannot safely "
                    "promote to a single shared buffer."
                )

        remapped[new_key] = first_inv_freq
        for _, key in layer_keys:
            remapped.pop(key, None)

    return remapped


def _fill_unused_esm_position_embeddings(
    checkpoint_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Fills missing ESM ``position_embeddings.weight`` keys from current state.

    On older transformers versions like 4.50.0,
    ``EsmEmbeddings.position_embeddings`` is unconditionally
    instantiated as an ``nn.Embedding``, despite the weight being unused when
    the model is configured with ``position_embedding_type="rotary"``. The
    original DISCO checkpoint was saved against an even older transformers
    release where this layer was conditional on the absolute-position branch,
    so the key is absent from the checkpoint and ``strict=True`` loads fail
    with a missing-key error.

    The remap is only applied when the checkpoint is missing the key but the
    current model has one, so it is a no-op for checkpoints that do carry an
    absolute-position embedding.
    """
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict(checkpoint_state)

    for key, tensor in current_state.items():
        if not _ESM_POSITION_EMBEDDINGS_RE.match(key):
            continue
        if key in remapped:
            continue
        # The fill assumes position_embedding_type="rotary", in which case the
        # weight is unused at inference. Warn so that a future operator running
        # this against an absolute-position ESM model notices the silent fill.
        logger.warning(
            "Filling missing ESM position embedding '%s' from the freshly "
            "initialized model weights. This is only safe when the ESM module "
            "is configured with position_embedding_type='rotary'; an absolute "
            "configuration would receive random init values.",
            key,
        )
        remapped[key] = tensor.clone()

    return remapped


def seq_tnsr_to_str(pred_tnsr: torch.Tensor) -> str:
    """Converts a residue index tensor to a one-letter amino acid sequence string.

    Args:
        pred_tnsr: 1-D integer tensor of residue type indices.

    Returns:
        Single-letter amino acid sequence string.
    """
    return "".join([PRO_RES_IDX_TO_RESNAME_ONE[i.item()] for i in pred_tnsr])


class InferenceRunner:
    """Main inference runner for generating structure and sequence predictions.

    Sets up the distributed environment, loads the DISCO model from a
    checkpoint, and provides methods to run forward-pass predictions and
    dump results to disk.

    Sequentially sets up the environment, output directories, model,
    checkpoint, and data dumper.

    Args:
        configs: Hydra DictConfig containing all inference settings
            (model, data, fabric, checkpoint path, etc.).
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(need_atom_confidence=configs.need_atom_confidence)

    def init_env(self) -> None:
        """Initializes the distributed environment and CUDA settings.

        Creates a Lightning Fabric instance with DDP strategy, launches
        the distributed processes, and configures optional kernel compilation
        flags (DeepSpeed EvoformerAttention, fast LayerNorm).
        """
        self.fabric = Fabric(
            strategy=DDPStrategy(find_unused_parameters=False),
            num_nodes=self.configs.fabric.num_nodes,
            loggers=[
                hydra.utils.instantiate(logger)
                for _, logger in self.configs.logger.items()
            ],
        )
        self.print(
            f"Fabric: {self.fabric}, rank: {self.fabric.global_rank}, world_size: {self.fabric.world_size}"
        )
        self.fabric.launch()
        self.device = self.fabric.device
        torch.cuda.set_device(self.device)
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        """Creates the main output and error directories for inference results."""
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """Initializes the DISCO model and moves it to the target device.

        Optionally instantiates a structure encoder and a sequence sampling
        strategy from the config before constructing the model.
        """
        structure_encoder = None
        if self.configs.structure_encoder.use_structure_encoder:
            structure_encoder = hydra.utils.instantiate(
                self.configs.structure_encoder.args
            )

        sequence_sampling_strategy = hydra.utils.instantiate(
            self.configs.sequence_sampling_strategy
        )
        self.model = DISCO(
            self.configs, structure_encoder, sequence_sampling_strategy
        ).to(self.device)

    def load_checkpoint(self) -> None:
        """Loads model weights from a checkpoint file.

        If ``load_checkpoint_path`` is ``null`` or the path does not exist
        locally, the checkpoint is automatically downloaded from the
        HuggingFace Hub (``DISCO-Design/DISCO``).

        Handles ``module.`` prefixes from DDP-saved checkpoints and logs
        a warning for parameters whose names are missing or whose shapes
        do not match the current model before skipping them. Sets the model
        to eval mode after loading.
        """
        checkpoint_path = self.configs.load_checkpoint_path

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            self.print(
                "Checkpoint not found locally. "
                "Downloading from HuggingFace Hub (DISCO-Design/DISCO)..."
            )
            checkpoint_path = hf_hub_download(
                repo_id="DISCO-Design/DISCO",
                filename="DISCO.pt",
            )

        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device, weights_only=False)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }

        current = self.model.state_dict()
        checkpoint["model"] = _remap_esm_rotary_inv_freq(checkpoint["model"], current)
        checkpoint["model"] = _fill_unused_esm_position_embeddings(
            checkpoint["model"], current
        )

        filtered = OrderedDict()
        for k, v in checkpoint["model"].items():
            if k in current and v.shape == current[k].shape:
                filtered[k] = v  # → OK: same name & same shape
            else:
                print(
                    f"Skipping '{k}': not found or shape changed "
                    f"(saved {tuple(v.shape)} → current "
                    f"{tuple(current.get(k, torch.empty(0)).shape)})"
                )

        self.model.load_state_dict(
            state_dict=filtered,
            strict=self.configs.load_strict,
        )

        self.model.eval()
        self.print("Finish loading checkpoint.")

    def init_dumper(self, need_atom_confidence: bool = False):
        """Initializes the DataDumper used to write predictions to disk.

        Args:
            need_atom_confidence: If True, the dumper will also save
                per-atom confidence data alongside summary scores.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir, need_atom_confidence=need_atom_confidence
        )

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(
        self, data: Mapping[str, Mapping[str, Any]], sample2feat: SampleDictToFeatures
    ) -> dict[str, torch.Tensor]:
        """Runs a forward pass through the model to generate predictions.

        Moves data to the target device and runs inference under the
        configured mixed-precision context.

        Args:
            data: Nested mapping containing at minimum an
                ``"input_feature_dict"`` key with the model input features.
            sample2feat: Converter that maps raw sample dictionaries to
                model-ready feature tensors.

        Returns:
            Dictionary of prediction tensors (e.g. ``"coordinate"``,
            ``"summary_confidence"``, ``"decoder_prediction"``).
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                sample2feat=sample2feat,
            )

        return prediction

    def print(self, msg: str):
        """Logs an info-level message only on global rank 0."""
        if self.fabric.is_global_zero:
            logging.info(msg)

    def debug(self, msg: str):
        """Logs a debug-level message only on global rank 0."""
        if self.fabric.is_global_zero:
            logging.debug(msg)


@hydra.main(config_path="../configs", config_name="inference.yaml", version_base=None)
def main(configs: DictConfig):
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    print_config_tree(configs, resolve=True)

    # Runner
    runner = InferenceRunner(configs)

    if isinstance(configs.seeds, int):
        configs.seeds = [configs.seeds]

    num_inference_seeds = configs.get("num_inference_seeds")
    if num_inference_seeds is not None:
        configs.seeds = list(range(num_inference_seeds))

    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    dataloader = get_inference_dataloader(
        runner.fabric,
        configs=configs,
        num_eval_seeds=configs.seeds,
    )

    dump_dir = Path(runner.dump_dir)
    pdbs_dir, seq_dir = dump_dir / "pdbs", dump_dir / "sequences"
    pdbs_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    num_data, curr_seed, pre_log_dicts = len(dataloader.dataset), None, []
    for batch in dataloader:
        try:
            data, atom_array, sample2feat, data_error_message = batch[0]

            if len(data_error_message) > 0:
                logger.info(data_error_message)
                with open(
                    opjoin(runner.error_dir, f"{data['sample_name']}.txt"),
                    "w",
                ) as f:
                    f.write(data_error_message)
                continue

            if data["seed"] != curr_seed:
                curr_seed = data["seed"]
                logger.info(f"Seed: {curr_seed + 1} / {len(configs.seeds)}")
                seed_everything(seed=curr_seed, deterministic=configs.deterministic)

            # add atom_array in data
            data["input_feature_dict"]["atom_array"] = atom_array
            prot_can_change_res_idx = (
                data["input_feature_dict"]["masked_prot_restype"] == MASK_TOKEN_IDX
            )

            sample_name = data["sample_name"]
            path = Path(f"{runner.dump_dir}/{sample_name}_sample_{curr_seed}.pdb")
            if path.exists():
                logger.info(f"{path} already exists -- skipping")
                continue

            logger.info(
                f"[Rank {runner.fabric.global_rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                f"N_atom {data['N_atom'].item()}"
            )

            prediction = runner.predict(data, sample2feat)

            file_format = "cif"
            output_format = configs.get("output_format", None)

            atom_array_pre = prediction.get("atom_array")
            atom_array = atom_array_pre[0] if atom_array_pre is not None else atom_array

            to_log_pred_dict = copy.deepcopy(prediction)
            if configs.n_seq_duplicates_per_structure > 1:
                to_log_pred_dict["coordinate"] = prediction["coordinate"][0].unsqueeze(
                    dim=0
                )

            structure_path = None
            if output_format is None:
                structure_path = runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=curr_seed,
                    pred_dict=to_log_pred_dict,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                    file_format=file_format,
                    dump_dir=pdbs_dir,
                )[0]
            elif output_format == "unconditional_monomer_protein":
                structure_path = runner.dumper._save_structure(
                    pred_coordinates=to_log_pred_dict["coordinate"],
                    prediction_save_dir=pdbs_dir,
                    sample_name=sample_name,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                    file_format="pdb",
                    pred_dict=to_log_pred_dict,
                    idx_offset=curr_seed,
                )[0]
            else:
                raise ValueError(f"Unknown output format: {output_format}")

            pdb_id = path.name.split(".")[0]

            # NOTE: we assume batchsize=1 and assuming lig["ligand"] is not CCD code but SMILES
            # Extract any SMILES
            all_sequences = (
                sample2feat[0] if isinstance(sample2feat, list) else sample2feat
            ).input_dict.get("sequences", [])

            ligand_smiles = [
                ligand["ligand"]
                for seq in all_sequences
                for key, ligand in seq.items()
                if key.startswith("ligand")
                and isinstance(ligand, dict)
                and "ligand" in ligand
            ]
            if ligand_smiles:
                lig_path = pdbs_dir / f"{pdb_id}_ligands.txt"
                with open(lig_path, "w") as f:
                    f.writelines(f"ligand_smiles {s}\n" for s in ligand_smiles)
                    f.flush()
                    os.fsync(f)

            this_dict = {"pdb_id": pdb_id, "model_pdb_path": str(structure_path)}

            ftr_dict = data["input_feature_dict"]
            decoder_pred = prediction.get("decoder_prediction")
            if decoder_pred is not None:
                if configs.n_seq_duplicates_per_structure > 1:
                    aa_seqs = [
                        seq_tnsr_to_str(decoder_pred[i])
                        for i in range(configs.n_seq_duplicates_per_structure)
                    ]
                else:
                    aa_seqs = [seq_tnsr_to_str(decoder_pred)]

                seq_path = seq_dir / f"{pdb_id}.txt"
                with open(seq_path, "w") as f:
                    for i, aa_seq in enumerate(aa_seqs):
                        f.write(f">cogen_seq {i}\n")

                        has_nucleic = (
                            ftr_dict["atom_array"].is_dna.any()
                            or ftr_dict["atom_array"].is_rna.any()
                        )
                        dna_rna_aa_lines = get_biomol_output_lines(
                            ftr_dict,
                            aa_seq,
                            has_nucleic or configs.eval_version == "conditional_biomol",
                            prot_can_change_res_idx,
                        )

                        f.writelines(dna_rna_aa_lines)

                        if len(ligand_smiles) > 0:
                            f.write("\n")
                            f.writelines(f"ligand_smiles {s}\n" for s in ligand_smiles)

                    # Force write the file to disk before releasing lock
                    f.flush()
                    os.fsync(f)

                    assert seq_path.exists(), f"{seq_path} does not exist!"

                this_dict["model_sequence"] = (
                    aa_seqs[0] if len(aa_seqs) == 1 else aa_seqs
                )
                this_dict["ligands"] = ligand_smiles

            pre_log_dicts.append(this_dict)

            logger.info(
                f"[Rank {runner.fabric.global_rank}] {data['sample_name']} succeeded.\n"
                f"Results saved to {configs.dump_dir}"
            )

        except Exception as e:
            name = (
                data.get("sample_name", "unknown")
                if isinstance(data, dict)
                else "unknown"
            )
            error_message = f"[Rank {runner.fabric.global_rank}]{name} {e}:\n{traceback.format_exc()}"
            logger.info(error_message)
            # Save error info
            if opexists(error_path := opjoin(runner.error_dir, f"{name}.txt")):
                os.remove(error_path)
            with open(error_path, "w") as f:
                f.write(error_message)
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
