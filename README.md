

<p align="center">
    <img src="assets/disco.png" alt="DISCO: Diffusion for Sequence-Structure Co-design" width="900"/><br>
    <img src="assets/carbene.gif" width="700"/><br>
  <a href="https://arxiv.org/abs/2604.05181"><img src="https://img.shields.io/badge/arXiv-94133F?style=for-the-badge&logo=arxiv" alt="arXiv"/></a>
  <a href="https://disco-design.github.io/"><img src="https://img.shields.io/badge/📝%20Blog-007A87?style=for-the-badge&logoColor=white" alt="Blog"/></a>
  <a href="https://huggingface.co/DISCO-Design/DISCO"><img src="https://img.shields.io/badge/HuggingFace-DE9B35.svg?style=for-the-badge&logo=HuggingFace" alt="HF"/></a>
</p>

DISCO (DIffusion for Sequence-structure CO-design) is a multimodal generative model that simultaneously co-designs protein sequences and 3D structures, conditioned on and co-folded with arbitrary biomolecules — including small-molecule ligands, DNA, and RNA. Unlike sequential pipelines that first generate a backbone and then apply inverse folding, DISCO generates both modalities jointly, enabling sequence-based objectives to inform structure generation and vice versa.

DISCO achieves state-of-the-art in silico performance in generating binders for diverse biomolecular targets with fine-grained property control, performing best on 178/179 evaluated ligands, as well as DNA and RNA. Applied to new-to-nature catalysis, DISCO was conditioned solely on reactive intermediates — without pre-specifying catalytic residues or relying on template scaffolds — to design diverse heme enzymes with novel active-site geometries. These enzymes catalyze new-to-nature carbene-transfer reactions, including alkene cyclopropanation, spirocyclopropanation, B–H and C(sp³)–H insertions, with top activities exceeding those of engineered enzymes. Random mutagenesis of a selected design further yielded a fourfold activity gain, indicating that the designed enzymes are evolvable.

<p align="center">
  <img src="assets/conditional_design_results.png" width="95%" alt="DISCO vs baselines on conditional protein design" />
</p>

## Quick Start

1. **Install** — see [Installation](#-installation) below.
2. **Set up prerequisites** — (optionally) configure [CUTLASS](#cutlass-optional) (see below).
3. **Run:**

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  seeds=\[0,1,2,3,4\]
```

If a run is interrupted, simply rerun the same command — DISCO automatically skips samples that have already been generated.

> **Note:** The first time you run inference, it may take some time before inference steps begin, as the pairformer kernels are compiled just in time.

## 📦 Installation

DISCO uses [uv](https://docs.astral.sh/uv/) for dependency management (you may need to [install uv](https://docs.astral.sh/uv/getting-started/installation/) first). To install:

> **AMD GPUs:** DeepSpeed does not support AMD GPUs. If you are using an AMD GPU, remove the `deepspeed` dependency from `pyproject.toml` before running `uv sync`, and run with `use_deepspeed_evo_attention=false`.

```bash
uv sync
```

By default, `uv sync` installs PyTorch with its default backend. If you need a specific CUDA or CPU backend, uninstall torch and reinstall with the desired index URL. For example, for CUDA 12.4:

```bash
uv pip uninstall torch
uv pip install torch --torch-backend=cu124
```

To activate the environment run from the top-level of the repository:

```bash
source .venv/bin/activate
```

## 🔧 Prerequisites

### CUTLASS (optional)

By default, DISCO uses [DeepSpeed4Science EvoformerAttention](https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/) for memory-efficient attention, which significantly reduces GPU memory usage and enables inference on longer sequences. This requires [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) to be available on disk and a GPU with **Ampere or newer architecture** (e.g. A100, L40S, H100, H200, B100, B200).

To set it up, clone the CUTLASS repository and set the `CUTLASS_PATH` environment variable:

```bash
git clone https://github.com/NVIDIA/cutlass.git /path/to/cutlass
export CUTLASS_PATH=/path/to/cutlass
```

You can add `CUTLASS_PATH` to your shell profile so it persists across sessions. The attention kernels will be compiled the first time they are invoked.

If you prefer to skip the CUTLASS installation, disable DeepSpeed attention on the command line:

```bash
python runner/inference.py use_deepspeed_evo_attention=false ...
```

This falls back to a naive attention implementation that materializes the full attention matrix and uses substantially more GPU memory.

## 🚀 Running Inference

Inference is run through the Hydra-based runner:

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/your_config.json \
  seeds=\[$(seq -s "," 0 4)\]
```

### Key command-line options

| Option | Description |
|--------|-------------|
| `experiment=` | Experiment preset (`designable` or `diverse`). See [Experiment Presets](#experiment-presets) below. |
| `input_json_path=` | Path to the input JSON file describing what to generate. |
| `seeds=` | List of random seeds, e.g. `[0,1,2]`. Each seed produces one sample per job in the input JSON, so the total number of generated samples equals `len(seeds) * len(jobs)`. |
| `num_inference_seeds=` | Alternative to `seeds=`: generates seeds `[0, 1, ..., N-1]`. For example, `num_inference_seeds=100` produces 100 samples per job. |
| `effort=` | Compute preset: `max` (default) or `fast`. **We only recommend `effort=fast` for unconditional generation; for conditional generation (e.g. ligand- or DNA/RNA-conditioned) use `effort=max`.** See [Trading off quality for speed](#trading-off-quality-for-speed). |
| `dump_dir=` | Output directory for generated structures. Defaults to `./output`. |

### Experiment presets

DISCO ships with two experiment presets that control the trade-off between designability and diversity:

- **`designable`** — Uses entropy-adaptive temperature scaling and noisy guidance over both sequence and structure. This steers the model toward samples that are more likely to refold correctly under an external structure predictor, at the cost of reduced structural variety.
- **`diverse`** — Disables noisy guidance and entropy-adaptive temperature. The model samples more freely from its learned distribution, producing greater structural variety at the cost of lower average designability.

Which preset to use depends on the task — see [Reproducing Paper Experiments](#-reproducing-paper-experiments) for guidance on which preset was used in each experiment.

> **Tip: cheaper designable runs.** The `designable` preset uses noisy guidance, which increases the effective batch size of each forward pass and slows down inference. You can disable it while keeping the rest of the designable settings by adding `sample_diffusion.noisy_guidance.enabled=false` on the command line. This gives slightly lower designability scores but reduces compute costs, which can be useful for rapid prototyping or large-scale screening runs.

### Trading off quality for speed

DISCO provides two `effort` presets that control the number of recycling cycles and diffusion steps:

| Preset | Diffusion steps | Recycling cycles | Description |
|--------|:-:|:-:|-------------|
| `effort=fast` | 100 | 2 | ~4x faster inference with only ~10% lower co-designability. Good for prototyping and large screening runs. This is the default. |
| `effort=max` | 200 | 4 | Full quality used in the paper. |

> **⚠️ Important:** We only recommend `effort=fast` for **unconditional** generation. For **conditional** generation (e.g. ligand- or DNA/RNA-conditioned), use `effort=max` for best results.

```bash
# Fast (default) — good for prototyping
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/your_config.json \
  seeds=\[0,1,2,3,4\]

# Max quality — reproducing paper results
python runner/inference.py \
  experiment=designable \
  effort=max \
  input_json_path=input_jsons/your_config.json \
  seeds=\[0,1,2,3,4\]
```

You can also override the individual parameters directly with `model.N_cycle=` and `sample_diffusion.N_step=`.

<p align="center">
  <img src="assets/n_cycle_steps_ablation.png" width="85%" alt="Co-designability and structural diversity vs. number of steps and cycles" />
</p>

The figure above shows the trade-off between co-designability, structural diversity, and compute, measured using the `designable` preset with noisy guidance disabled. Beyond 2 cycles and 100 steps, returns diminish quickly.

> **Note:** When benchmarking against DISCO, use `effort=max` to reproduce the full-quality results reported in the paper.

### Output directory

Generated structures are saved under `dump_dir` with the following layout:

```
dump_dir/
  pdbs/
    <name>_sample_<seed>.pdb
    <name>_sample_<seed>_ligands.txt   # only if ligands are present
  sequences/
    <name>_sample_<seed>.txt
  ERR/
    <name>.txt                         # only for failed samples
```

Here `<name>` is the job name from the input JSON (e.g. `length_200_heme_b`) and `<seed>` is the random seed used for that sample.

You can override `dump_dir` on the command line:

```bash
python runner/inference.py dump_dir=/my/output/dir ...
```

By default it resolves to `./output` relative to the working directory.

## 🔬 Reproducing Paper Experiments

The sections below walk through each class of experiment from the paper. We provide all input JSON files needed to reproduce the reported results. To make comparisons to DISCO easier, the raw generated samples and results for **all** *in silico* experiments are available on [Hugging Face](https://huggingface.co/datasets/DISCO-Design/DISCO_benchmark_data).

### 🧬 Unconditional protein generation

In the unconditional setting, DISCO receives no conditioning target and generates both a protein sequence and a 3D structure from scratch. We evaluate at chain lengths 70, 100, 200, and 300 to assess how generation quality varies with protein size.

```bash
python runner/inference.py \
  experiment=designable \
  input_json_path=input_jsons/unconditional_config.json \
  seeds=\[$(seq -s "," 0 99)\]
```

The unconditional config (`input_jsons/unconditional_config.json`) contains fully masked protein chains of varying lengths. The model generates both sequence and structure from scratch. 

### 💊 Ligand-conditioned protein design

In the ligand-conditioned setting, DISCO is given only a small molecule and generates a protein sequence and structure with a complementary binding site, without requiring a template structure or seed sequence.

We provide five representative ligand input files in `input_jsons/`:

| File | Ligand | Description |
|------|--------|-------------|
| `heme_b.json` | Heme B | Iron-porphyrin cofactor central to the paper's enzyme design experiments |
| `NDI.json` | NDI | Naphthalenediimide derivative |
| `PLP.json` | PLP | Pyridoxal phosphate, a common enzyme cofactor |
| `thyroxine.json` | Thyroxine | Thyroid hormone |
| `warfarin.json` | Warfarin | Anticoagulant drug |

Each file specifies fully masked protein chains (at lengths 150, 200, and 250) alongside a ligand provided as an SDF file. To run:

```bash
python runner/inference.py \
  experiment=diverse \
  effort=max \
  input_json_path=input_jsons/heme_b.json \
  seeds=\[$(seq -s "," 0 4)\]
```

To reproduce the paper's ligand-conditioned results, use `experiment=diverse`.

### 🧪 Nucleic acid-conditioned protein design

DISCO can also design proteins conditioned on nucleic acid sequences, generating protein chains that form complexes with DNA or RNA.

Two nucleic acid conditioning files are provided:

| File | Target | Description |
|------|--------|-------------|
| `6YMC_rna.json` | RNA | Protein design conditioned on a 26-nt RNA sequence (PDB: 6YMC) |
| `7S03_dna.json` | DNA | Protein design conditioned on double-stranded DNA (PDB: 7S03) |

These files sweep over protein chain lengths (50--80) while keeping the nucleic acid sequence fixed. To run:

```bash
python runner/inference.py \
  experiment=diverse \
  effort=max \
  input_json_path=input_jsons/6YMC_rna.json \
  seeds=\[$(seq -s "," 0 4)\]
```

To reproduce the paper's nucleic acid results, use `experiment=diverse`.

## 🕺 Studio-179: A Ligand Benchmark for Generative Protein Design 💃

To systematically evaluate ligand-conditioned protein design, we curated **Studio-179**: a benchmark of 170 natural and non-natural ligands — plus 9 multi-ligand combinations — spanning catalysis, pharmaceuticals, luminescence, and sensing.

The library covers a range of chemical and geometric properties relevant to protein-ligand interactions:

- **Rigid molecules** — e.g., the persistent organic pollutant tetrachlorodibenzodioxin
- **Large or flexible molecules** — e.g., CoQ10, a 50-heavy-atom cofactor with a long isoprenoid tail
- **Metals and metalloclusters** — e.g., [4Fe-4S] iron-sulfur clusters

The SDF files for all 179 ligands are included in the `studio-179/` directory, organized by priority tier. The full benchmark is split across four input files for parallel execution:

```
input_jsons/all_priorities_ligands_split_0.json
input_jsons/all_priorities_ligands_split_1.json
input_jsons/all_priorities_ligands_split_2.json
input_jsons/all_priorities_ligands_split_3.json
```

To run a split:

```bash
python runner/inference.py \
  experiment=diverse \
  effort=max \
  input_json_path=input_jsons/all_priorities_ligands_split_0.json \
  seeds=\[$(seq -s "," 0 4)\]
```

To run against a **single ligand** instead of all 179, create an input JSON with jobs for that ligand at the three benchmark protein lengths (150, 200, 250 residues). For example, to benchmark only heme B:

```json
[
  {
    "name": "length_150_heme_b",
    "sequences": [
      {"proteinChain": {"sequence": "<150 '-' characters>", "count": 1}},
      {"ligand": {"ligand": "FILE_studio-179/priority_1/heme_b_final_0.sdf", "count": 1}}
    ]
  },
  {
    "name": "length_200_heme_b",
    "sequences": [
      {"proteinChain": {"sequence": "<200 '-' characters>", "count": 1}},
      {"ligand": {"ligand": "FILE_studio-179/priority_1/heme_b_final_0.sdf", "count": 1}}
    ]
  },
  {
    "name": "length_250_heme_b",
    "sequences": [
      {"proteinChain": {"sequence": "<250 '-' characters>", "count": 1}},
      {"ligand": {"ligand": "FILE_studio-179/priority_1/heme_b_final_0.sdf", "count": 1}}
    ]
  }
]
```

The SDF files for each ligand can be found in `studio-179/priority_{0,1,2,3}/`. See `input_jsons/heme_b.json` for a complete working example.

### Evaluation metric: co-designability

For each ligand, we quantify the fraction of generated designs that are both structurally diverse and **co-designable**. A design is considered co-designable if the protein backbone and all ligand centroids have an RMSD < 2 Å upon refolding with Chai-1. This evaluates whether the generated sequence encodes the intended structure and binding mode, rather than only assessing the plausibility of the generated structure in isolation.

Studio-179 is intended as a community resource. We encourage others to evaluate their ligand-conditioned design methods on this benchmark and to report results using the same co-designability metric for comparability.

## 🎨 Running Your Own Designs

The sections above reproduce experiments from the paper using provided input files. This section walks through how to set up DISCO for your own custom targets.

### Custom ligand-conditioned design

To design a protein around your own small molecule, create an input JSON with a fully masked protein chain and your ligand. There are three ways to specify a ligand:

#### Option 1: SMILES string

The simplest approach — no files needed. DISCO will automatically generate a 3D conformer from the SMILES string using RDKit.

```json
[
  {
    "name": "my_ligand_design",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
          "count": 1
        }
      },
      {
        "ligand": {
          "ligand": "CC(=O)Oc1ccccc1C(=O)O",
          "count": 1
        }
      }
    ]
  }
]
```

The protein sequence length is determined by the number of `-` characters (200 in this example). Replace the SMILES string with whatever molecule you want to target.

```bash
python runner/inference.py \
  experiment=diverse \
  effort=max \
  input_json_path=input_jsons/my_ligand_design.json \
  seeds=\[$(seq -s "," 0 4)\]
```

#### Option 2: Molecular structure file (SDF, MOL, MOL2, or PDB)

If you already have a 3D conformer for your molecule, you can pass it directly. Prefix the file path with `FILE_`:

```json
{
  "ligand": {
    "ligand": "FILE_/absolute/path/to/molecule.sdf",
    "count": 1
  }
}
```

Supported file formats are **SDF**, **MOL**, **MOL2**, and **PDB**. The file **must** contain a 3D conformer (2D structures will be rejected). Paths can be absolute or relative to the repository root:

```json
{
  "ligand": {
    "ligand": "FILE_my_ligands/caffeine.mol2",
    "count": 1
  }
}
```

> **Note:** XYZ files are not currently supported. If you have an XYZ file, convert it to SDF or MOL2 first using a tool like Open Babel (`obabel input.xyz -O output.sdf`).

#### Option 3: CCD code

For standard ligands in the PDB Chemical Component Dictionary, use the CCD code prefixed with `CCD_`:

```json
{
  "ligand": {
    "ligand": "CCD_ATP",
    "count": 1
  }
}
```

For multi-component ligands (e.g., glycans), concatenate CCD codes with underscores: `"CCD_NAG_BMA_BGC"`.

#### Putting it together: a full ligand example

Here is a complete input JSON that designs 150- and 200-residue proteins around a ligand provided as an SDF file, similar to the provided heme B example:

```json
[
  {
    "name": "my_mol_length_150",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "------------------------------------------------------------------------------------------------------------------------------------------------------",
          "count": 1
        }
      },
      {
        "ligand": {
          "ligand": "FILE_my_ligands/my_molecule.sdf",
          "count": 1
        }
      }
    ]
  },
  {
    "name": "my_mol_length_200",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
          "count": 1
        }
      },
      {
        "ligand": {
          "ligand": "FILE_my_ligands/my_molecule.sdf",
          "count": 1
        }
      }
    ]
  }
]
```

Each entry in the top-level list is a separate design job. Here we sweep over two protein lengths for the same ligand.

### Custom nucleic acid-conditioned design

DISCO can design proteins that bind DNA or RNA sequences. Provide the nucleic acid as a fixed sequence alongside a fully masked protein chain.

#### RNA-binding protein design

```json
[
  {
    "name": "my_rna_binder",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "----------------------------------------------------------------------",
          "count": 1
        }
      },
      {
        "rnaSequence": {
          "sequence": "GGCUAGCCAUUUGAC",
          "count": 1
        }
      }
    ]
  }
]
```

RNA sequences use the letters `A`, `U`, `G`, `C`, and `N` (unknown). Masking RNA positions with `-` is **not supported** — all nucleic acid positions must be fully specified.

```bash
python runner/inference.py \
  experiment=diverse \
  effort=max \
  input_json_path=input_jsons/my_rna_binder.json \
  seeds=\[$(seq -s "," 0 4)\]
```

#### DNA-binding protein design

DNA works the same way, but uses `dnaSequence` with the letters `A`, `T`, `G`, `C`, and `N`. As with RNA, masking DNA positions with `-` is **not supported** — all positions must be fully specified. For double-stranded DNA, add each strand as a separate entry:

```json
[
  {
    "name": "my_dna_binder",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "----------------------------------------------------------------------",
          "count": 1
        }
      },
      {
        "dnaSequence": {
          "sequence": "GATTACAGATC",
          "count": 1
        }
      },
      {
        "dnaSequence": {
          "sequence": "GATCTGTAATC",
          "count": 1
        }
      }
    ]
  }
]
```

> **Important:** `dnaSequence` represents a single strand. For double-stranded DNA, you must add the reverse complement as a second `dnaSequence` entry, as shown above.

### Tips for custom designs

- **Protein length:** The number of `-` characters sets the designed protein length. If you're unsure what length to use, try sweeping over a range (e.g., 100, 150, 200) by including multiple jobs in the same JSON file.
- **Partial masking:** You can fix specific residues and mask others. For example, `"MKTL----VPEG"` fixes the termini and designs the middle region.
- **Multiple seeds:** Use `seeds=[0,1,...,N]` or `num_inference_seeds=N` to generate multiple independent samples per job for diversity.
- **Experiment preset:** Use `experiment=diverse` for maximum structural diversity (recommended for ligand and nucleic acid targets). Use `experiment=designable` when you want higher confidence that the designed sequence will refold to the intended structure.
- **Covalent bonds:** You can specify covalent bonds between entities (e.g., a covalently attached ligand).

## 📝 Input JSON Format

The input JSON closely follows the [AlphaFold Server](https://alphafoldserver.com/) format. Each file is a list of jobs, where each job specifies the entities to model:

```json
[
  {
    "name": "my_design",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MKTL------VPEG",
          "count": 1
        }
      },
      {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      }
    ],
    "covalent_bonds": []  // optional
  }
]
```

### Masking with `-`

The character **`-`** (hyphen) denotes a **masked position** in protein sequences. DISCO will generate residues at these positions. Fixed residues are specified with their standard letter codes. For example:

- `"--------"` — fully masked 8-residue protein (design all positions)
- `"MKTL----VPEG"` — fix the N- and C-terminal residues, design the middle

Masking is only supported for protein chains. DNA and RNA sequences must be fully specified.

### Supported entity types

- **`proteinChain`** — Protein sequence (standard 20 amino acids, `X` for unknown, `-` for masked)
- **`dnaSequence`** — Single-stranded DNA (`A`, `T`, `G`, `C`, `N` for unknown). Masking is not supported; all positions must be specified.
- **`rnaSequence`** — Single-stranded RNA (`A`, `U`, `G`, `C`, `N` for unknown). Masking is not supported; all positions must be specified.
- **`ligand`** — Small molecule, specified as:
  - A CCD code (e.g., `CCD_ATP`)
  - A SMILES string
  - A path to an SDF/MOL/MOL2/PDB file, prefixed with `FILE_` (e.g., `FILE_path/to/ligand.sdf`). The path can be **absolute** or **relative to the repository root**. For example, `FILE_studio-179/priority_1/heme_b_final_0.sdf` would resolve relative to the repo base directory.
- **`ion`** — Ion specified by CCD code (e.g., `MG`, `ZN`)

## ⚠️ Known Limitations

- **No protein-protein complex design.** DISCO does not currently support designing multi-chain protein complexes. The language model used to replace the MSA module (DPLM) was trained exclusively on single-chain proteins, and as a result it predicts multi-chain proteins poorly. Inputs containing more than one protein chain will raise an error. Protein-ligand, protein-DNA, and protein-RNA complexes with a single protein chain are fully supported.
- **No motif scaffolding.** Motif scaffolding (designing a protein around a fixed structural motif) is not currently supported but will be added in a future update.

## 🔜 Coming Soon

- **Feynman-Kac correctors.** The Feynman-Kac correctors code for improved sampling will be added to the repository shortly.
- **Training code.** Code for training DISCO from scratch will also be released.

## 🙏 Acknowledgements

We gratefully acknowledge the authors of [Protenix](https://github.com/bytedance/protenix), as this codebase is built on top of their repository.

## 📖 Citation

```bibtex
@Article{disco2026,
      title={General Multimodal Protein Design Enables DNA-Encoding of Chemistry},
      author={Jarrid Rector-Brooks and Théophile Lambert and Marta Skreta and Daniel Roth and Yueming Long and Zi-Qi Li and Xi Zhang and Miruna Cretu and Francesca-Zhoufan Li and Tanvi Ganapathy and Emily Jin and Avishek Joey Bose and Jason Yang and Kirill Neklyudov and Yoshua Bengio and Alexander Tong and Frances H. Arnold and Cheng-Hao Liu},
      year={2026},
      eprint={2604.05181},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.05181},
}
```
