# Structure–Stability Ablations (ISMB 2026 submission) — repository overview

This repository contains code used for the experiments in our ISMB 2026 submission. The implementation builds on the GeoStab architecture and includes additional scripts and fixes that were developed to run feature generation, model training / ablations, and structural analyses used in the paper.

This README gives a quick tour of what is in the repo and how the major pieces fit together.

## High-level structure

- `generate_features/`
  - Utilities to generate per-sample features used by the models:
    - `esm2_embedding.py` — compute ESM-2 per-residue embeddings (supports batching).
    - `esm1v_logits.py` — compute ESM-1v masked-marginal logits used in ensembles.
    - `fixed_embedding.py` — per-residue physicochemical descriptors (7 dims + optional pH/pLDDT concatenation elsewhere).
    - `coordinate.py` — parse PDBs and write `pos14` / `pos14_mask` tensors (per-residue 14-atom layout).
    - `pair.py` / `pair_advanced.py` — compute pairwise geometric features (rotation/quaternion, relative vectors, RBF-expanded distance features). `pair_advanced` computes extra features (dmin RBF, CB vectors).
    - `ensemble_*` scripts — utilities that assemble inputs (wt / mut) into ensemble tensors used downstream for ddG/dTm/fitness models.
- `scripts/`
  - Orchestration and analysis:
    - `generate_esmfold_sharded.py` — run ESMFold (WT de-duplication, stable sharding, safe propagation of WT outputs).
    - `run_feature_scripts_sharded.py` — wrapper to run feature generation across many samples in a sharded / array-job friendly way (dedup WT, propagate outputs).
    - `analyze_graph_vs_performance.py` — structural analysis (TM-align mapping + Kabsch superposition) and correlation of structural metrics vs model performance. Produces CSVs and plots.
- `model_dTm_3D/` (imported by training scripts)
  - Model architecture used for the dTm models (Geometric attention blocks, encoders, readouts).
- `train_code/`
  - Training / ablation scripts:
    - `train_geodtm_ablation.py` — training with ablation flags (per-feature ablation, duplicate WT structure option, rosetta mutant files).
    - `train_geodtm_radius_ablation.py` — radius-based ablation (crop features within mutation-centered window).
    - `fitness_model/` — pretraining model code for masked prediction tasks.

## Typical workflows
0. Generate Fastas and directory structure for WT/mutants from dataset: run Create_GeoStab_FASTAs.py
1. Generate structural predictions for WT / MUT with ESMFold:
   - `python scripts/generate_esmfold_sharded.py --data_parent /path/to/features_root --N <num_shards> --ID <shard_id> ...`

2. Generate per-sample features (ESM2 embeddings, fixed embeddings, coordinates, pair features):
   - Use `scripts/run_feature_scripts_sharded.py` to process many samples with sharding and WT deduplication.
   - Or call feature scripts directly:
     - `python generate_features/esm2_embedding.py --fasta_files sample.fasta --saved_folders sample_dir`
     - `python generate_features/fixed_embedding.py --fasta_file sample.fasta --saved_folder sample_dir`
     - `python generate_features/coordinate.py --pdb_file sample.pdb --saved_folder sample_dir --out_fn sample/coordinate.pt`
     - `python generate_features/pair.py --coordinate_file sample/coordinate.pt --saved_folder sample_dir --out_fn sample/pair.pt`
     - For the more feature-rich pair matrix use `pair_advanced.py`.

4. Train / run ablations:
   - `python train_code/train_geodtm_ablation.py --train_csv ... --features_dir ... --out_dir ... [ablation flags]`
   - `python train_code/train_geodtm_radius_ablation.py --radius 0 --train_csv ...` (radius windowing)
   - In our paper, we used passed the `seed` arg with values 0-9 and retrained/evaluated performancw with each of these for each ablation

5. Analyze structural mismatch vs model error:
   - `python scripts/analyze_graph_vs_performance.py --features_dir ... --tmalign_bin /path/to/TMalign ...`

## Dependencies (non-exhaustive)
- Python 3.8+
- PyTorch (version matching CUDA on your system)
- Biopython
- esm / esmfold (Facebook Research ESM)
- numpy, pandas, scipy, matplotlib, tqdm
- TM-align (external binary) for structural comparisons

Install core Python deps with your preferred package manager (conda/pip). ESMFold has additional requirements; see the ESM repository for details.

## Notes / caveats
- Many scripts are written to be restart-safe and to support SLURM-array-style sharding; read the script-level docstrings for sharding semantics.
- There are multiple variants of structural files (ESMFold vs Rosetta); training scripts can prefer rosetta files when available.
- Fixed-dimension behavior for concatenated fixed features is deterministic in the training scripts: fixed feature length = 7 (physchem) + (pH present?) + (pLDDT present?).
- The code in this repo is the research code used for experiments and is under active development; expect some rough edges and scripts tailored for our dataset layout.

```
