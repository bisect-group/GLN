# GLN EMULaToR Adaptation Plan

## 1. Baseline Summary From README

- Baseline: GLN Conditional Graph Logic Network; retrosynthesis prediction.
- Docs inspected: `README.md`, `gln/data_process/`, `gln/training/scripts/run_mf.sh`, and `gln/test/`.
- Native flow: data-processing steps 0/0.1/1/2/3/4/5, `run_mf.sh` for training, and `main_test.py`/`test_all.sh` for evaluation.
- Default model settings remain those in `run_mf.sh` (mean-field, latent 128, embedding 256, 64 negatives, three layers, 3000 iterations per validation, top-k/beam 50, GPU 0).
- The README describes roughly ten 3000-iteration validation cycles, while the script leaves the cycle limit to the Python entrypoint; the wrapper records this discrepancy and uses the documented script defaults.

## 2. Expected Input Format

- GLN raw CSV: `id,class,reactants>reagents>production`.
- EMULaToR source: `/home/adhil/github/EMULaToR/data/processed/datasets/reaction_outcome_dataset`.
- Required source fields: `rxn_smiles`, `atom_mapped_rxn_smiles`, `reactants`, `products`.
- Mapping failures are excluded and written to an audit report; confidence is retained as metadata.

## 3. Featurization and Preprocessing Path

- The adapter writes mapped GLN CSV views, then invokes GLN's native steps 0, 0.1, 1, 2, 3, 4, and 5.
- Schema-5 preprocessing is a content-addressed stage DAG. Each completed stage has a manifest with direct-input and implementation fingerprints, output hashes/sizes/counts, dependency keys, and validation results. The schema-5 root is separate from the old schema-4 namespace, which is quarantined rather than adopted by default.
- Per-item worker results are immutable microshards. A persistent `imap_unordered(..., chunksize=1)` pool runs each parallel stage and resumes only missing shards; serial work is limited to deterministic merging, validation, and atomic publication.
- Raw CSV views retain every successfully atom-mapped input row. Template-extraction failures are written to audits and never cause rows to be removed from train/validation/test views.
- Static graph/preprocessing artifacts are reused; learned model state is never cached as a feature.

## 4. Training and Evaluation Entrypoints

- Training delegates to `gln/training/main.py` through the documented `run_mf.sh` argument set.
- Evaluation delegates to `gln/test/main_test.py` and preserves native cumulative top-k evaluation.
- Wrappers additionally emit top-1 and top-10 exact-match summaries and select checkpoints by validation top-1.

## 5. Dataset and Split Mapping

- Use `random_splits_grouped_rxn_smiles` (41,108 rows), `uniprot_time_splits` (41,108 rows), and `drfp_tanimoto_splits/threshold_0.11` (39,732 rows).
- Do not use the stale `random_splits` directory.
- Copy validated split views to `/home/adhil/github/EMULaToR/data/processed/baselines/GLN`.
- Preserve atom maps supplied by the user; wrappers never run LocalMapper.

## 6. Files To Add Under `emulator_bench/`

- `README.md`, `utils.py`, and `dataset_adapter.py` for validation, mapping, manifests, and CSV generation.
- `cache_features.py` and `stage_cache.py` for schema-5 durable stage manifests, validation-only runs, narrowly-scoped invalidation, atomic publication, and resumable split-specific caches.
- `train.py`, `evaluate.py`, `tune_optuna.py`, and `launch_parallel.py` for training, metrics, safe tuning, and multi-GPU orchestration.
- `torch_compat.py` for the bench-local PyTorch segmented-log-softmax fallback required by current PyTorch versions.
- `smoke_test.py` and `commands.txt` for reproducible end-to-end verification.

## 7. Minimal Edits Outside `emulator_bench/`

- `Plan.md` is the only root-level addition. GLN model, trainer, and preprocessing code remain unchanged.

## 8. Exact Execution Plan

1. Validate the refreshed master and selected split parquets, excluding only rows whose atom mapping itself failed.
2. Generate mapped GLN CSV views and manifests under the EMULaToR GLN baseline directory.
3. Run schema-5 preprocessing once per split with `--legacy-cache-policy fresh`; quarantine schema-4 data and validate every stage before publishing `preprocess.complete`.
4. Launch native GLN training with the original settings, AMP, checkpoint resume, and Rich progress.
5. Evaluate native top-k plus top-1/top-10 metrics and write consolidated results.
6. Run Optuna only over safe training hyperparameters and distribute trials across requested GPUs.
7. Run the small end-to-end smoke test with `CUDA_VISIBLE_DEVICES=3` in conda environment `test`.
