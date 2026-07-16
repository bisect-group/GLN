# GLN EMULaToR wrappers

These wrappers preserve GLN's native mapped-reaction preprocessing, training, and retrosynthesis evaluation while materializing EMULaToR split-specific views and manifests.

The refreshed dataset uses `random_splits_grouped_rxn_smiles`, `uniprot_time_splits`, and `drfp_tanimoto_splits/threshold_0.11`. The old `random_splits` directory is stale and is not used. Atom maps are read from `atom_mapped_rxn_smiles`; LocalMapper is never called. Rows with failed atom mapping are excluded and recorded in `manifest.json`. A template-extraction failure is audited instead: it never rewrites or removes a row from any raw split view.

Preprocessing uses the schema-5 cache contract. Each of the ten stages has a content-addressed manifest containing direct dependency keys, implementation/ABI identity where relevant, output SHA-256 and byte-size metadata, and validation results. Results are atomically published and verified before a cache hit is accepted. Schema-4 data is not adopted by default: `--legacy-cache-policy fresh` quarantines it and creates a clean `schema-5/namespaces/<split>` tree. `--resume` reuses validated stages, `--force-stage` invalidates only that node and true descendants, and `--verify-cache-only` performs no computation.

Run from the GLN root with conda environment `test`. Raw views are under each schema-5 namespace's `dropbox/reaction_outcome_dataset/`; native cooked artifacts are under `cooked_reaction_outcome_dataset/`; copied split parquets and manifests are under the selected split-group directory.

See `../commands.txt` for exact commands. `cache_features.py` invokes GLN steps 0, 0.1, 1, 2, 3, 4, and 5. `train.py` and `evaluate.py` delegate to GLN's original Python entrypoints. `torch_compat.py` supplies a bench-local jagged log-softmax fallback for current PyTorch.

For the complete caching phase, run `CUDA_VISIBLE_DEVICES=3 conda run -n test python -m emulator_bench.run_all --gpus 0 --split-groups all --num-cores 64 --center-timeout 60 --prepare-only --resume --legacy-cache-policy fresh`. Physical GPU 3 is exposed as local GPU 0; preprocessing remains CPU-only. Rerun with `--verify-cache-only` to validate manifests and native dumps without computing. Results are written to `results/per_job_metrics.jsonl`, `results/summary.csv`, and `results/summary_by_split.csv` for training runs.

The input master and split parquet files must contain `atom_mapped_rxn_smiles`; the adapter deliberately does not run a mapper or substitute unmapped reactions.
