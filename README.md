# Regime-Aware Distributional Reinforcement Learning for Commodity Spread Trading

This repository contains the code and audit materials for a dissertation study of regime-aware distributional reinforcement learning applied to the soybean--soybean-meal spread. The experiment compares four IQN configurations and a parameter-matched Double DQN under an expanding-window out-of-sample design.

## Experiment snapshot

- Pipeline version: 6
- Test folds: 2021, 2022, 2023, 2024 and 2025
- Core seeds for all five configurations: 42, 123 and 2024
- Additional seeds for Full IQN and matched DQN: 7 and 999
- Formal training tasks: 95
- Actions: short, flat and long
- State size: 92 for every configuration; regime channels are zero-masked in state ablations
- Evaluation basis: additive net spread P&L, including transaction costs

The repository reports a negative-result study. None of the RL configurations establishes a robust net-performance advantage over the simple directional benchmark. The analysis separates gross signal, transaction-cost erosion, turnover, regime distribution shift and random-seed uncertainty.

## Repository contents

```text
.
|-- rl_agent.py / rl_env.py       model and environment definitions
|-- train.py / evaluate.py        training and evaluation
|-- run_walkforward*.py           expanding-window experiment runners
|-- paper_*analysis.py            paper tables, figures and diagnostics
|-- verify_release.py             public-result integrity check
|-- walkforward_manifest.tsv      authoritative 95-task manifest
|-- analysis_output/
|   `-- walkforward_v6/
|       |-- task_results/         95 daily out-of-sample result JSON files
|       |-- paper_analysis/       paper-ready CSV tables and PNG figures
|       |-- fold_results.csv
|       `-- pooled_seed_results.csv
`-- docs/REPRODUCIBILITY.md
```

## Verify the reported result bundle

The public verification path does not require PyTorch or the licensed market data:

```bash
python verify_release.py
```

A valid release ends with `PASS`, confirms 95 unique tasks and recomputes every task's performance metrics from its daily P&L path. This is the quickest way to verify that the included paper tables are tied to the formal result set.

## Install and run the code

The reference environment used Python 3.9.6 and PyTorch 2.1.0. For a CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For UCL Myriad, use `requirements-myriad.txt` and load the cluster-provided GPU PyTorch module as described in `MYRIAD_README.md`.

Full retraining additionally requires the licensed source workbooks and the derived feature file. Place `analysis_output/spread_features_SSM.xlsx` at the documented relative path, then run a manifest task, for example:

```bash
python run_walkforward_task.py --task-id 1 --project-root .
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete workflow and limitations.

## Data availability

Raw market data and the derived feature workbook are not redistributed because their redistribution rights have not been established. Checkpoints are also excluded because they are large and contain machine-specific provenance paths. The repository does include all 95 daily out-of-sample strategy paths, aggregated non-sensitive result tables, figures, the deterministic manifest and source code.

`CumReturn`, `AnnReturn` and `MaxDrawdown` are additive spread-return P&L quantities. They are not portfolio-capital percentages, because no capital or futures-margin base is defined.

## Reproducibility scope

There are two reproducibility levels:

1. **Result verification:** immediately available from the included daily task JSON files using `verify_release.py`.
2. **End-to-end retraining:** available to authorised users who supply the licensed input workbooks and derived feature file.

The saved `formal_integrity_audit.txt` records the stronger local audit performed against the original feature file and checkpoint hashes before this public package was created.

## Citation and licence

If this repository supports related work, please cite the accompanying dissertation. No software licence is granted by this repository unless a licence file is added by the author.

