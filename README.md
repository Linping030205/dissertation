# Regime-Aware Distributional Reinforcement Learning for Commodity Spread Trading

This repository contains the source code, archived formal outputs and result-audit materials for a dissertation study of regime-aware distributional reinforcement learning applied to the soybean--soybean-meal spread. The experiment compares four IQN configurations and a parameter-matched Double DQN under an expanding-window out-of-sample design.

The public repository is a **code and archived-result verification package**. It is not a self-contained raw-data-to-training reproduction package because the licensed source workbooks, the derived feature workbook and trained checkpoints are not redistributed. The included formal daily result paths can nevertheless be independently checked without those files, as described below.

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

A valid release ends with `PASS`, confirms 95 unique tasks and recomputes every task's performance metrics from its daily P&L path. It also checks the corrected maximum-drawdown convention, including initial equity `E_0 = 0`, and verifies the locked headline values in the dissertation result bundle. This is the quickest public check that the archived formal outputs are internally consistent.

This command verifies the archived outputs; it does **not** rerun feature engineering, retrain the agents or independently recreate the results from raw market data.

## Install and run the code

The reference environment used Python 3.9.6 and PyTorch 2.1.0. For a CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For UCL Myriad, use `requirements-myriad.txt` and load the cluster-provided GPU PyTorch module as described in `MYRIAD_README.md`.

Full retraining additionally requires authorised copies of the licensed source workbooks and the final derived feature file. Place `analysis_output/spread_features_SSM.xlsx` at the documented relative path, then run a manifest task, for example:

```bash
python run_walkforward_task.py --task-id 1 --project-root .
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete workflow and limitations.

## Data availability

Raw market data exported from Refinitiv Workspace and the derived feature workbook are not redistributed because their redistribution rights have not been established. The export files do not contain enough metadata to reconstruct the exact continuation-series settings, so those settings cannot be recovered from this repository alone. Checkpoints are also excluded because they are large and contain machine-specific provenance paths. The repository does include all 95 daily out-of-sample strategy paths, aggregated non-sensitive result tables, figures, the deterministic manifest and source code.

`CumReturn`, `AnnReturn` and `MaxDrawdown` are additive spread-return P&L quantities. They are not portfolio-capital percentages, because no capital or futures-margin base is defined.

## Reproducibility scope

There are two distinct reproducibility levels:

1. **Archived-result verification:** immediately available from the included daily task JSON files using `verify_release.py`. This is the reproducibility level supported by the public package itself.
2. **End-to-end regeneration and retraining:** requires authorised users to supply the licensed input workbooks, recreate or supply the exact final derived feature workbook, and rerun the manifest tasks. This level is not self-contained in the public package.

The saved `analysis_output/walkforward_v6/formal_integrity_audit.txt` records the stronger local audit performed against the original feature file and checkpoint hashes before this public package was created. The accompanying `audit_formal_results.py` is retained for transparency, but it is expected to fail in the public package until the omitted `analysis_output/spread_features_SSM.xlsx` and required local artefacts have been restored. Use `verify_release.py` for the public, dependency-free integrity check.

## Interpretation boundaries

- The formal experiment is conditional on the soybean--soybean-meal pair selected using information available through 2019; it is not an end-to-end dynamic pair-reselection strategy.
- `cvar_alpha = 1.0` is used throughout the main experiment. IQN therefore provides a distributional value-function approximation but uses a risk-neutral action-selection rule; the reported Full IQN is not a CVaR risk-averse agent.
- The transaction-cost coefficient is an additive two-leg spread-turnover proxy, not a contract-calibrated estimate of commissions, bid--ask spread, market impact, margin usage or futures roll cost.
- The reported negative results are specific to the maintained data, pair, structural episode, cost proxy and training protocol. They do not establish that IQN or regime-aware reinforcement learning is generally ineffective.

## Citation and licence

If this repository supports related work, please cite the accompanying dissertation. No software licence is granted by this repository unless a licence file is added by the author.
