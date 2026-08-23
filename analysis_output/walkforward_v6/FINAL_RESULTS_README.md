# Final Walk-Forward Results Guide

## Experiment validity

Before using any result, check `completion_audit.txt`. The valid final audit is 95 expected, 95 completed and 0 missing. Also check `formal_integrity_audit.txt`; the valid final audit reports zero errors and `PASS`.

Design:

- five test years: 2021–2025
- five controlled model variants
- three seeds for all variants
- two additional seeds for Full IQN and Parameter-Matched Double DQN
- 95 total training tasks
- pipeline version 6

## Primary result files

- `walkforward_summary.xlsx` — compact original aggregation workbook
- `fold_results.csv` — one row per model × seed × annual fold
- `pooled_seed_results.csv` — genuine 2021–2025 pooled path metrics by model and seed
- `task_results/` — 95 authoritative task JSON files with daily OOS paths

## Checkpoints

- `seed_42/`, `seed_123/`, `seed_2024/` — all five variants
- `seed_7/`, `seed_999/` — Full IQN and matched DQN only

These are the final trained checkpoints. Do not substitute checkpoints from `archive_legacy/`.

## Paper-ready analysis

All paper tables, figures and narrative drafts are in `paper_analysis/`.

### Recommended main tables

- `benchmark_summary.csv`
- `model_summary_uncertainty.csv`
- `baseline_comparisons.csv`
- `yearly_summary.csv`
- `mechanism_summary.csv`
- `cost_sensitivity_summary.csv`
- `training_sign_baseline_summary.csv`
- `regime_component_shift.csv`
- `cost_source_summary.csv`
- `block_length_robustness.csv`
- `fold_win_rates.csv`
- `regime_high_low_paired_tests.csv`
- `research_question_evidence_map.csv`

### Recommended figures

- `fig_research_framework.png`
- `fig_equity_vs_baselines.png`
- `fig_yearly_sharpe.png`
- `fig_seed_year_heatmap.png`
- `fig_gross_cost_net.png`
- `fig_cost_sensitivity.png`
- `fig_cost_decomposition.png`
- `fig_regime_distribution_shift.png`
- `fig_regime_component_shift.png`
- `fig_action_proportions.png`

### Thesis drafting material

- `thesis_results_discussion_draft.md`
- `deepening_results_discussion_draft.md`
- `threats_to_validity_draft.md`
- `research_question_evidence_map.md`

### Audit-level data (normally not pasted directly into the thesis)

- files ending in `_daily.csv`
- files ending in `_seed_results.csv`
- `daily_analysis_panel.csv`
- `rl_daily_gross_cost_net.csv`
- `core_fold_pair_results.csv`

These support reproducibility and the summary tables but are too detailed for the main text.

## Important reporting convention

`CumReturn`, `AnnReturn` and `MaxDrawdown` are expressed in additive spread-return P&L units. No portfolio capital, futures margin base or compounded portfolio return is defined. Do not present these values as conventional investment-return percentages.
