# Reproducibility guide

## 1. Public result verification

Run from the repository root:

```bash
python verify_release.py
```

The verifier checks:

- exactly 95 task JSON files;
- uniqueness of `(test_year, seed, variant)`;
- the expected five folds and variant-specific seed coverage;
- daily dates, P&L and position arrays have equal lengths;
- all headline metrics are reconstructed from the daily paths;
- the corrected maximum-drawdown convention includes initial equity `E_0 = 0`;
- the headline values in `model_summary_uncertainty.csv` match the locked dissertation result bundle.

The verification script only uses the Python standard library.

## 2. Reference software

- Python 3.9.6
- PyTorch 2.1.0 (cluster build reported `2.1.0+cu121`)
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.13.1
- statsmodels 0.14.4
- Gymnasium 1.0.0
- Matplotlib 3.9.4
- seaborn 0.13.2
- openpyxl 3.1.5

CUDA is not required for verification. GPU availability must be checked inside an allocated compute job rather than on a login node.

## 3. End-to-end experiment

The training pipeline expects the final derived feature workbook at:

```text
analysis_output/spread_features_SSM.xlsx
```

The upstream scripts expect the licensed commodity workbooks in the project root. They are intentionally absent from the public repository. After supplying authorised copies:

1. Inspect or regenerate the 95-row manifest with `make_walkforward_manifest.py`.
2. Run individual tasks with `run_walkforward_task.py` or the local/cluster queue scripts.
3. Aggregate with `collect_walkforward_results.py`.
4. Run the integrity audit locally with `audit_formal_results.py` while the feature file and checkpoints are present.
5. Generate the paper analysis with the `paper_*analysis.py` scripts.

The authoritative experimental design is encoded in `walkforward_manifest.tsv`, `run_walkforward.py`, `train.py`, `rl_env.py` and `evaluate.py`.

## 4. Formal result policy

Only `analysis_output/walkforward_v6/` belongs to the formal dissertation experiment. Earlier development runs, smoke tests and superseded checkpoints are not included.

The formal task set contains three seeds for every model. Full IQN and Parameter-Matched Double DQN have two additional seeds, producing 95 rather than 75 tasks. Comparisons must respect this unequal seed coverage and the estimand stated in the paper.

## 5. Metric convention

Daily P&L is additive. For daily values `r_t`, the drawdown calculation uses:

```text
E = [0, cumsum(r_1), ..., cumsum(r_T)]
MaxDrawdown = min(E - cumulative_max(E))
```

This prevents an opening loss from being treated as the initial peak. No compounded portfolio equity or capital-normalised percentage return is claimed.

