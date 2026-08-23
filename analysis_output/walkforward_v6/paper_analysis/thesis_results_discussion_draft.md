# Supplementary Results and Discussion Draft

## Transaction-cost attribution

Decomposing out-of-sample performance into gross spread P&L and transaction costs reveals different failure mechanisms across the two core models. Full IQN generated a mean gross P&L of -0.130 across five seeds and incurred mean costs of 0.358, resulting in net P&L of -0.487. Its performance was therefore already negative before costs. By contrast, the parameter-matched Double DQN generated positive mean gross P&L of 0.223, but incurred costs of 0.378, producing net P&L of -0.155. DQN therefore extracted some directional signal, but its economic magnitude was insufficient to cover turnover. IQN exhibited both weak gross signal extraction and substantial cost exposure.

## Policy behaviour

Full IQN allocated, on average, 36.0% of test days to short positions, 17.1% to cash, and 46.9% to long positions. Double DQN allocated 32.6%, 24.3%, and 43.0% respectively. Because the realised spread exhibited a favourable short-direction drift over 2021–2025, failure to maintain sufficient short exposure helps explain why both learned policies underperformed the diagnostic always-short benchmark. This does not establish that the short direction was predictable ex ante; rather, it identifies a source of ex-post opportunity cost.

## Regime distribution shift

The stability-score distribution shifted materially between the historical training sample and the 2021–2025 out-of-sample period. Applying fold-specific 33rd and 67th percentile thresholds estimated only from the expanding training window classified approximately 91.6% of out-of-sample observations as Low stability, 8.4% as Mid stability, and 0.0% as High stability. No High-stability observations occurred under these training-anchored thresholds. Consequently, the regime-aware reward was evaluated predominantly in a part of the state space associated with persistent instability. This provides direct evidence of covariate shift and limits the extent to which the training-period regime mapping could generalise.

## Seed and year heterogeneity

The seed-by-year results show that performance differences were not driven by a single random seed, but neither model was consistently superior across calendar periods. Double DQN produced positive mean annual Sharpe ratios in 2022 and 2025, whereas Full IQN was positive only in 2025. Both models performed poorly in 2024, and Full IQN experienced particularly weak outcomes in 2023. This temporal reversal is consistent with a non-stationary learning problem in which relationships learned from an expanding historical window do not remain stable in the subsequent year.

## Interpretation

Taken together, the diagnostics show that the negative net results do not have a single cause. DQN learned a positive gross signal, but frequent reallocation consumed more than the signal generated; IQN was negative even before costs. Both policies varied materially across years and were evaluated under a substantial shift in the regime distribution. The richer return-distribution representation of IQN therefore did not overcome the more fundamental generalisation problem. Under the present risk-neutral setting, model complexity was not a substitute for a stable and economically large predictive signal.

## Limitations and reporting caution

The always-short strategy is a diagnostic benchmark, not evidence of an ex-ante implementable forecasting rule. The within-test-year stability tertiles are likewise descriptive and were not used in training or model selection. Return and drawdown statistics are expressed in additive spread-return units because no portfolio capital or margin base is defined. The transaction-cost specification is a consistent two-leg turnover proxy rather than a complete futures execution model. These restrictions should be stated explicitly when interpreting economic significance.
