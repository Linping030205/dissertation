# Additional mechanism-analysis draft

## Training-only feature dependence

Figure `fig_feature_correlation.png` reports the mean Spearman correlation
matrix across the five expanding training samples, rather than using the full
sample. The explicit Regime features are only weakly associated with the six
base market features. The largest absolute cross-group correlations are about
0.20 (RegimeHLScore versus soybean-meal volume and RegimeCVScore versus
soybean-meal volume), while RegimeCVScore versus RollStd20 is approximately
-0.18. Consequently, the absence of a performance gain from Regime-in-State
cannot be attributed confidently to simple pairwise duplication by volatility
or DeltaBeta5d. A more defensible interpretation is that the Regime variables
provide statistically distinct information, but that the learned policy did
not convert it into a stable out-of-sample decision advantage.

There is, however, substantial dependence within the Regime block itself:
RegimeEGScore and RegimeHLScore have a mean Spearman correlation of about 0.83.
This is economically plausible because stronger mean reversion and stronger
cointegration evidence are related properties, but it also means that the
three added channels should not be described as three independent sources of
information. The correlation analysis is descriptive and does not establish
non-linear redundancy or causal feature importance.

## Turnover and holding behaviour

The learned policies trade much more frequently than the deterministic
Z-score rule. Across seeds, Full IQN makes about 327 economically counted
position changes over the five annual test folds (approximately 66 per year),
the parameter-matched DQN about 355 (72 per year), and the state ablation about
397 (80 per year). Their median non-zero holding spell is only about two
trading days. The Z-score rule makes 62 economically counted changes and has a
median non-zero holding spell of nine days. Counts include the forced close at
the end of each annual fold; direct reversals are counted as one position
decision but economically close and reopen exposure through the shared
two-leg transaction-cost formula.

This evidence strengthens the gross-cost-net mechanism. DQN produces positive
mean gross P&L (about 0.223 spread-return units) but incurs about 0.378 units of
cost, yielding mean net P&L of about -0.155. The state ablation also produces
positive gross P&L (about 0.107) but incurs about 0.432 of cost. Full IQN is
different: its mean gross P&L is already negative (about -0.130) before its
roughly 0.358 cost is deducted. Thus the DQN result is primarily a failure of
economic implementation efficiency, whereas Full IQN combines weak gross
signal extraction with substantial turnover.

Cost-to-gross ratios must be interpreted cautiously when signed gross P&L is
close to zero. The main table should therefore report gross P&L, total cost and
net P&L separately; the ratio is a supplementary diagnostic rather than a
stand-alone performance measure.

## Time evolution of the Regime components

Figure `fig_regime_component_timeline.png` shows that the post-2020 decline in
the composite stability score is not a single binary switch. Cointegration
evidence is frequently close to zero, the half-life component shifts downward,
and hedge stability becomes both lower and more erratic. Temporary recoveries
in individual components do not restore the joint relationship consistently.
This explains why a diagnostic regime measure can identify relationship
deterioration without necessarily providing the policy with a profitable new
direction: recognising that the old equilibrium is unreliable is not the same
as forecasting the sign of the next spread return.

## Spread-return tails and distribution shift

The spread return itself is strongly non-normal, so the motivation for
distributional value modelling does not need to be inferred from the two legs.
Across the 1,251 formal out-of-sample evaluation dates, spread returns have
skewness about -0.92 and excess kurtosis about 9.75; the Jarque-Bera test
strongly rejects normality. The lower-tail five-percent mean is approximately
-0.0314, compared with an upper-tail five-percent mean of approximately
+0.0276. The asymmetry is particularly pronounced in 2021 and 2022, whereas
2023 is much closer to symmetric and only mildly leptokurtic.

This supports the ex-ante motivation for IQN but does not imply that IQN must
outperform DQN. Under cvar_alpha=1.0, the policy averages the full learned
quantile distribution and remains risk-neutral. Estimating the distribution is
also more demanding than estimating a scalar conditional mean under limited,
non-stationary historical data. The results therefore distinguish a valid
motivation for distributional modelling from evidence that the chosen
distributional agent generated superior decisions.
