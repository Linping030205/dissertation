# Threats to Validity

## Internal validity

Network capacity is controlled closely and state ablations preserve the 92-dimensional input, but reward ablations intentionally change the training objective. Consequently, reward comparisons identify the effect of alternative objectives rather than performance under an identical loss. Random initialisation is addressed through three seeds for all variants and five seeds for the core comparison, although residual optimisation uncertainty remains.

## External validity

The evidence is based on one commodity spread, daily observations and the 2021–2025 out-of-sample period. Results may not generalise to other commodities, intraday horizons, alternative spread constructions or market eras. The unusually low stability distribution in the test period is itself a substantive finding but limits extrapolation to more stable environments.

## Construct validity

RegimeStabilityScore is an engineered proxy based on cointegration, half-life and hedge-ratio stability; it is not a directly observed economic regime. The absence of training-anchored High-regime test observations indicates that fixed historical cut-offs do not retain the same interpretation under distribution shift. Within-year tertiles are therefore used only for descriptive analysis, not training or model selection.

## Statistical conclusion validity

Five test years and three to five seeds provide substantially more evidence than a single split, but confidence intervals remain wide. Daily observations are serially dependent, so circular block bootstrap and block sign-flip procedures are used. Conclusions are checked across alternative block lengths and multiple-comparison-adjusted inference; failure to reject equality must not be interpreted as proof that models are identical.

## Economic validity

P&L is measured in additive spread-return units because no portfolio-capital, contract-multiplier or margin convention is defined. Costs consistently price discrete action changes and hedge-ratio rebalancing, but remain a turnover proxy rather than realised bid–ask, roll and market-impact costs. Always-long and always-short strategies are diagnostic directional benchmarks; the Training-Sign benchmark is the corresponding ex-ante implementable comparison.
