# Extended Robustness and Mechanism Results

## Transaction-cost sensitivity and break-even analysis

The fixed-policy cost sensitivity analysis confirms that Full IQN cannot break even at any non-negative transaction-cost level because its mean gross P&L is already negative (-0.130). Double DQN has positive mean gross P&L, but its break-even cost multiplier is only 0.589 relative to the main specification. Equivalently, its break-even lambda is approximately 0.000295, compared with the maintained value of 0.000500. DQN would therefore require a reduction of approximately 41.1% in the assumed cost rate to reach zero mean net P&L, holding the learned policy fixed. This result distinguishes statistical signal extraction from economic implementability: DQN contains some gross predictive information, but not enough to survive the main trading-cost assumption.

## Ex-ante directional benchmark

The Training-Sign Directional Baseline selects the next test year's position using only the sign of mean spread returns observed through the end of the corresponding training window. The historical mean was slightly negative in every fold, so the rule selected a short position for all five test years. It consequently reproduced the always-short diagnostic result, with cumulative net P&L of 0.240 and Sharpe ratio of 0.235. Unlike an ex-post choice of direction, this baseline is implementable without test-period information. However, the training means were economically very small, so the directional rule should be interpreted as a deliberately simple benchmark rather than a robust forecasting model.

## Regime-component distribution shift

All regime components shifted downward out of sample, but the largest standardised change occurred in RegimeCVScore (standardised mean difference -2.134), followed by RegimeStabilityScore (-1.796). The corresponding shifts for the remaining components were RegimeHLScore (-0.939) and RegimeEGScore (-0.524). Thus, the collapse in the composite stability measure was driven most strongly by deterioration in hedge-ratio stability, with additional weakening in the half-life and cointegration components. This identifies the economic source of the covariate shift rather than treating the aggregate score as a black box.

## Sources of transaction costs

For Full IQN, discrete action changes accounted for 62.9% of total costs and hedge-ratio rebalancing for 36.3%; forced annual closure contributed the small remainder. For Double DQN, the corresponding shares were 63.7% and 35.4%. The dominant source was therefore frequent changes in the discrete trading decision, although continuous adjustment of the hedge leg contributed more than one-third of costs. Reducing turnover would need to address both action persistence and dynamic hedge rebalancing.

## Inference robustness and fold-level consistency

The conclusion that DQN has better point estimates but no statistically established superiority is unchanged for circular-block lengths of 5, 10, 20, 40 and 60 trading days. Across all specifications, the confidence interval for the DQN-minus-IQN Sharpe difference crosses zero; the same is true for DQN relative to the Z-score rule. At the seed-by-year level, DQN exceeded IQN in 15 of 25 comparisons (60.0%). It exceeded the annual Z-score Sharpe in 13 of 25 cases (52.0%). Positive Sharpe ratios occurred in only 8 DQN cases and 5 IQN cases. These win rates reinforce that no model has consistent dominance.

## Conditional regime effects

Using the balanced, ex-post descriptive stability tertiles within each test year, DQN's mean High-minus-Low Sharpe difference was 0.119, with a 95% interval from -1.018 to 1.256 and exact sign-flip p-value 0.875. For Full IQN, the corresponding difference was -0.042, with interval [-0.559, 0.475] and p-value 0.875. Neither model therefore shows statistically reliable High-versus-Low regime improvement. Because these groups are defined ex post for descriptive balance, the comparison is evidence of conditional heterogeneity, not a pre-specified real-time trading rule or a monotonic causal effect.

## Overall implication

The additional analyses sharpen the negative result into three distinct mechanisms. First, Full IQN fails at the gross-signal stage, whereas DQN learns a positive but economically small gross signal. Second, discrete action turnover and hedge-leg rebalancing jointly remove the DQN gross advantage. Third, a large out-of-sample shift—especially in hedge-ratio stability—weakens the interpretation and generalisability of the regime inputs. These findings explain why greater representational complexity did not translate into robust economic performance.
