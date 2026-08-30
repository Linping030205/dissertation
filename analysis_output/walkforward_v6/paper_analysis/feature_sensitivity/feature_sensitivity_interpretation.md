# Grouped out-of-sample permutation sensitivity

## Scope

This diagnostic replays fixed final-v6 checkpoints; it does not retrain or
select a model. It covers Full IQN and w/o Regime-in-Reward IQN on common seeds
42, 123 and 2024 across the five 2021--2025 test folds. Each selected feature
is replaced as a complete ten-day history using 20-day block permutations (20
repetitions). Position state, unselected inputs, realised returns and the
two-leg cost path remain unchanged. Positive Sharpe loss means that permutation
reduced performance relative to the unperturbed checkpoint.

## Main results

- The policies use the Regime inputs behaviourally. Jointly permuting all
  three Regime histories changes 28.8% of Full-IQN actions and 28.3% of
  w/o-Regime-in-Reward actions on average. This rules out the simple explanation
  that the trained policies completely ignore the engineered channels.
- Behavioural use is not the same as economic value. For Full IQN, all-Regime
  permutation has mean Sharpe loss -0.074 and mean net-P&L loss -0.004; on
  average the perturbed policy is therefore slightly better, not worse.
- Without reward shaping, individual EG and half-life histories show positive
  mean Sharpe losses of 0.219 and 0.255, with action disagreement of 13.2% and
  15.1%. The corresponding Full-IQN losses are -0.059 and -0.052. This sign
  reversal is consistent with the dissertation's context-dependence and
  possible non-additivity interpretation.
- The joint EG-plus-half-life perturbation does not produce a larger behavioural
  response than the individual effects (16.5--16.8% disagreement), consistent
  with overlapping information. Its performance effect is nevertheless highly
  dependent on reward specification: mean Sharpe loss is -0.184 with shaping
  and +0.219 without shaping.
- All-base perturbation changes roughly half of actions (51.1% and 50.3%),
  whereas all-Regime perturbation changes about 28%. The base block therefore
  exerts greater aggregate control over action choice, although dimensionality
  differs (six versus three features) and this is not a normalised comparison.
- Effects vary materially by seed and year. Across the 15 checkpoint/fold units,
  all-Regime Sharpe loss is positive in 40.0% of Full-IQN units and 46.7% of
  reward-off units. The analysis therefore does not support a stable ranking of
  feature importance.

## Interpretation boundary

This is fixed-policy, out-of-sample permutation sensitivity, not causal feature
attribution. Permutation can create input combinations outside the joint
distribution observed during training, and it does not answer what performance
would have been had a feature been removed and the policy retrained. The result
can replace claims that the networks' input use was wholly unexamined, but it
does not justify saying that a feature caused performance or that a retrained
component ablation would have the same effect.
