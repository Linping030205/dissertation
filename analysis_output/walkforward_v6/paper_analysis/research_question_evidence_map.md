| Research question | Evidence | Conclusion |
|---|---|---|
| RQ1: Does regime state information help? | Full IQN vs w/o Regime-in-State plus training-only feature correlations | No stable mean-performance gain. Training-only correlations do not support simple pairwise redundancy with base volatility or hedge-ratio features, although the Regime block has substantial internal dependence (EG-HL mean Spearman rho about 0.83). |
| RQ2: Does regime reward shaping help? | Full IQN vs w/o Regime-in-Reward | Effects are seed-dependent and interact with state information. |
| RQ3: Does explicit regime modelling help overall? | Full IQN vs No Explicit Regime IQN | No robust return improvement; regime treatment changes variability and behaviour. |
| RQ4: Does IQN outperform scalar-value DQN? | Parameter-matched IQN–DQN, five seeds | DQN has better point estimates, but paired uncertainty intervals cross zero. |
| RQ5: Are learned policies economically viable? | Cash, Z-score, directional baselines and cost sensitivity | DQN has positive gross signal but costs erase it; IQN is negative before costs. |
| RQ6: Why does OOS performance deteriorate? | Year, regime, action and cost diagnostics | Temporal heterogeneity, regime covariate shift and turnover limit generalisation. |
