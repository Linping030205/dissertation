"""Additional robustness and mechanism analyses for the dissertation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

from commodity_utils import spread_trade_cost
from evaluate import compute_metrics


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
PA = WF / "paper_analysis"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"
LAMBDA = 0.0005
CORE = ["Full IQN", "Parameter-Matched Double DQN"]
RNG_SEED = 260822
N_BOOT = 3000


def sharpe(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return 0.0 if sd < 1e-14 else float(x.mean()/sd*np.sqrt(252))


def circular_ix(n, block, rng):
    starts = rng.integers(0, n, int(np.ceil(n/block)))
    return np.concatenate([(np.arange(block)+s) % n for s in starts])[:n]


def infer(a, b, block, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = circular_ix(len(a), block, rng)
        boots[i] = sharpe(a[ix])-sharpe(b[ix])
    d = a-b
    pieces = np.array_split(d, np.arange(block, len(d), block))
    sums = np.array([x.sum() for x in pieces])
    signs = rng.choice([-1., 1.], (N_BOOT, len(sums)))
    obs = abs(sums.sum())
    p = (1+(np.abs(signs@sums) >= obs).sum())/(N_BOOT+1)
    return sharpe(a)-sharpe(b), *np.percentile(boots, [2.5, 97.5]), p


def baseline_fold(data, year, direction):
    test = data.loc[f"{year}-01-01":f"{year}-12-31"]
    prev = 0
    rows = []
    for i in range(len(test)-1):
        pos = direction
        beta = float(test.RollingBeta.iloc[i])
        beta_prev = float(test.RollingBeta.iloc[i-1]) if i else beta
        gross = pos*float(np.nan_to_num(test.SpreadReturn.iloc[i+1]))
        cost = spread_trade_cost(LAMBDA, pos, prev, beta, beta_prev)
        if i == len(test)-2 and pos:
            cost += spread_trade_cost(LAMBDA, 0, pos, beta, beta)
        rows.append({"test_year": year, "date": test.index[i+1],
                     "position": pos, "gross_pnl": gross,
                     "cost": cost, "net_pnl": gross-cost})
        prev = pos
    return pd.DataFrame(rows)


def exact_sign_flip(values):
    x = np.asarray(values, float)
    observed = abs(x.mean())
    vals = [abs(np.mean(x*np.asarray(s)))
            for s in itertools.product([-1., 1.], repeat=len(x))]
    return float(np.mean(np.asarray(vals) >= observed-1e-15))


def main():
    PA.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(FEATURES, sheet_name="Features", index_col=0,
                         parse_dates=True).sort_index()
    daily = pd.read_csv(PA/"rl_daily_gross_cost_net.csv", parse_dates=["date"])
    mechanism = pd.read_csv(PA/"mechanism_summary.csv")

    # 1. Cost sensitivity and break-even costs from fixed OOS policies.
    multipliers = [0, .25, .5, .75, 1, 1.5, 2]
    sens_rows = []
    for (strategy, seed), g in daily.groupby(["strategy", "seed"]):
        for mult in multipliers:
            pnl = g.gross_pnl.values - mult*g.cost.values
            m = compute_metrics(pnl, strategy, g.position.values)
            sens_rows.append({"strategy": strategy, "seed": seed,
                              "CostMultiplier": mult,
                              "LambdaCost": LAMBDA*mult,
                              "CumNetPnL": pnl.sum(), "Sharpe": m["Sharpe"],
                              "MaxDrawdown": m["MaxDrawdown"]})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(PA/"cost_sensitivity_seed_results.csv", index=False)
    sens_summary = sens.groupby(["strategy", "CostMultiplier", "LambdaCost"],
                                as_index=False).agg(
        N_seeds=("seed", "count"), CumNetPnL_Mean=("CumNetPnL", "mean"),
        CumNetPnL_SD=("CumNetPnL", "std"), Sharpe_Mean=("Sharpe", "mean"),
        Sharpe_SD=("Sharpe", "std"))
    sens_summary.to_csv(PA/"cost_sensitivity_summary.csv", index=False)
    break_even = mechanism[["strategy", "GrossPnL_Mean", "TransactionCost_Mean"]].copy()
    break_even["BreakEvenCostMultiplier"] = np.where(
        break_even.GrossPnL_Mean > 0,
        break_even.GrossPnL_Mean/break_even.TransactionCost_Mean, np.nan)
    break_even["BreakEvenLambda"] = LAMBDA*break_even.BreakEvenCostMultiplier
    break_even["Interpretation"] = np.where(
        break_even.GrossPnL_Mean > 0,
        "Positive gross signal; viable only below break-even cost",
        "Negative gross P&L; no non-negative cost can break even")
    break_even.to_csv(PA/"break_even_costs.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for strategy in CORE:
        g = sens_summary[sens_summary.strategy == strategy]
        ax.plot(g.CostMultiplier, g.CumNetPnL_Mean, marker="o", lw=2, label=strategy)
        ax.fill_between(g.CostMultiplier,
                        g.CumNetPnL_Mean-g.CumNetPnL_SD,
                        g.CumNetPnL_Mean+g.CumNetPnL_SD, alpha=.15)
    ax.axhline(0, color="black", lw=.8)
    ax.axvline(1, color="#777777", ls="--", lw=1, label="Main specification")
    ax.set(title="Transaction-Cost Sensitivity of Fixed OOS Policies",
           xlabel="Cost multiplier relative to main specification",
           ylabel="Five-year additive net spread P&L (mean ± SD)")
    ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(PA/"fig_cost_sensitivity.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)

    # 2. Ex-ante directional baseline: sign chosen from expanding train only.
    ts_frames, direction_rows = [], []
    for year in range(2021, 2026):
        train_mean = float(data.loc[:f"{year-2}-12-31", "SpreadReturn"].mean())
        direction = 1 if train_mean > 0 else -1 if train_mean < 0 else 0
        g = baseline_fold(data, year, direction)
        ts_frames.append(g)
        direction_rows.append({"test_year": year, "train_end": year-2,
                               "TrainingMeanSpreadReturn": train_mean,
                               "ChosenPosition": direction,
                               "ChosenDirection": {1:"Long", -1:"Short", 0:"Flat"}[direction],
                               "TestGrossPnL": g.gross_pnl.sum(),
                               "TestCost": g.cost.sum(), "TestNetPnL": g.net_pnl.sum()})
    ts = pd.concat(ts_frames, ignore_index=True)
    ts.to_csv(PA/"training_sign_baseline_daily.csv", index=False)
    directions = pd.DataFrame(direction_rows)
    directions.to_csv(PA/"training_sign_baseline_folds.csv", index=False)
    m = compute_metrics(ts.net_pnl.values, "Training-Sign Directional Baseline",
                        ts.position.values)
    pd.DataFrame([{**m, "GrossPnL": ts.gross_pnl.sum(),
                   "TransactionCost": ts.cost.sum()}]).to_csv(
        PA/"training_sign_baseline_summary.csv", index=False)

    # 3. Regime component distribution shifts.
    components = ["RegimeEGScore", "RegimeHLScore", "RegimeCVScore",
                  "RegimeStabilityScore"]
    comp_rows = []
    for col in components:
        train = data.loc[:"2019-12-31", col].dropna()
        test = data.loc["2021-01-01":"2025-12-31", col].dropna()
        pooled_sd = np.sqrt((train.var(ddof=1)+test.var(ddof=1))/2)
        ks = stats.ks_2samp(train, test)
        comp_rows.append({"component": col, "TrainMean": train.mean(),
                          "TestMean": test.mean(), "MeanDifference": test.mean()-train.mean(),
                          "StandardisedMeanDifference": (test.mean()-train.mean())/pooled_sd,
                          "TrainMedian": train.median(), "TestMedian": test.median(),
                          "KSStatistic": ks.statistic, "KSPValue": ks.pvalue,
                          "WassersteinDistance": stats.wasserstein_distance(train, test)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(PA/"regime_component_shift.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, col in zip(axes.flat, components):
        tr = np.sort(data.loc[:"2019-12-31", col].dropna().values)
        te = np.sort(data.loc["2021-01-01":"2025-12-31", col].dropna().values)
        ax.plot(tr, np.arange(1,len(tr)+1)/len(tr), label="Train ≤2019")
        ax.plot(te, np.arange(1,len(te)+1)/len(te), label="OOS 2021–2025")
        ax.set(title=col, xlabel="Score", ylabel="ECDF"); ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Distribution Shift in Regime Components")
    fig.savefig(PA/"fig_regime_component_shift.png", dpi=300,
                bbox_inches="tight"); plt.close(fig)

    # 4. Cost decomposition: action leg, hedge rebalance, forced close.
    source_rows = []
    for (strategy, seed, year), g in daily.sort_values("date").groupby(
            ["strategy", "seed", "test_year"]):
        prev = 0
        for i, row in enumerate(g.itertuples(index=False)):
            loc = data.index.get_loc(row.date)-1
            beta, beta_prev = float(data.RollingBeta.iloc[loc]), float(data.RollingBeta.iloc[loc-1])
            primary = LAMBDA*abs(row.position-prev)
            hedge = LAMBDA*abs(row.position*beta-prev*beta_prev)
            close = 0.0
            if i == len(g)-1 and row.position != 0:
                close = LAMBDA*(abs(row.position)+abs(row.position*beta))
            reconstructed = primary+hedge+close
            if abs(reconstructed-row.cost) > 1e-9:
                raise AssertionError("Cost-source reconstruction mismatch")
            source_rows.append({"strategy":strategy,"seed":seed,"test_year":year,
                                "ActionLegCost":primary,"HedgeRebalanceCost":hedge,
                                "ForcedCloseCost":close,"TotalCost":reconstructed,
                                "DirectReversal":int(prev*row.position == -1),
                                "Open":int(prev==0 and row.position!=0),
                                "Close":int(prev!=0 and row.position==0),
                                "HoldSame":int(prev==row.position)})
            prev = row.position
    sources = pd.DataFrame(source_rows)
    source_seed = sources.groupby(["strategy","seed"],as_index=False).sum(numeric_only=True)
    source_seed.to_csv(PA/"cost_source_seed_results.csv",index=False)
    source_summary = source_seed.groupby("strategy",as_index=False).agg(
        N_seeds=("seed","count"),ActionLegCost_Mean=("ActionLegCost","mean"),
        HedgeRebalanceCost_Mean=("HedgeRebalanceCost","mean"),
        ForcedCloseCost_Mean=("ForcedCloseCost","mean"),TotalCost_Mean=("TotalCost","mean"),
        DirectReversal_Mean=("DirectReversal","mean"),Open_Mean=("Open","mean"),
        Close_Mean=("Close","mean"),HoldSame_Mean=("HoldSame","mean"))
    for col in ["ActionLegCost","HedgeRebalanceCost","ForcedCloseCost"]:
        source_summary[col+"Pct"] = 100*source_summary[col+"_Mean"]/source_summary.TotalCost_Mean
    source_summary.to_csv(PA/"cost_source_summary.csv",index=False)

    s = source_summary.set_index("strategy").loc[CORE]
    fig, ax = plt.subplots(figsize=(9,5.5))
    bottom = np.zeros(len(s))
    for col,label,color in [("ActionLegCost_Mean","Discrete action changes","#4472c4"),
                            ("HedgeRebalanceCost_Mean","Hedge-ratio rebalancing","#ed7d31"),
                            ("ForcedCloseCost_Mean","Annual forced close","#a5a5a5")]:
        ax.bar(s.index,s[col],bottom=bottom,label=label,color=color)
        bottom += s[col].values
    ax.set(title="Transaction-Cost Decomposition",ylabel="Five-year cost (seed mean)")
    ax.legend(frameon=False, loc="upper center", ncol=3); fig.tight_layout()
    fig.savefig(PA/"fig_cost_decomposition.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 5. Block-length robustness for core and benchmark comparisons.
    mean_paths = daily.groupby(["strategy","date"],as_index=False).net_pnl.mean()
    baseline = pd.read_csv(PA/"baseline_daily.csv",parse_dates=["date"])
    z = baseline[baseline.strategy=="Z-score Rule"].set_index("date").pnl
    paths = {s:g.sort_values("date").set_index("date").net_pnl
             for s,g in mean_paths.groupby("strategy")}
    rng = np.random.default_rng(RNG_SEED)
    robust_rows=[]
    for block in [5,10,20,40,60]:
        for a_name,b_name,b in [(CORE[1],CORE[0],paths[CORE[0]]),(CORE[1],"Z-score Rule",z)]:
            a=paths[a_name]; common=a.index.intersection(b.index)
            diff,lo,hi,p=infer(a.loc[common],b.loc[common],block,rng)
            robust_rows.append({"strategy_A":a_name,"strategy_B":b_name,
                                "BlockLength":block,"SharpeDiff":diff,
                                "CI_L":lo,"CI_U":hi,"BlockSignFlipP":p})
    robust=pd.DataFrame(robust_rows); robust.to_csv(PA/"block_length_robustness.csv",index=False)

    # 6. Fold/seed win rates.
    annual=pd.read_csv(PA/"yearly_seed_results.csv")
    a=annual[annual.strategy==CORE[1]][["seed","test_year","Sharpe"]].rename(columns={"Sharpe":"DQN"})
    b=annual[annual.strategy==CORE[0]][["seed","test_year","Sharpe"]].rename(columns={"Sharpe":"IQN"})
    paired=a.merge(b,on=["seed","test_year"])
    zyear=annual[annual.strategy=="Z-score Rule"][["test_year","Sharpe"]].rename(columns={"Sharpe":"ZScore"})
    paired=paired.merge(zyear,on="test_year")
    paired["DQN_gt_IQN"]=paired.DQN>paired.IQN
    paired["DQN_gt_ZScore"]=paired.DQN>paired.ZScore
    paired["IQN_gt_ZScore"]=paired.IQN>paired.ZScore
    paired.to_csv(PA/"core_fold_pair_results.csv",index=False)
    wins=pd.DataFrame([
        {"comparison":"DQN > IQN","wins":paired.DQN_gt_IQN.sum(),"N":len(paired)},
        {"comparison":"DQN > Z-score","wins":paired.DQN_gt_ZScore.sum(),"N":len(paired)},
        {"comparison":"IQN > Z-score","wins":paired.IQN_gt_ZScore.sum(),"N":len(paired)},
        {"comparison":"DQN Sharpe > 0","wins":(paired.DQN>0).sum(),"N":len(paired)},
        {"comparison":"IQN Sharpe > 0","wins":(paired.IQN>0).sum(),"N":len(paired)},
    ]); wins["WinRatePct"]=100*wins.wins/wins.N
    wins.to_csv(PA/"fold_win_rates.csv",index=False)

    # 7. Paired High-vs-Low regime effects across seeds.
    reg=pd.read_csv(PA/"regime_expost_seed_results.csv")
    regime_rows=[]
    for strategy in CORE:
        g=reg[reg.strategy==strategy].pivot(index="seed",columns="regime_band",
                                            values=["MeanDailyPnL","Sharpe"])
        for metric in ["MeanDailyPnL","Sharpe"]:
            diff=g[(metric,"High")]-g[(metric,"Low")]
            tcrit=stats.t.ppf(.975,len(diff)-1)
            regime_rows.append({"strategy":strategy,"metric":metric,
                                "HighMinusLowMean":diff.mean(),"SD":diff.std(ddof=1),
                                "CI_L":diff.mean()-tcrit*diff.std(ddof=1)/np.sqrt(len(diff)),
                                "CI_U":diff.mean()+tcrit*diff.std(ddof=1)/np.sqrt(len(diff)),
                                "ExactSignFlipP":exact_sign_flip(diff),"N_seeds":len(diff)})
    regime_test=pd.DataFrame(regime_rows)
    regime_test.to_csv(PA/"regime_high_low_paired_tests.csv",index=False)

    # 8. Research framework diagram.
    fig,ax=plt.subplots(figsize=(12,4.8)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    labels=["Raw futures data","Dynamic hedge ratio\nand spread",
            "Base features +\nRegime components","92-dimensional\nrolling state",
            "Five-model\ncontrolled ablation","Expanding-window\ntraining",
            "2021–2025 OOS\nevaluation","Benchmarks, uncertainty\n& mechanism diagnostics"]
    xs=np.linspace(.07,.93,len(labels))
    for i,(x,label) in enumerate(zip(xs,labels)):
        box=FancyBboxPatch((x-.055,.38),.11,.24,boxstyle="round,pad=0.012",
                           facecolor="#d9eaf7" if i<4 else "#fce4d6",
                           edgecolor="#1f4e79",linewidth=1.2)
        ax.add_patch(box);ax.text(x,.5,label,ha="center",va="center",fontsize=8.5)
        if i<len(labels)-1:
            ax.annotate("",xy=(xs[i+1]-.06,.5),xytext=(x+.06,.5),
                        arrowprops=dict(arrowstyle="->",color="#555555",lw=1.3))
    ax.text(.5,.82,"Empirical Research Framework",ha="center",fontsize=15,weight="bold")
    ax.text(.5,.18,"Capacity control • no-look-ahead folds • multiple seeds • cost-consistent evaluation",
            ha="center",fontsize=10,color="#555555")
    fig.tight_layout();fig.savefig(PA/"fig_research_framework.png",dpi=300,bbox_inches="tight");plt.close(fig)

    # 9-10. RQ map and validity threats.
    rq=pd.DataFrame([
        ["RQ1: Does regime state information help?","Full IQN vs w/o Regime-in-State plus training-only feature correlations","No stable mean-performance gain. Training-only correlations do not support simple pairwise redundancy with base volatility or hedge-ratio features, although the Regime block has substantial internal dependence (EG-HL mean Spearman rho about 0.83)."],
        ["RQ2: Does regime reward shaping help?","Full IQN vs w/o Regime-in-Reward","Effects are seed-dependent and interact with state information."],
        ["RQ3: Does explicit regime modelling help overall?","Full IQN vs No Explicit Regime IQN","No robust return improvement; regime treatment changes variability and behaviour."],
        ["RQ4: Does IQN outperform scalar-value DQN?","Parameter-matched IQN–DQN, five seeds","DQN has better point estimates, but paired uncertainty intervals cross zero."],
        ["RQ5: Are learned policies economically viable?","Cash, Z-score, directional baselines and cost sensitivity","DQN has positive gross signal but costs erase it; IQN is negative before costs."],
        ["RQ6: Why does OOS performance deteriorate?","Year, regime, action and cost diagnostics","Temporal heterogeneity, regime covariate shift and turnover limit generalisation."],
    ],columns=["ResearchQuestion","Evidence","Conclusion"])
    rq.to_csv(PA/"research_question_evidence_map.csv",index=False)
    md_lines = ["| Research question | Evidence | Conclusion |",
                "|---|---|---|"]
    for row in rq.itertuples(index=False):
        cells = [str(x).replace("|", "\\|") for x in row]
        md_lines.append("| " + " | ".join(cells) + " |")
    (PA/"research_question_evidence_map.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8")
    threats="""# Threats to Validity

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
"""
    (PA/"threats_to_validity_draft.md").write_text(threats,encoding="utf-8")

    dqn_be = break_even.set_index("strategy").loc[CORE[1]]
    iqn_be = break_even.set_index("strategy").loc[CORE[0]]
    cs = source_summary.set_index("strategy")
    dqn_cs, iqn_cs = cs.loc[CORE[1]], cs.loc[CORE[0]]
    comp_rank = comp.sort_values("StandardisedMeanDifference")
    wr = wins.set_index("comparison")
    dqn_reg = regime_test[(regime_test.strategy==CORE[1]) &
                          (regime_test.metric=="Sharpe")].iloc[0]
    iqn_reg = regime_test[(regime_test.strategy==CORE[0]) &
                          (regime_test.metric=="Sharpe")].iloc[0]
    deep = f"""# Extended Robustness and Mechanism Results

## Transaction-cost sensitivity and break-even analysis

The fixed-policy cost sensitivity analysis confirms that Full IQN cannot break even at any non-negative transaction-cost level because its mean gross P&L is already negative ({iqn_be.GrossPnL_Mean:.3f}). Double DQN has positive mean gross P&L, but its break-even cost multiplier is only {dqn_be.BreakEvenCostMultiplier:.3f} relative to the main specification. Equivalently, its break-even lambda is approximately {dqn_be.BreakEvenLambda:.6f}, compared with the maintained value of {LAMBDA:.6f}. DQN would therefore require a reduction of approximately {(1-dqn_be.BreakEvenCostMultiplier)*100:.1f}% in the assumed cost rate to reach zero mean net P&L, holding the learned policy fixed. This result distinguishes statistical signal extraction from economic implementability: DQN contains some gross predictive information, but not enough to survive the main trading-cost assumption.

## Ex-ante directional benchmark

The Training-Sign Directional Baseline selects the next test year's position using only the sign of mean spread returns observed through the end of the corresponding training window. The historical mean was slightly negative in every fold, so the rule selected a short position for all five test years. It consequently reproduced the always-short diagnostic result, with cumulative net P&L of {m['CumReturn']:.3f} and Sharpe ratio of {m['Sharpe']:.3f}. Unlike an ex-post choice of direction, this baseline is implementable without test-period information. However, the training means were economically very small, so the directional rule should be interpreted as a deliberately simple benchmark rather than a robust forecasting model.

## Regime-component distribution shift

All regime components shifted downward out of sample, but the largest standardised change occurred in {comp_rank.iloc[0].component} (standardised mean difference {comp_rank.iloc[0].StandardisedMeanDifference:.3f}), followed by {comp_rank.iloc[1].component} ({comp_rank.iloc[1].StandardisedMeanDifference:.3f}). The corresponding shifts for the remaining components were {comp_rank.iloc[2].component} ({comp_rank.iloc[2].StandardisedMeanDifference:.3f}) and {comp_rank.iloc[3].component} ({comp_rank.iloc[3].StandardisedMeanDifference:.3f}). Thus, the collapse in the composite stability measure was driven most strongly by deterioration in hedge-ratio stability, with additional weakening in the half-life and cointegration components. This identifies the economic source of the covariate shift rather than treating the aggregate score as a black box.

## Sources of transaction costs

For Full IQN, discrete action changes accounted for {iqn_cs.ActionLegCostPct:.1f}% of total costs and hedge-ratio rebalancing for {iqn_cs.HedgeRebalanceCostPct:.1f}%; forced annual closure contributed the small remainder. For Double DQN, the corresponding shares were {dqn_cs.ActionLegCostPct:.1f}% and {dqn_cs.HedgeRebalanceCostPct:.1f}%. The dominant source was therefore frequent changes in the discrete trading decision, although continuous adjustment of the hedge leg contributed more than one-third of costs. Reducing turnover would need to address both action persistence and dynamic hedge rebalancing.

## Inference robustness and fold-level consistency

The conclusion that DQN has better point estimates but no statistically established superiority is unchanged for circular-block lengths of 5, 10, 20, 40 and 60 trading days. Across all specifications, the confidence interval for the DQN-minus-IQN Sharpe difference crosses zero; the same is true for DQN relative to the Z-score rule. At the seed-by-year level, DQN exceeded IQN in {int(wr.loc['DQN > IQN','wins'])} of {int(wr.loc['DQN > IQN','N'])} comparisons ({wr.loc['DQN > IQN','WinRatePct']:.1f}%). It exceeded the annual Z-score Sharpe in {int(wr.loc['DQN > Z-score','wins'])} of {int(wr.loc['DQN > Z-score','N'])} cases ({wr.loc['DQN > Z-score','WinRatePct']:.1f}%). Positive Sharpe ratios occurred in only {int(wr.loc['DQN Sharpe > 0','wins'])} DQN cases and {int(wr.loc['IQN Sharpe > 0','wins'])} IQN cases. These win rates reinforce that no model has consistent dominance.

## Conditional regime effects

Using the balanced, ex-post descriptive stability tertiles within each test year, DQN's mean High-minus-Low Sharpe difference was {dqn_reg.HighMinusLowMean:.3f}, with a 95% interval from {dqn_reg.CI_L:.3f} to {dqn_reg.CI_U:.3f} and exact sign-flip p-value {dqn_reg.ExactSignFlipP:.3f}. For Full IQN, the corresponding difference was {iqn_reg.HighMinusLowMean:.3f}, with interval [{iqn_reg.CI_L:.3f}, {iqn_reg.CI_U:.3f}] and p-value {iqn_reg.ExactSignFlipP:.3f}. Neither model therefore shows statistically reliable High-versus-Low regime improvement. Because these groups are defined ex post for descriptive balance, the comparison is evidence of conditional heterogeneity, not a pre-specified real-time trading rule or a monotonic causal effect.

## Overall implication

The additional analyses sharpen the negative result into three distinct mechanisms. First, Full IQN fails at the gross-signal stage, whereas DQN learns a positive but economically small gross signal. Second, discrete action turnover and hedge-leg rebalancing jointly remove the DQN gross advantage. Third, a large out-of-sample shift—especially in hedge-ratio stability—weakens the interpretation and generalisability of the regime inputs. These findings explain why greater representational complexity did not translate into robust economic performance.
"""
    (PA/"deepening_results_discussion_draft.md").write_text(deep,encoding="utf-8")
    print("Deepening analyses complete")


if __name__=="__main__":
    main()
