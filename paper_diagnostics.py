"""Mechanism diagnostics and thesis-ready narrative for the v6 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commodity_utils import spread_trade_cost


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
PA = WF / "paper_analysis"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"
CORE = ["Full IQN", "Parameter-Matched Double DQN"]
COLORS = {"Full IQN": "#1f4e79", "Parameter-Matched Double DQN": "#c55a11",
          "No Explicit Regime IQN": "#8064a2", "w/o Regime-in-State": "#548235",
          "w/o Regime-in-Reward": "#bf9000", "Z-score Rule": "#333333"}


def load_rl(data):
    rows, max_error = [], 0.0
    for path in sorted((WF / "task_results").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        dates = pd.to_datetime(item["daily"]["dates"])
        pnls = np.asarray(item["daily"]["pnls"], float)
        positions = np.asarray(item["daily"]["positions"], int)
        prev = 0
        for i, (date, net, pos) in enumerate(zip(dates, pnls, positions)):
            loc = data.index.get_loc(date)
            signal_loc = loc - 1
            gross = pos * float(np.nan_to_num(data["SpreadReturn"].iloc[loc]))
            beta_t = float(data["RollingBeta"].iloc[signal_loc])
            beta_prev = float(data["RollingBeta"].iloc[signal_loc - 1])
            formula_cost = spread_trade_cost(0.0005, pos, prev, beta_t, beta_prev)
            if i == len(dates) - 1 and pos != 0:
                formula_cost += spread_trade_cost(0.0005, 0, pos, beta_t, beta_t)
            implied_cost = gross - net
            max_error = max(max_error, abs(formula_cost - implied_cost))
            rows.append({"strategy": item["variant"], "seed": item["seed"],
                         "test_year": item["test_year"], "date": date,
                         "position": pos, "gross_pnl": gross,
                         "cost": implied_cost, "net_pnl": net,
                         "position_changed": int(pos != prev),
                         "stability_score": float(data["RegimeStabilityScore"].iloc[signal_loc])})
            prev = pos
    return pd.DataFrame(rows), max_error


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def main():
    PA.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(FEATURES, sheet_name="Features", index_col=0,
                         parse_dates=True).sort_index()
    rl, cost_error = load_rl(data)
    if cost_error > 1e-9:
        raise AssertionError(f"Cost reconstruction error: {cost_error}")
    rl.to_csv(PA / "rl_daily_gross_cost_net.csv", index=False)

    # Gross-to-net and action diagnostics at seed level and model summary.
    seed_rows = []
    for (strategy, seed), g in rl.groupby(["strategy", "seed"]):
        seed_rows.append({
            "strategy": strategy, "seed": seed, "N_days": len(g),
            "GrossPnL": g.gross_pnl.sum(), "TransactionCost": g.cost.sum(),
            "NetPnL": g.net_pnl.sum(),
            "PctShort": 100 * (g.position == -1).mean(),
            "PctFlat": 100 * (g.position == 0).mean(),
            "PctLong": 100 * (g.position == 1).mean(),
            "NPositionChanges": int(g.position_changed.sum()),
        })
    seed = pd.DataFrame(seed_rows)
    seed.to_csv(PA / "mechanism_seed_results.csv", index=False)
    summary = seed.groupby("strategy", as_index=False).agg(
        N_seeds=("seed", "count"), GrossPnL_Mean=("GrossPnL", "mean"),
        GrossPnL_SD=("GrossPnL", "std"), TransactionCost_Mean=("TransactionCost", "mean"),
        TransactionCost_SD=("TransactionCost", "std"), NetPnL_Mean=("NetPnL", "mean"),
        NetPnL_SD=("NetPnL", "std"), PctShort_Mean=("PctShort", "mean"),
        PctFlat_Mean=("PctFlat", "mean"), PctLong_Mean=("PctLong", "mean"),
        NPositionChanges_Mean=("NPositionChanges", "mean"))
    summary["CostAsPctGrossMagnitude"] = (
        summary.TransactionCost_Mean / summary.GrossPnL_Mean.abs() * 100)
    summary.to_csv(PA / "mechanism_summary.csv", index=False)

    # Fold-specific anchored regime distribution shift.
    shift_rows = []
    for year in range(2021, 2026):
        train = data.loc[:f"{year-2}-12-31", "RegimeStabilityScore"].dropna()
        test_signal = data.loc[f"{year}-01-01":f"{year}-12-31",
                               "RegimeStabilityScore"].iloc[:-1].dropna()
        lo, hi = np.percentile(train, [33, 67])
        counts = pd.cut(test_signal, [-np.inf, lo, hi, np.inf],
                        labels=["Low", "Mid", "High"]).value_counts()
        shift_rows.append({"test_year": year, "train_end": year-2,
                           "TrainMean": train.mean(), "TrainMedian": train.median(),
                           "TestMean": test_signal.mean(), "TestMedian": test_signal.median(),
                           "LowPct": 100*counts.get("Low", 0)/len(test_signal),
                           "MidPct": 100*counts.get("Mid", 0)/len(test_signal),
                           "HighPct": 100*counts.get("High", 0)/len(test_signal),
                           "LowCut": lo, "HighCut": hi})
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(PA / "regime_distribution_shift.csv", index=False)

    # Position behaviour conditional on balanced within-year stability bands.
    band_frames = []
    for year, g in rl.groupby("test_year"):
        dates = g[["date", "stability_score"]].drop_duplicates("date").copy()
        dates["regime_band"] = pd.qcut(dates.stability_score.rank(method="first"), 3,
                                        labels=["Low", "Mid", "High"])
        band_frames.append(g.merge(dates[["date", "regime_band"]], on="date"))
    banded = pd.concat(band_frames, ignore_index=True)
    behavior = banded.groupby(["strategy", "seed", "regime_band"], observed=True).agg(
        N_days=("date", "count"), PctShort=("position", lambda x: 100*(x == -1).mean()),
        PctFlat=("position", lambda x: 100*(x == 0).mean()),
        PctLong=("position", lambda x: 100*(x == 1).mean()),
        MeanNetPnL=("net_pnl", "mean")).reset_index()
    behavior.to_csv(PA / "regime_action_seed_results.csv", index=False)
    behavior_summary = behavior.groupby(["strategy", "regime_band"], as_index=False,
                                         observed=True).agg(
        N_seeds=("seed", "count"), PctShort_Mean=("PctShort", "mean"),
        PctFlat_Mean=("PctFlat", "mean"), PctLong_Mean=("PctLong", "mean"),
        MeanNetPnL_Mean=("MeanNetPnL", "mean"))
    behavior_summary.to_csv(PA / "regime_action_summary.csv", index=False)

    # Plot settings.
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25})

    # Distribution-shift ECDF: initial training window vs full OOS signals.
    train = data.loc[:"2019-12-31", "RegimeStabilityScore"].dropna().sort_values()
    test = data.loc["2021-01-01":"2025-12-31", "RegimeStabilityScore"].iloc[:-1].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(train, np.arange(1, len(train)+1)/len(train), label="Training through 2019", lw=2)
    ax.plot(test, np.arange(1, len(test)+1)/len(test), label="OOS 2021–2025", lw=2)
    ax.set(title="Distribution Shift in Regime Stability Score",
           xlabel="Regime Stability Score", ylabel="Empirical cumulative probability")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(PA / "fig_regime_distribution_shift.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)

    # Gross-cost-net attribution.
    s = summary.set_index("strategy").loc[[x for x in COLORS if x in summary.strategy.values]]
    x = np.arange(len(s)); width = .25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x-width, s.GrossPnL_Mean, width, label="Gross P&L", color="#5b9bd5")
    ax.bar(x, -s.TransactionCost_Mean, width, label="Transaction costs", color="#a5a5a5")
    ax.bar(x+width, s.NetPnL_Mean, width, label="Net P&L", color="#ed7d31")
    ax.axhline(0, color="black", lw=.7); ax.set_xticks(x, s.index, rotation=25, ha="right")
    ax.set(title="Gross-to-Net P&L Attribution by RL Model",
           ylabel="Five-year additive spread P&L (seed mean)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(PA / "fig_gross_cost_net.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)

    # Core-model action proportions.
    a = summary.set_index("strategy").loc[CORE]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(a.index, a.PctShort_Mean, label="Short", color="#c00000")
    ax.bar(a.index, a.PctFlat_Mean, bottom=a.PctShort_Mean,
           label="Flat", color="#a5a5a5")
    ax.bar(a.index, a.PctLong_Mean, bottom=a.PctShort_Mean+a.PctFlat_Mean,
           label="Long", color="#4472c4")
    ax.set(title="Out-of-Sample Position Allocation", ylabel="Share of trading days (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(); fig.savefig(PA / "fig_action_proportions.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)

    # Seed x year heatmaps for core models.
    annual = pd.read_csv(PA / "yearly_seed_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                             constrained_layout=True)
    all_core = annual[annual.strategy.isin(CORE)]
    vmax = np.nanmax(np.abs(all_core.Sharpe)); vmax = max(1.0, vmax)
    for ax, strategy in zip(axes, CORE):
        h = annual[annual.strategy == strategy].pivot(index="seed", columns="test_year", values="Sharpe")
        im = ax.imshow(h.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(h.columns)), [int(x) for x in h.columns])
        ax.set_yticks(range(len(h.index)), [int(x) for x in h.index])
        ax.set(title=strategy, xlabel="Test year", ylabel="Random seed")
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                ax.text(j, i, f"{h.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(h.iloc[i,j]) > .55*vmax else "black")
    fig.colorbar(im, ax=axes, shrink=.82, pad=.025, label="Annualised Sharpe ratio")
    fig.suptitle("Seed × Test-Year Out-of-Sample Performance")
    fig.savefig(PA / "fig_seed_year_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Thesis-ready narrative, generated from audited tables.
    sm = summary.set_index("strategy")
    dqn = sm.loc[CORE[1]]; iqn = sm.loc[CORE[0]]
    b = behavior_summary.set_index(["strategy", "regime_band"])
    narrative = f"""# Supplementary Results and Discussion Draft

## Transaction-cost attribution

Decomposing out-of-sample performance into gross spread P&L and transaction costs reveals different failure mechanisms across the two core models. Full IQN generated a mean gross P&L of {iqn.GrossPnL_Mean:.3f} across five seeds and incurred mean costs of {iqn.TransactionCost_Mean:.3f}, resulting in net P&L of {iqn.NetPnL_Mean:.3f}. Its performance was therefore already negative before costs. By contrast, the parameter-matched Double DQN generated positive mean gross P&L of {dqn.GrossPnL_Mean:.3f}, but incurred costs of {dqn.TransactionCost_Mean:.3f}, producing net P&L of {dqn.NetPnL_Mean:.3f}. DQN therefore extracted some directional signal, but its economic magnitude was insufficient to cover turnover. IQN exhibited both weak gross signal extraction and substantial cost exposure.

## Policy behaviour

Full IQN allocated, on average, {iqn.PctShort_Mean:.1f}% of test days to short positions, {iqn.PctFlat_Mean:.1f}% to cash, and {iqn.PctLong_Mean:.1f}% to long positions. Double DQN allocated {dqn.PctShort_Mean:.1f}%, {dqn.PctFlat_Mean:.1f}%, and {dqn.PctLong_Mean:.1f}% respectively. Because the realised spread exhibited a favourable short-direction drift over 2021–2025, failure to maintain sufficient short exposure helps explain why both learned policies underperformed the diagnostic always-short benchmark. This does not establish that the short direction was predictable ex ante; rather, it identifies a source of ex-post opportunity cost.

## Regime distribution shift

The stability-score distribution shifted materially between the historical training sample and the 2021–2025 out-of-sample period. Applying fold-specific 33rd and 67th percentile thresholds estimated only from the expanding training window classified approximately {shift.LowPct.mean():.1f}% of out-of-sample observations as Low stability, {shift.MidPct.mean():.1f}% as Mid stability, and {shift.HighPct.mean():.1f}% as High stability. No High-stability observations occurred under these training-anchored thresholds. Consequently, the regime-aware reward was evaluated predominantly in a part of the state space associated with persistent instability. This provides direct evidence of covariate shift and limits the extent to which the training-period regime mapping could generalise.

## Seed and year heterogeneity

The seed-by-year results show that performance differences were not driven by a single random seed, but neither model was consistently superior across calendar periods. Double DQN produced positive mean annual Sharpe ratios in 2022 and 2025, whereas Full IQN was positive only in 2025. Both models performed poorly in 2024, and Full IQN experienced particularly weak outcomes in 2023. This temporal reversal is consistent with a non-stationary learning problem in which relationships learned from an expanding historical window do not remain stable in the subsequent year.

## Interpretation

Taken together, the diagnostics show that the negative net results do not have a single cause. DQN learned a positive gross signal, but frequent reallocation consumed more than the signal generated; IQN was negative even before costs. Both policies varied materially across years and were evaluated under a substantial shift in the regime distribution. The richer return-distribution representation of IQN therefore did not overcome the more fundamental generalisation problem. Under the present risk-neutral setting, model complexity was not a substitute for a stable and economically large predictive signal.

## Limitations and reporting caution

The always-short strategy is a diagnostic benchmark, not evidence of an ex-ante implementable forecasting rule. The within-test-year stability tertiles are likewise descriptive and were not used in training or model selection. Return and drawdown statistics are expressed in additive spread-return units because no portfolio capital or margin base is defined. The transaction-cost specification is a consistent two-leg turnover proxy rather than a complete futures execution model. These restrictions should be stated explicitly when interpreting economic significance.
"""
    (PA / "thesis_results_discussion_draft.md").write_text(narrative, encoding="utf-8")
    print(f"Diagnostics written to {PA}; max cost reconstruction error={cost_error:.3e}")


if __name__ == "__main__":
    main()
