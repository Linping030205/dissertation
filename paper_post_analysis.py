"""Paper-ready benchmark, uncertainty, yearly, and regime analysis.

Uses the completed v6 walk-forward daily paths.  Deterministic baselines are
recomputed independently for every test fold, using only that fold's expanding
training window to estimate the Z-score entry threshold.  All strategies use
the same additive spread-P&L and turnover-cost convention as the RL environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from commodity_utils import spread_trade_cost
from evaluate import compute_metrics


ROOT = Path(__file__).resolve().parent
WF_ROOT = ROOT / "analysis_output" / "walkforward_v6"
OUT = WF_ROOT / "paper_analysis"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"
YEARS = range(2021, 2026)
LAMBDA_COST = 0.0005
ENTRY_PERCENTILE = 95.0
EXIT_Z = 0.5
BLOCK_LENGTH = 20
N_BOOT = 5000
RNG_SEED = 260821


def metrics(pnl, label="", positions=None):
    result = compute_metrics(np.asarray(pnl, float), label, positions)
    # Undefined ratios (e.g. cash has zero volatility/drawdown) should be
    # blank in publication tables, not displayed as misleading infinities.
    for key, value in list(result.items()):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            result[key] = None
    return result


def run_baseline_fold(data: pd.DataFrame, year: int, kind: str) -> pd.DataFrame:
    """Run one no-look-ahead baseline with the same annual reset/close as RL."""
    test = data.loc[f"{year}-01-01":f"{year}-12-31"]
    if len(test) < 2:
        raise ValueError(f"Insufficient observations for {year}")
    train = data.loc[:f"{year - 2}-12-31"]
    entry = float(np.nanpercentile(np.abs(train["ZScore20"]), ENTRY_PERCENTILE))
    pos = prev_pos = 0
    rows = []
    for i in range(len(test) - 1):
        z = float(test["ZScore20"].iloc[i])
        if kind == "Z-score Rule":
            if pos == 0 and np.isfinite(z):
                if z > entry:
                    pos = -1
                elif z < -entry:
                    pos = 1
            elif pos != 0 and np.isfinite(z) and abs(z) < EXIT_Z:
                pos = 0
        elif kind == "Always Long Spread":
            pos = 1
        elif kind == "Always Short Spread":
            pos = -1
        elif kind == "Cash / No Trade":
            pos = 0
        else:
            raise ValueError(kind)

        beta_t = float(test["RollingBeta"].iloc[i])
        beta_prev = float(test["RollingBeta"].iloc[i - 1]) if i else beta_t
        gross = pos * float(np.nan_to_num(test["SpreadReturn"].iloc[i + 1]))
        cost = spread_trade_cost(LAMBDA_COST, pos, prev_pos, beta_t, beta_prev)
        if i == len(test) - 2 and pos != 0:
            cost += spread_trade_cost(LAMBDA_COST, 0, pos, beta_t, beta_t)
        rows.append({
            "strategy": kind, "test_year": year,
            "signal_date": test.index[i], "date": test.index[i + 1],
            "pnl": gross - cost, "gross_pnl": gross, "cost": cost,
            "position": pos, "entry_threshold": entry,
            "stability_score": float(test["RegimeStabilityScore"].iloc[i]),
        })
        prev_pos = pos
    return pd.DataFrame(rows)


def load_rl_daily() -> pd.DataFrame:
    rows = []
    for path in sorted((WF_ROOT / "task_results").glob("*.json")):
        with path.open(encoding="utf-8") as f:
            item = json.load(f)
        daily = item["daily"]
        for date, pnl, pos in zip(daily["dates"], daily["pnls"], daily["positions"]):
            rows.append({"strategy": item["variant"], "seed": int(item["seed"]),
                         "test_year": int(item["test_year"]),
                         "date": pd.Timestamp(date), "pnl": float(pnl),
                         "position": int(pos)})
    return pd.DataFrame(rows)


def circular_block_indices(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(block) + s) % n for s in starts])[:n]


def sharpe(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return 0.0 if sd < 1e-14 else float(x.mean() / sd * np.sqrt(252))


def paired_block_inference(a, b, rng):
    """Paired circular-block bootstrap CI plus block sign-flip mean test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("Paired paths differ in length")
    sh_diff = np.empty(N_BOOT)
    for j in range(N_BOOT):
        ix = circular_block_indices(len(a), BLOCK_LENGTH, rng)
        sh_diff[j] = sharpe(a[ix]) - sharpe(b[ix])
    ci_lo, ci_hi = np.percentile(sh_diff, [2.5, 97.5])

    d = a - b
    blocks = np.array_split(d, np.arange(BLOCK_LENGTH, len(d), BLOCK_LENGTH))
    block_sums = np.array([x.sum() for x in blocks])
    observed = abs(block_sums.sum())
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, len(block_sums)))
    permuted = np.abs(signs @ block_sums)
    p = float((1 + np.sum(permuted >= observed)) / (N_BOOT + 1))
    return {
        "MeanDailyPnLDiff": float(d.mean()),
        "SharpeDiff": sharpe(a) - sharpe(b),
        "SharpeDiff_CI_L": float(ci_lo), "SharpeDiff_CI_U": float(ci_hi),
        "BlockSignFlip_p": p,
    }


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, ix in enumerate(order):
        running = max(running, (m - rank) * p[ix])
        adjusted[ix] = min(1.0, running)
    return adjusted


def seed_summary(rl: pd.DataFrame):
    seed_rows = []
    for (strategy, seed), g in rl.sort_values("date").groupby(["strategy", "seed"]):
        seed_rows.append({"strategy": strategy, "seed": seed,
                          **metrics(g.pnl, strategy, g.position)})
    seed_df = pd.DataFrame(seed_rows)
    out = []
    for strategy, g in seed_df.groupby("strategy"):
        n = len(g)
        tcrit = stats.t.ppf(.975, n - 1) if n > 1 else np.nan
        row = {"strategy": strategy, "N_seeds": n}
        for col in ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "Sortino",
                    "MaxDrawdown", "Calmar", "PctPositive", "NPositionChanges"]:
            mean, sd = g[col].mean(), g[col].std(ddof=1)
            row[f"{col}_Mean"] = mean
            row[f"{col}_SD"] = sd
            row[f"{col}_SeedCI_L"] = mean - tcrit * sd / np.sqrt(n)
            row[f"{col}_SeedCI_U"] = mean + tcrit * sd / np.sqrt(n)
        out.append(row)
    return seed_df, pd.DataFrame(out)


def assign_regimes(panel, data):
    thresholds = []
    frames = []
    for year in YEARS:
        train_scores = data.loc[:f"{year-2}-12-31", "RegimeStabilityScore"].dropna()
        lo, hi = np.percentile(train_scores, [33, 67])
        thresholds.append({"test_year": year, "train_end": year - 2,
                           "low_cut": lo, "high_cut": hi})
        g = panel[panel.test_year == year].copy()
        # The signal observed on t drives P&L dated t+1.
        signal_scores = data["RegimeStabilityScore"].shift(1)
        g["stability_score"] = g.date.map(signal_scores)
        g["regime_band"] = pd.cut(g.stability_score,
                                   [-np.inf, lo, hi, np.inf],
                                   labels=["Low", "Mid", "High"])
        frames.append(g)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(thresholds)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(FEATURES, sheet_name="Features", index_col=0,
                         parse_dates=True).sort_index()
    rl = load_rl_daily()

    baseline_daily = pd.concat([
        run_baseline_fold(data, year, kind)
        for year in YEARS
        for kind in ["Z-score Rule", "Always Long Spread",
                     "Always Short Spread", "Cash / No Trade"]
    ], ignore_index=True)
    baseline_daily.to_csv(OUT / "baseline_daily.csv", index=False)

    base_rows = []
    for strategy, g in baseline_daily.sort_values("date").groupby("strategy"):
        base_rows.append({"strategy": strategy, **metrics(g.pnl, strategy, g.position),
                          "TotalCost": g.cost.sum()})
    base_summary = pd.DataFrame(base_rows)
    base_summary.to_csv(OUT / "benchmark_summary.csv", index=False)

    seed_df, uncertainty = seed_summary(rl)
    seed_df.to_csv(OUT / "model_seed_results.csv", index=False)
    uncertainty.to_csv(OUT / "model_summary_uncertainty.csv", index=False)

    # Compare the across-seed average policy path with every deterministic rule.
    mean_paths = rl.groupby(["strategy", "date"], as_index=False).agg(
        pnl=("pnl", "mean"), test_year=("test_year", "first"))
    rng = np.random.default_rng(RNG_SEED)
    comparisons = []
    for benchmark, b in baseline_daily.groupby("strategy"):
        b = b.sort_values("date").set_index("date")
        family_start = len(comparisons)
        for strategy, g in mean_paths.groupby("strategy"):
            g = g.sort_values("date").set_index("date")
            common = g.index.intersection(b.index)
            result = paired_block_inference(g.loc[common, "pnl"], b.loc[common, "pnl"], rng)
            comparisons.append({"strategy": strategy, "benchmark": benchmark,
                                "N_days": len(common), **result})
        family_end = len(comparisons)
        family_p = [x["BlockSignFlip_p"] for x in comparisons[family_start:family_end]]
        for row, adjusted in zip(comparisons[family_start:family_end], holm_adjust(family_p)):
            row["HolmAdjusted_p"] = adjusted
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUT / "baseline_comparisons.csv", index=False)

    core = {}
    for strategy in ["Full IQN", "Parameter-Matched Double DQN"]:
        core[strategy] = mean_paths[mean_paths.strategy == strategy].sort_values("date").set_index("date")
    common = core["Full IQN"].index.intersection(core["Parameter-Matched Double DQN"].index)
    core_test = paired_block_inference(
        core["Parameter-Matched Double DQN"].loc[common, "pnl"],
        core["Full IQN"].loc[common, "pnl"], rng)
    pd.DataFrame([{"strategy_A": "Parameter-Matched Double DQN",
                   "strategy_B": "Full IQN", "N_days": len(common),
                   **core_test}]).to_csv(OUT / "core_comparison.csv", index=False)

    # Per-year metrics: seed-level observations for RL, one observation/baseline.
    annual_rows = []
    for (strategy, seed, year), g in rl.groupby(["strategy", "seed", "test_year"]):
        annual_rows.append({"strategy": strategy, "seed": seed, "test_year": year,
                            "type": "RL", **metrics(g.pnl, strategy, g.position)})
    for (strategy, year), g in baseline_daily.groupby(["strategy", "test_year"]):
        annual_rows.append({"strategy": strategy, "seed": np.nan, "test_year": year,
                            "type": "Baseline", **metrics(g.pnl, strategy, g.position)})
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(OUT / "yearly_seed_results.csv", index=False)
    yearly_summary = annual.groupby(["strategy", "type", "test_year"], as_index=False).agg(
        N=("Sharpe", "count"), Sharpe_Mean=("Sharpe", "mean"), Sharpe_SD=("Sharpe", "std"),
        CumReturn_Mean=("CumReturn", "mean"), CumReturn_SD=("CumReturn", "std"),
        MaxDrawdown_Mean=("MaxDrawdown", "mean"))
    yearly_summary.to_csv(OUT / "yearly_summary.csv", index=False)

    # Regime analysis on training-anchored bands; add baselines as seed=0 paths.
    base_panel = baseline_daily[["strategy", "test_year", "date", "pnl", "position"]].copy()
    base_panel["seed"] = 0
    panel = pd.concat([rl, base_panel], ignore_index=True, sort=False)
    panel, thresholds = assign_regimes(panel, data)
    thresholds.to_csv(OUT / "regime_thresholds.csv", index=False)
    regime_rows = []
    for (strategy, seed, band), g in panel.groupby(["strategy", "seed", "regime_band"], observed=True):
        m = metrics(g.pnl, strategy, path_positions := g.position)
        regime_rows.append({"strategy": strategy, "seed": seed, "regime_band": str(band),
                            "N_days": len(g), "MeanDailyPnL": g.pnl.mean(),
                            "AnnReturn": m["AnnReturn"], "AnnVol": m["AnnVol"],
                            "Sharpe": m["Sharpe"], "PctPositive": m["PctPositive"]})
    regime_seed = pd.DataFrame(regime_rows)
    regime_seed.to_csv(OUT / "regime_seed_results.csv", index=False)
    regime_summary = regime_seed.groupby(["strategy", "regime_band"], as_index=False).agg(
        N_series=("seed", "count"), N_days=("N_days", "mean"),
        MeanDailyPnL_Mean=("MeanDailyPnL", "mean"), MeanDailyPnL_SD=("MeanDailyPnL", "std"),
        Sharpe_Mean=("Sharpe", "mean"), Sharpe_SD=("Sharpe", "std"),
        PctPositive_Mean=("PctPositive", "mean"))
    regime_summary.to_csv(OUT / "regime_summary.csv", index=False)

    # Balanced, ex-post descriptive bands within each test year.  These do
    # not enter training/model selection; they exist only to make conditional
    # performance comparisons estimable when the anchored High band is empty.
    expost_frames = []
    for year, g in panel.groupby("test_year"):
        g = g.copy()
        unique_dates = g[["date", "stability_score"]].drop_duplicates("date")
        ranked = unique_dates.stability_score.rank(method="first")
        unique_dates["regime_band_expost"] = pd.qcut(
            ranked, 3, labels=["Low", "Mid", "High"])
        g = g.merge(unique_dates[["date", "regime_band_expost"]], on="date", how="left")
        expost_frames.append(g)
    expost_panel = pd.concat(expost_frames, ignore_index=True)
    expost_rows = []
    for (strategy, seed, band), g in expost_panel.groupby(
            ["strategy", "seed", "regime_band_expost"], observed=True):
        m = metrics(g.pnl, strategy, g.position)
        expost_rows.append({"strategy": strategy, "seed": seed,
                            "regime_band": str(band), "N_days": len(g),
                            "MeanDailyPnL": g.pnl.mean(), "AnnReturn": m["AnnReturn"],
                            "AnnVol": m["AnnVol"], "Sharpe": m["Sharpe"],
                            "PctPositive": m["PctPositive"]})
    expost_seed = pd.DataFrame(expost_rows)
    expost_seed.to_csv(OUT / "regime_expost_seed_results.csv", index=False)
    expost_summary = expost_seed.groupby(["strategy", "regime_band"], as_index=False).agg(
        N_series=("seed", "count"), N_days=("N_days", "mean"),
        MeanDailyPnL_Mean=("MeanDailyPnL", "mean"), MeanDailyPnL_SD=("MeanDailyPnL", "std"),
        Sharpe_Mean=("Sharpe", "mean"), Sharpe_SD=("Sharpe", "std"),
        PctPositive_Mean=("PctPositive", "mean"))
    expost_summary.to_csv(OUT / "regime_expost_summary.csv", index=False)

    # Daily panel supports reproducibility and equity-curve plots.
    panel.to_csv(OUT / "daily_analysis_panel.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25})
    fig, ax = plt.subplots(figsize=(11, 6))
    selected = ["Full IQN", "Parameter-Matched Double DQN"]
    directional_label = "Always Short / Training-Sign Directional"
    colors = {"Full IQN": "#1f4e79", "Parameter-Matched Double DQN": "#c55a11",
              "Z-score Rule": "#333333", "Cash / No Trade": "#777777",
              directional_label: "#2e7d32"}
    for strategy in selected:
        g = mean_paths[mean_paths.strategy == strategy].sort_values("date")
        ax.plot(g.date, g.pnl.cumsum(), label=f"{strategy} (seed mean)",
                lw=1.8, color=colors[strategy])
    for strategy in ["Z-score Rule", "Cash / No Trade"]:
        g = baseline_daily[baseline_daily.strategy == strategy].sort_values("date")
        ax.plot(g.date, g.pnl.cumsum(), label=strategy, lw=1.5,
                color=colors[strategy], linestyle="--" if strategy == "Cash / No Trade" else "-")
    # The ex-ante Training-Sign rule selects short in every formal fold and is
    # therefore path-identical to Always Short.  Plot one line with a combined
    # label rather than drawing the same series twice.
    g = baseline_daily[baseline_daily.strategy == "Always Short Spread"].sort_values("date")
    ax.plot(g.date, g.pnl.cumsum(), label=directional_label, lw=1.7,
            color=colors[directional_label])
    ax.axhline(0, color="black", lw=.7)
    ax.set(title="Out-of-Sample Walk-Forward Cumulative Net Spread P&L",
           xlabel="Return date", ylabel="Cumulative additive spread P&L")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_equity_vs_baselines.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_equity_vs_baselines_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    yearly_selected = selected + ["Z-score Rule", "Always Short Spread"]
    plot = yearly_summary[yearly_summary.strategy.isin(yearly_selected)]
    pivot = plot.pivot(index="test_year", columns="strategy", values="Sharpe_Mean")
    pivot = pivot.rename(columns={"Always Short Spread": directional_label})
    plotted = selected + ["Z-score Rule", directional_label]
    pivot[plotted].plot(kind="bar", ax=ax, color=[colors[x] for x in plotted])
    ax.axhline(0, color="black", lw=.7)
    ax.set(title="Out-of-Sample Sharpe Ratio by Test Year",
           xlabel="Test year", ylabel="Annualised Sharpe ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_yearly_sharpe.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_yearly_sharpe_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot = expost_summary[expost_summary.strategy.isin(selected + ["Z-score Rule"])]
    pivot = plot.pivot(index="regime_band", columns="strategy", values="Sharpe_Mean").reindex(["Low", "Mid", "High"])
    pivot[selected + ["Z-score Rule"]].plot(kind="bar", ax=ax,
        color=[colors[x] for x in selected + ["Z-score Rule"]])
    ax.axhline(0, color="black", lw=.7)
    ax.set(title="Performance by Within-Year Relative Stability Regime",
           xlabel="Ex-post descriptive stability tertile", ylabel="Annualised Sharpe ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regime_sharpe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote paper analysis to {OUT}")


if __name__ == "__main__":
    main()
