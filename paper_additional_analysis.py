"""Additional mechanism analyses for the final dissertation results.

Uses only the audited v6 daily paths and the existing feature workbook.
No model is retrained and no test observation is used to fit a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
OUT = WF / "paper_analysis"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"

FEATURE_COLS = [
    "ZScore20", "SpreadReturn", "RollStd20", "DeltaBeta5d",
    "VolS_roll20", "VolSM_roll20", "RegimeEGScore",
    "RegimeHLScore", "RegimeCVScore",
]
REGIME_COLS = ["RegimeEGScore", "RegimeHLScore", "RegimeCVScore",
               "RegimeStabilityScore"]
YEARS = range(2021, 2026)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Serif", "font.size": 10,
})


def feature_correlation(data):
    """Fold-training-only Spearman correlations, then aggregate by cell."""
    matrices = []
    long_rows = []
    for test_year in YEARS:
        train_end = f"{test_year - 2}-12-31"
        train = data.loc[:train_end, FEATURE_COLS].dropna()
        corr = train.corr(method="spearman")
        matrices.append(corr)
        for a in FEATURE_COLS:
            for b in FEATURE_COLS:
                long_rows.append({"test_year": test_year,
                                  "train_end": train_end,
                                  "feature_a": a, "feature_b": b,
                                  "spearman_rho": corr.loc[a, b],
                                  "N_train": len(train)})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "feature_correlation_by_fold.csv", index=False)
    stack = np.stack([x.to_numpy() for x in matrices])
    mean = pd.DataFrame(stack.mean(0), index=FEATURE_COLS, columns=FEATURE_COLS)
    lo = pd.DataFrame(stack.min(0), index=FEATURE_COLS, columns=FEATURE_COLS)
    hi = pd.DataFrame(stack.max(0), index=FEATURE_COLS, columns=FEATURE_COLS)
    rows = []
    for a in FEATURE_COLS:
        for b in FEATURE_COLS:
            rows.append({"feature_a": a, "feature_b": b,
                         "MeanSpearman": mean.loc[a, b],
                         "MinAcrossFolds": lo.loc[a, b],
                         "MaxAcrossFolds": hi.loc[a, b]})
    pd.DataFrame(rows).to_csv(OUT / "feature_correlation_summary.csv", index=False)

    labels = ["Z-score", "Spread ret.", "Spread vol.", "Delta beta",
              "S volume", "SM volume", "EG score", "HL score", "CV score"]
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    im = ax.imshow(mean, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mean.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > .58 else "black")
    ax.set_title("Training-Only Feature Correlations\nMean Spearman rho across five walk-forward folds")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_correlation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _spell_stats(group):
    """Position-transition and non-zero holding-spell statistics."""
    group = group.sort_values(["test_year", "date"])
    entries = exits = reversals = changes = forced_closes = 0
    spells = []
    for _, year_data in group.groupby("test_year"):
        p = year_data["position"].to_numpy(dtype=int)
        prev = np.r_[0, p[:-1]]
        changes += int(np.sum(p != prev))
        entries += int(np.sum((prev == 0) & (p != 0)))
        exits += int(np.sum((prev != 0) & (p == 0)))
        reversals += int(np.sum((prev != 0) & (p != 0) & (prev != p)))
        # The environment charges an economic close-out at each fold end,
        # although the final recorded target position remains non-zero.
        if len(p) and p[-1] != 0:
            forced_closes += 1
            exits += 1
            changes += 1
        start = None
        current = 0
        for i, value in enumerate(p):
            if value != current:
                if current != 0 and start is not None:
                    spells.append(i - start)
                start = i if value != 0 else None
                current = value
        if current != 0 and start is not None:
            spells.append(len(p) - start)
    gross = float(group["gross_pnl"].sum())
    cost = float(group["cost"].sum())
    n = len(group)
    return {
        "N_days": n, "PositionChanges": changes,
        "AnnualisedPositionChanges": changes / n * 252,
        "Entries": entries, "Exits": exits, "ForcedCloses": forced_closes,
        "DirectReversals": reversals,
        "MeanNonzeroHoldingDays": float(np.mean(spells)) if spells else np.nan,
        "MedianNonzeroHoldingDays": float(np.median(spells)) if spells else np.nan,
        "PctFlat": float((group["position"] == 0).mean() * 100),
        "GrossPnL": gross, "TotalCost": cost,
        "NetPnL": float(group["net_pnl"].sum()),
        "CostAsPctGrossMagnitude": (cost / abs(gross) * 100
                                     if abs(gross) > 1e-12 else np.nan),
    }


def turnover_analysis():
    rl = pd.read_csv(OUT / "rl_daily_gross_cost_net.csv", parse_dates=["date"])
    seed_rows = []
    for (strategy, seed), g in rl.groupby(["strategy", "seed"]):
        seed_rows.append({"strategy": strategy, "seed": seed, **_spell_stats(g)})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "turnover_holding_seed_results.csv", index=False)
    metrics = [c for c in seed_df.columns if c not in ("strategy", "seed")]
    summary = seed_df.groupby("strategy")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b.capitalize()}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["CostAsPctAbsoluteMeanGross"] = (
        summary["TotalCost_Mean"] / summary["GrossPnL_Mean"].abs() * 100
    )
    summary.to_csv(OUT / "turnover_holding_summary.csv", index=False)

    # Deterministic baselines are reported separately to avoid pretending
    # that their single realised paths provide a seed standard deviation.
    base = pd.read_csv(OUT / "baseline_daily.csv", parse_dates=["date"])
    base = base.rename(columns={"pnl": "net_pnl"})
    base_rows = [{"strategy": strategy, **_spell_stats(g)}
                 for strategy, g in base.groupby("strategy")]
    pd.DataFrame(base_rows).to_csv(OUT / "turnover_holding_baselines.csv",
                                   index=False)


def regime_timeline(data):
    plot = data.loc["2005-01-01":"2025-12-31", REGIME_COLS]
    titles = ["Cointegration component", "Half-life component",
              "Hedge-stability component", "Composite stability score"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, col, title in zip(axes, REGIME_COLS, titles):
        ax.plot(plot.index, plot[col], lw=.9, color="#2468a2")
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"),
                   color="#d95f02", alpha=.12, label="Formal OOS 2021-2025")
        ax.axvline(pd.Timestamp("2021-01-01"), color="#d95f02", lw=.9, ls="--")
        ax.set_ylabel("Score")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Evolution of Regime-Stability Components", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT / "fig_regime_component_timeline.png", dpi=320,
                bbox_inches="tight")
    plt.close(fig)


def spread_tail_statistics(data):
    rl_dates = pd.read_csv(OUT / "rl_daily_gross_cost_net.csv",
                           usecols=["date"], parse_dates=["date"])["date"].unique()
    rows = []
    periods = [("Training through 2019", data.loc[:"2019-12-31", "SpreadReturn"]),
               ("Formal OOS evaluation dates 2021-2025",
                data.reindex(pd.DatetimeIndex(rl_dates))["SpreadReturn"])]
    periods += [(str(y), data.loc[f"{y}-01-01":f"{y}-12-31", "SpreadReturn"])
                for y in YEARS]
    for label, values in periods:
        x = values.dropna().to_numpy(dtype=float)
        jb = stats.jarque_bera(x)
        q05 = float(np.quantile(x, .05))
        q95 = float(np.quantile(x, .95))
        rows.append({
            "Period": label, "N": len(x), "Mean": np.mean(x),
            "Std": np.std(x, ddof=1), "Skewness": stats.skew(x, bias=False),
            "ExcessKurtosis": stats.kurtosis(x, fisher=True, bias=False),
            "Min": np.min(x), "P01": np.quantile(x, .01), "P05": q05,
            "P95": q95, "P99": np.quantile(x, .99), "Max": np.max(x),
            "LowerTailMean5Pct": np.mean(x[x <= q05]),
            "UpperTailMean5Pct": np.mean(x[x >= q95]),
            "JarqueBera": jb.statistic, "JarqueBeraP": jb.pvalue,
        })
    pd.DataFrame(rows).to_csv(OUT / "spread_return_tail_statistics.csv",
                              index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(FEATURES, sheet_name="Features", index_col=0,
                         parse_dates=True).sort_index()
    feature_correlation(data)
    turnover_analysis()
    regime_timeline(data)
    spread_tail_statistics(data)
    print(f"Additional dissertation analyses written to {OUT}")


if __name__ == "__main__":
    main()
