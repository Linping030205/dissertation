"""
spread_features.py
==================
Constructs the daily feature matrix for the S-SM (Soybeans / Soybean Meal)
cointegrating pair used in RL training.

At every date t, the hedge ratio beta_t is estimated by OLS on the preceding
BETA_WINDOW trading days only — no look-ahead bias.

Features produced
-----------------
  LogS          log closing price of Soybeans (CQG: S)
  LogSM         log closing price of Soybean Meal (CQG: SM)
  RollingBeta   OLS hedge ratio estimated on [t-BETA_WINDOW, t] window
  RollingAlpha  OLS intercept from the same regression
  Spread        log(S_t) - beta_t * log(SM_t) - alpha_t
                (signal-generation level series; beta_t/alpha_t are
                re-estimated every day, so this level series itself is
                NOT meant to represent a tradeable position)
  SpreadReturn  tradeable realised return of a position opened at t-1:
                dlogS_t - beta_{t-1} * dlogSM_t
                (uses the hedge ratio actually in force during [t-1, t],
                not diff(Spread), which would also pick up day-to-day
                re-estimation drift in beta_t/alpha_t that a trader
                holding a fixed position could not have realised)
  RollMean20    20-day rolling mean of Spread
  RollStd20     20-day rolling std  of Spread
  RollMean60    60-day rolling mean of Spread
  RollStd60     60-day rolling std  of Spread
  ZScore20      (Spread - RollMean20) / RollStd20
  RegimeStable  1 when S-SM passes the 3-criterion walk-forward screen
                (forward-filled from monthly evaluation dates to daily)
  RegimeEGScore -log10(eg_p), clipped to [0,6]; higher = more significant
                cointegration. Continuous companion to RegimeStable: eg_p
                keeps varying day to day even in stretches where the hard
                pass/fail flag is pinned at 0 (e.g. most of 2021-2025).
  RegimeHLScore 1/(1+half_life); higher = faster mean reversion.
  RegimeCVScore 1/(1+hedge_cv); higher = more stable hedge ratio.
                (All three forward-filled from the same monthly
                walk-forward evaluation as RegimeStable.)
  RegimeStabilityScore  Mean of the three scores above, each min-max
                normalised on TRAIN-PERIOD-ONLY bounds, in [0,1]. Used by
                reward shaping in place of the binary RegimeStable flag.

Output  (in analysis_output/)
------------------------------
  spread_features_SSM.xlsx    full feature table, one row per trading day
  figSF1_spread_zscore.png    three-panel diagnostic chart
  spread_features_report.txt  console log (UTF-8)
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from commodity_utils import load_commodities
from rl_env import TRAIN_END

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIG
# =========================================================================
DATA_DIR   = str(Path(__file__).resolve().parent)
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_output")

BETA_WINDOW   = 120   # rolling OLS window for hedge ratio estimation (days)
ZSCORE_WINDOW = 20    # rolling window for Z-score normalisation (Gatev et al.)
LONG_WINDOW   = 60    # secondary rolling window for reference bands


# =========================================================================
# LOGGING
# =========================================================================
class _SafeTee:
    def __init__(self, console, log_file):
        self.console  = console
        self.log_file = log_file

    def write(self, obj):
        try:
            enc = getattr(self.console, "encoding", "utf-8") or "utf-8"
            self.console.write(obj.encode(enc, errors="replace").decode(enc))
        except Exception:
            pass
        self.log_file.write(obj)
        self.log_file.flush()

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        self.log_file.flush()


_log = open(os.path.join(OUTPUT_DIR, "spread_features_report.txt"),
            "w", encoding="utf-8")
sys.stdout = _SafeTee(sys.__stdout__, _log)


def _save(fname):
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fname}")


plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =========================================================================
# LOAD DATA
# =========================================================================
print("=" * 70)
print("  SPREAD FEATURE CONSTRUCTION: S-SM  (Soybeans / Soybean Meal)")
print(f"  Beta window   : {BETA_WINDOW} days (rolling OLS, no look-ahead)")
print(f"  Z-score window: {ZSCORE_WINDOW} days")
print(f"  Long window   : {LONG_WINDOW} days (reference only)")
print("=" * 70)

dfs, common_idx, _ = load_commodities(DATA_DIR)

# Drop any rows where either log price is NaN/inf (mirrors walk-forward .dropna())
_lp = pd.DataFrame({"S": dfs["S"]["LogPrice"],
                    "SM": dfs["SM"]["LogPrice"]}).dropna()
log_s      = _lp["S"]
log_sm     = _lp["SM"]
common_idx = _lp.index
n          = len(common_idx)

print(f"\nAligned trading days : {n}")
print(f"Date range           : {common_idx[0].date()}  to  {common_idx[-1].date()}")


# =========================================================================
# ROLLING OLS BETA  (the hot loop — ~2-3 s for n=5666)
# =========================================================================
print(f"\nEstimating rolling OLS beta/alpha (window = {BETA_WINDOW} days)...")

beta_arr  = np.full(n, np.nan)
alpha_arr = np.full(n, np.nan)

for i in range(BETA_WINDOW - 1, n):
    lp_s  = log_s.iloc[i - BETA_WINDOW + 1 : i + 1].values
    lp_sm = log_sm.iloc[i - BETA_WINDOW + 1 : i + 1].values
    X     = sm.add_constant(lp_sm)
    fit   = sm.OLS(lp_s, X).fit()
    alpha_arr[i] = fit.params[0]
    beta_arr[i]  = fit.params[1]

valid_beta = np.sum(~np.isnan(beta_arr))
print(f"  Valid beta estimates : {valid_beta} of {n}  "
      f"(warm-up = {n - valid_beta} days)")
print(f"  Beta range : [{np.nanmin(beta_arr):.4f},  {np.nanmax(beta_arr):.4f}]")
print(f"  Beta mean  : {np.nanmean(beta_arr):.4f}   std: {np.nanstd(beta_arr):.4f}")


# =========================================================================
# BUILD FEATURE DATAFRAME
# =========================================================================
spread_arr = log_s.values - beta_arr * log_sm.values - alpha_arr

feat = pd.DataFrame(index=common_idx)
feat["LogS"]         = log_s.values
feat["LogSM"]        = log_sm.values
feat["RollingBeta"]  = beta_arr
feat["RollingAlpha"] = alpha_arr
feat["Spread"]       = spread_arr

# Tradeable return: a position opened at t-1 uses beta_{t-1} (the hedge
# ratio actually in force), not the re-estimated beta_t. diff(Spread)
# would additionally include the day-to-day drift of beta_t/alpha_t
# themselves, which is not P&L a trader could have realised.
feat["SpreadReturn"] = (
    feat["LogS"].diff() - feat["RollingBeta"].shift(1) * feat["LogSM"].diff()
)

feat["RollMean20"] = feat["Spread"].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
feat["RollStd20"]  = feat["Spread"].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()
feat["RollMean60"] = feat["Spread"].rolling(LONG_WINDOW,   min_periods=LONG_WINDOW).mean()
feat["RollStd60"]  = feat["Spread"].rolling(LONG_WINDOW,   min_periods=LONG_WINDOW).std()
feat["ZScore20"]   = (feat["Spread"] - feat["RollMean20"]) / feat["RollStd20"]

# 5-day change in rolling beta (early proxy for hedge-ratio drift)
feat["DeltaBeta5d"] = feat["RollingBeta"].diff(5)

# 20-day rolling mean volume for S and SM (0-volume days treated as NaN)
_vol_s  = dfs["S"]["Volume"].reindex(common_idx).replace(0, np.nan)
_vol_sm = dfs["SM"]["Volume"].reindex(common_idx).replace(0, np.nan)
feat["VolS_roll20"]  = _vol_s.rolling(20,  min_periods=15).mean()
feat["VolSM_roll20"] = _vol_sm.rolling(20, min_periods=15).mean()


# =========================================================================
# REGIME STABLE FLAG  (forward-filled from walkforward_results.xlsx)
# =========================================================================
print("\nLoading regime stability signal from walkforward_results.xlsx ...")

wf_path = os.path.join(OUTPUT_DIR, "walkforward_results.xlsx")
df_wf   = pd.read_excel(wf_path, sheet_name="AllRecords")

df_ssm  = (df_wf[df_wf["pair"] == "S-SM"][["eval_date", "all_pass"]]
           .copy()
           .assign(eval_date=lambda d: pd.to_datetime(d["eval_date"]))
           .set_index("eval_date")["all_pass"])

# Reindex to daily common_idx and forward-fill: the last screening outcome
# remains valid until the next monthly re-evaluation. Dates before the first
# evaluation remain unavailable (NaN); they must not be labelled "unstable".
regime = df_ssm.reindex(common_idx).ffill().astype("Float64")
feat["RegimeStable"] = regime.values

stable_days   = feat["RegimeStable"].sum()
total_valid   = feat["RegimeStable"].notna().sum()
print(f"  Walk-forward eval dates: {len(df_ssm)}")
print(f"  First available date   : {regime.first_valid_index().date()}")
print(f"  Daily RegimeStable=1   : {int(stable_days)} "
      f"({stable_days/total_valid*100:.1f}% of regime-available days)")

# ---- continuous regime diagnostics (same source, forward-filled) ----
# The hard RegimeStable flag is ~0% for most of 2021-2025 (see
# spread_features_report / walkforward_report), so it carries almost no
# state information during that stretch even though the underlying
# continuous statistics (eg_p, half_life, hedge_cv) keep varying day to
# day. These give the model real signal to condition on throughout.
df_ssm_full = (df_wf[df_wf["pair"] == "S-SM"]
              [["eval_date", "eg_p", "half_life", "hedge_cv"]]
              .copy()
              .assign(eval_date=lambda d: pd.to_datetime(d["eval_date"]))
              .set_index("eval_date"))
_reg_daily = df_ssm_full.reindex(common_idx).ffill()

feat["RegimeEGScore"] = -np.log10(_reg_daily["eg_p"].clip(lower=1e-6, upper=1.0))
feat["RegimeHLScore"] = 1.0 / (1.0 + _reg_daily["half_life"].replace(np.inf, 1e6).clip(lower=0))
feat["RegimeCVScore"] = 1.0 / (1.0 + _reg_daily["hedge_cv"].replace(np.inf, 1e6).clip(lower=0))

print(f"  RegimeEGScore : mean={feat['RegimeEGScore'].mean():.3f}  "
      f"std={feat['RegimeEGScore'].std():.3f}  "
      f"[{feat['RegimeEGScore'].min():.3f}, {feat['RegimeEGScore'].max():.3f}]")
print(f"  RegimeHLScore : mean={feat['RegimeHLScore'].mean():.3f}  "
      f"std={feat['RegimeHLScore'].std():.3f}  "
      f"[{feat['RegimeHLScore'].min():.3f}, {feat['RegimeHLScore'].max():.3f}]")
print(f"  RegimeCVScore : mean={feat['RegimeCVScore'].mean():.3f}  "
      f"std={feat['RegimeCVScore'].std():.3f}  "
      f"[{feat['RegimeCVScore'].min():.3f}, {feat['RegimeCVScore'].max():.3f}]")

# ---- combined continuous stability score, bounded [0,1] ----
# Used by reward shaping (rl_env.py) IN PLACE OF the binary RegimeStable,
# which is ~0% for most of 2021-2025 and so reduces the reward penalty to
# an almost-constant multiplier during that stretch rather than something
# that actually tracks changing conditions. Each of the three diagnostics
# is min-max normalised using TRAIN-PERIOD-ONLY bounds (no test-period
# information used to define the scale) and averaged; no new hard
# threshold is introduced anywhere in this computation.
_train_mask = feat.index <= pd.Timestamp(TRAIN_END)

def _minmax_train(col: str) -> pd.Series:
    train_vals = feat.loc[_train_mask, col].dropna()
    lo, hi = train_vals.min(), train_vals.max()
    return ((feat[col] - lo) / (hi - lo + 1e-12)).clip(0.0, 1.0)

feat["RegimeStabilityScore"] = (
    _minmax_train("RegimeEGScore")
    + _minmax_train("RegimeHLScore")
    + _minmax_train("RegimeCVScore")
) / 3.0

print(f"  RegimeStabilityScore : mean={feat['RegimeStabilityScore'].mean():.3f}  "
      f"std={feat['RegimeStabilityScore'].std():.3f}  "
      f"[{feat['RegimeStabilityScore'].min():.3f}, {feat['RegimeStabilityScore'].max():.3f}]")


# =========================================================================
# SUMMARY STATISTICS
# =========================================================================
valid = feat.dropna(subset=["ZScore20"])
warmup_days = n - len(valid)

print(f"\n{'='*70}")
print(f"  FEATURE SUMMARY  ({len(valid)} valid rows after {warmup_days}-day warm-up)")
print(f"{'='*70}")

print(f"\nSpread  (log(S) - beta*log(SM) - alpha):")
print(f"  Mean   : {feat['Spread'].mean():.6f}")
print(f"  Std    : {feat['Spread'].std():.6f}")
print(f"  Min    : {feat['Spread'].min():.6f}")
print(f"  Max    : {feat['Spread'].max():.6f}")

print(f"\nZ-score (window={ZSCORE_WINDOW}d):")
print(f"  Mean      : {valid['ZScore20'].mean():.4f}  (expect ~0 if spread is stationary)")
print(f"  Std       : {valid['ZScore20'].std():.4f}   (expect ~1 if normalisation holds)")
print(f"  Min / Max : [{valid['ZScore20'].min():.4f},  {valid['ZScore20'].max():.4f}]")
q = valid["ZScore20"].quantile([.01, .05, .25, .50, .75, .95, .99])
print(f"  Percentiles: 1%={q[.01]:.2f}  5%={q[.05]:.2f}  25%={q[.25]:.2f}  "
      f"50%={q[.50]:.2f}  75%={q[.75]:.2f}  95%={q[.95]:.2f}  99%={q[.99]:.2f}")
print(f"  |Z| > 2 : {(valid['ZScore20'].abs() > 2).mean()*100:.1f}% of valid days")
print(f"  |Z| > 3 : {(valid['ZScore20'].abs() > 3).mean()*100:.1f}% of valid days")

print(f"\nZ-score split by Regime:")
stable_z   = valid.loc[valid["RegimeStable"] == 1, "ZScore20"]
unstable_z = valid.loc[valid["RegimeStable"] == 0, "ZScore20"]
for label, z in [("Stable  ", stable_z), ("Unstable", unstable_z)]:
    if len(z) == 0:
        continue
    print(f"  {label} (n={len(z):5d}): "
          f"mean={z.mean():+.3f}  std={z.std():.3f}  "
          f"|Z|>2={( z.abs()>2).mean()*100:.1f}%  "
          f"|Z|>3={( z.abs()>3).mean()*100:.1f}%")

print(f"\nRolling beta (hedge ratio):")
print(f"  Mean   : {np.nanmean(beta_arr):.4f}")
print(f"  Std    : {np.nanstd(beta_arr):.4f}")
print(f"  Min/Max: [{np.nanmin(beta_arr):.4f},  {np.nanmax(beta_arr):.4f}]")

# Year-by-year regime stable rate
print(f"\nYear-by-year RegimeStable rate:")
feat["Year"] = feat.index.year
yr = feat.groupby("Year")["RegimeStable"].mean() * 100
for y, pct in yr.items():
    if pd.isna(pct):
        print(f"  {y}:   N/A   (screen unavailable)")
        continue
    bar = "#" * int(pct / 5)
    print(f"  {y}: {pct:5.1f}%  {bar}")
feat.drop(columns=["Year"], inplace=True)


# =========================================================================
# FIGURE: figSF1_spread_zscore.png
# =========================================================================
print("\nPlotting figSF1_spread_zscore.png ...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 2]})
fig.suptitle(
    "S-SM  Spread Feature Construction\n"
    f"Rolling OLS beta (window={BETA_WINDOW}d)  •  Z-score (window={ZSCORE_WINDOW}d)  •  "
    "Green shading = RegimeStable=1",
    fontsize=13, fontweight="bold",
)


def _shade_regimes(ax, dates, regime_arr,
                   color_stable="#2E7D32", color_unstable="#AAAAAA",
                   alpha_s=0.13, alpha_u=0.05):
    """
    Shade stable vs unstable regime periods on the given axes.
    Iterates over regime changes and calls axvspan for each contiguous segment.
    """
    if len(dates) == 0:
        return
    valid_positions = np.flatnonzero(pd.notna(regime_arr))
    if len(valid_positions) == 0:
        return
    first = int(valid_positions[0])
    current = regime_arr[first]
    start   = dates[first]
    for i in range(first + 1, len(dates)):
        if pd.isna(regime_arr[i]):
            continue
        if regime_arr[i] != current:
            color = color_stable if current == 1 else color_unstable
            alpha = alpha_s       if current == 1 else alpha_u
            ax.axvspan(start, dates[i - 1], color=color, alpha=alpha, lw=0)
            current = regime_arr[i]
            start   = dates[i]
    # last segment
    color = color_stable if current == 1 else color_unstable
    alpha = alpha_s if current == 1 else alpha_u
    ax.axvspan(start, dates[-1], color=color, alpha=alpha, lw=0)


dates_arr  = feat.index.to_numpy()
regime_arr = feat["RegimeStable"].to_numpy(dtype=float, na_value=np.nan)

# ---- Panel 1: Spread + rolling bands ----
ax = axes[0]
_shade_regimes(ax, dates_arr, regime_arr)
ax.plot(feat.index, feat["Spread"],
        color="#1565C0", linewidth=0.65, alpha=0.85, label="Spread")
ax.plot(feat.index, feat["RollMean20"],
        color="#C62828", linewidth=1.1, linestyle="--",
        label=f"Roll.Mean {ZSCORE_WINDOW}d")
ax.fill_between(feat.index,
                feat["RollMean20"] - 2 * feat["RollStd20"],
                feat["RollMean20"] + 2 * feat["RollStd20"],
                color="#C62828", alpha=0.10, label=r"Mean ± 2$\sigma$")
ax.set_ylabel("Spread  [log(S) − β·log(SM) − α]")
ax.set_title(f"Panel 1: Spread  ({ZSCORE_WINDOW}-day rolling mean ± 2σ)")
ax.legend(loc="upper right", ncol=3, fontsize=8)

# ---- Panel 2: Z-score ----
ax = axes[1]
_shade_regimes(ax, dates_arr, regime_arr)
ax.plot(feat.index, feat["ZScore20"],
        color="#1565C0", linewidth=0.65, alpha=0.85,
        label=f"Z-score ({ZSCORE_WINDOW}d)")
ax.axhline( 2, color="#C62828", linewidth=1.0, linestyle="--", label="±2σ  (signal)")
ax.axhline(-2, color="#C62828", linewidth=1.0, linestyle="--")
ax.axhline( 1, color="#FF8F00", linewidth=0.7, linestyle=":",  label="±1σ")
ax.axhline(-1, color="#FF8F00", linewidth=0.7, linestyle=":")
ax.axhline( 0, color="black",   linewidth=0.5)
ax.set_ylabel("Z-score")
ax.set_ylim(-7, 7)
ax.set_title("Panel 2: Z-score  (|Z| > 2 = classic entry threshold)")
ax.legend(loc="upper right", ncol=3, fontsize=8)

# ---- Panel 3: Rolling beta + RegimeStable ----
ax   = axes[2]
ax_r = ax.twinx()

ax.plot(feat.index, feat["RollingBeta"],
        color="#6A1B9A", linewidth=0.9, label=f"Rolling Beta ({BETA_WINDOW}d)")
ax.set_ylabel("Beta  (hedge ratio)", color="#6A1B9A")
ax.tick_params(axis="y", labelcolor="#6A1B9A")

ax_r.fill_between(feat.index, feat["RegimeStable"],
                  step="post", color="#2E7D32", alpha=0.40,
                  label="RegimeStable = 1")
ax_r.set_ylabel("RegimeStable  (0/1)", color="#2E7D32")
ax_r.set_ylim(-0.2, 2.8)
ax_r.set_yticks([0, 1])
ax_r.tick_params(axis="y", labelcolor="#2E7D32")

ax.set_title(
    "Panel 3: Rolling OLS Beta (hedge ratio drift)  +  RegimeStable flag\n"
    "Beta instability in 2021-2022 accompanies the broader 2021-2025 breakdown; "
    "the later episode follows the Russia-Ukraine supply shock"
)

lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=8)

# X-axis tick formatting (bottom panel only, sharex propagates)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(axes[-1].get_xticklabels(), rotation=30)

plt.tight_layout()
_save("figSF1_spread_zscore.png")


# =========================================================================
# EXPORT TO EXCEL
# =========================================================================
out_cols = [
    "LogS", "LogSM",
    "RollingBeta", "RollingAlpha",
    "Spread", "SpreadReturn",
    "RollMean20", "RollStd20",
    "RollMean60", "RollStd60",
    "ZScore20",
    "DeltaBeta5d",
    "VolS_roll20", "VolSM_roll20",
    "RegimeStable",
    "RegimeEGScore", "RegimeHLScore", "RegimeCVScore",
    "RegimeStabilityScore",
]
feat[out_cols].to_excel(
    os.path.join(OUTPUT_DIR, "spread_features_SSM.xlsx"),
    sheet_name="Features",
)
print("[saved] spread_features_SSM.xlsx")

print("\n" + "=" * 70)
print("  SPREAD FEATURE CONSTRUCTION COMPLETE")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 70)

_log.close()
sys.stdout = sys.__stdout__
print(f"\nDone. Output -> {OUTPUT_DIR}")
