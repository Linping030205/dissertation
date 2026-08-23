"""
data_analysis.py  (static full-sample analysis)
================================================
Runs the complete data analysis for the inter-commodity spread
trading dissertation.  All statistical functions and shared config
are imported from commodity_utils.py.

Output folder: <DATA_DIR>/analysis_output/
"""

import io
import os
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

from commodity_utils import (
    DATA_FILES, FULL_NAMES, SECTORS, UNITS, POOL, PAIRS,
    TH_EG_PVAL, TH_HALF_LIFE, TH_HURST, TH_HEDGECV,
    load_commodities, eg_test, screen_pair,
)

warnings.filterwarnings("ignore")

# =========================================================================
# LOCAL CONFIG  (paths + plot aesthetics)
# =========================================================================
DATA_DIR   = str(Path(__file__).resolve().parent)
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLS5     = ["#1565C0", "#C62828", "#FF8F00", "#2E7D32", "#6A1B9A"]
COLOR_MAP = dict(zip(POOL, COLS5))


class _SafeTee:
    """Write to both console (encoding-safe) and a UTF-8 log file."""
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


_log = open(os.path.join(OUTPUT_DIR, "analysis_report.txt"), "w", encoding="utf-8")
sys.stdout = _SafeTee(sys.__stdout__, _log)


def _save(fname):
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fname}")


def _adf(series, label):
    """Print a single ADF result row."""
    res     = adfuller(series.dropna(), autolag="AIC")
    stat, p = res[0], res[1]
    c5      = res[4]["5%"]
    verdict = "Stationary" if p < 0.05 else "Non-stationary"
    print(f"  {label:<32} ADF={stat:>8.3f}  p={p:>7.4f}  "
          f"crit5%={c5:>7.3f}  -> {verdict}")


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
print("  INTER-COMMODITY SPREAD TRADING -- DATA ANALYSIS")
print("  Commodity pool: CL / NG / HG / S / SM")
print("  Source: Bloomberg Terminal")
print("=" * 70)

dfs, common_idx, extras = load_commodities(DATA_DIR, supplementary=("CL2",))
raw_cl2 = extras["CL2"]

print(f"\nCommon trading days after inner-join: {len(common_idx)}")
print(f"Period: {common_idx[0].date()} to {common_idx[-1].date()}")

# =========================================================================
# A2  MISSING VALUE ANALYSIS  (pre-alignment)
# =========================================================================
print("\n" + "=" * 70)
print("  A2: MISSING VALUE ANALYSIS  (before inner-join)")
print("=" * 70)

FIELD_COLS = ["Close", "Open", "High", "Low", "Volume", "OpenInterest"]
for key in POOL:
    from commodity_utils import _read_excel, DATA_FILES as _DF
    df_raw = _read_excel(os.path.join(DATA_DIR, _DF[key]))
    miss   = df_raw[FIELD_COLS].isnull().sum()
    pct    = (miss / len(df_raw) * 100).round(2)
    print(f"\n{key} -- {FULL_NAMES[key]}  ({UNITS[key]})")
    print(f"  Raw rows: {len(df_raw)}")
    for col in FIELD_COLS:
        if miss[col] > 0:
            print(f"  {col:<16}: {miss[col]:>3} rows ({pct[col]:.2f}%)")
    if miss[FIELD_COLS].sum() == 0:
        print("  -> No missing values detected.")

# =========================================================================
# A3  DESCRIPTIVE STATISTICS  (aligned sample)
# =========================================================================
print("\n" + "=" * 70)
print("  A3: DESCRIPTIVE STATISTICS  (aligned sample)")
print("=" * 70)

for key in POOL:
    df = dfs[key]
    print(f"\n--- {key}: {FULL_NAMES[key]} ({UNITS[key]}) ---")
    for field in ["Close", "LogReturn"]:
        s = df[field].dropna()
        print(f"  {field}:")
        print(f"    n={len(s)}  mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")
        print(f"    skew={s.skew():.4f}  ex.kurt={s.kurtosis():.4f}")

print("\n--- Annualised Return Summary ---")
print(f"{'Code':<6} {'Full Name':<20} {'Sector':<12} "
      f"{'Ann.Mean':>10} {'Ann.Vol':>9} {'Sharpe':>8}")
print("-" * 68)
for key in POOL:
    lr  = dfs[key]["LogReturn"].dropna()
    mu  = lr.mean() * 252
    vol = lr.std()  * np.sqrt(252)
    print(f"{key:<6} {FULL_NAMES[key]:<20} {SECTORS[key]:<12} "
          f"{mu:>10.4f} {vol:>9.4f} {mu/vol:>8.4f}")

# =========================================================================
# figA1  PRICE HISTORY (normalised + log-scale)
# =========================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
for key in POOL:
    p    = dfs[key]["Close"]
    norm = p / p.iloc[0] * 100
    ax.plot(p.index, norm, color=COLOR_MAP[key], linewidth=0.85,
            label=f"{key} - {FULL_NAMES[key]}")
ax.set_title("Normalised Price Index (Base = 100, Aligned Start Date)")
ax.set_ylabel("Index")
ax.legend(loc="upper left")

ax = axes[1]
for key in POOL:
    ax.plot(dfs[key].index, dfs[key]["Close"], color=COLOR_MAP[key],
            linewidth=0.8, label=f"{key} ({UNITS[key]})", alpha=0.85)
ax.set_title("Actual Closing Prices (Log Scale)")
ax.set_ylabel("Price")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax.get_xticklabels(), rotation=30)

fig.suptitle("Five-Commodity Pool: Price History (2004-2026)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save("figA1_price_history.png")

# =========================================================================
# figA2  RETURN DISTRIBUTIONS + JARQUE-BERA
# =========================================================================
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
fig.suptitle("Log Return Distributions: All Five Commodities",
             fontsize=13, fontweight="bold")

print("\n" + "=" * 70)
print("  A5: NORMALITY TEST -- Jarque-Bera")
print("=" * 70)
print(f"\n{'Code':<6} {'JB Stat':>14} {'p-value':>12} "
      f"{'Skew':>8} {'Ex.Kurt':>9} {'Normal?':>8}")
print("-" * 62)

for i, key in enumerate(POOL):
    lr = dfs[key]["LogReturn"].dropna()
    x  = np.linspace(lr.min(), lr.max(), 400)

    ax1 = axes[i, 0]
    ax1.hist(lr, bins=100, density=True, color=COLOR_MAP[key],
             alpha=0.55, edgecolor="none")
    ax1.plot(x, stats.norm.pdf(x, lr.mean(), lr.std()),
             "k--", linewidth=1.4, label="Normal fit")
    ax1.set_title(f"{key} ({FULL_NAMES[key]}): Return Distribution")
    ax1.set_xlabel("Log Return")
    ax1.set_ylabel("Density")
    ax1.legend()

    jb, jb_p = stats.jarque_bera(lr)
    ax1.text(0.97, 0.97,
             f"Skew={lr.skew():.3f}\nEx.Kurt={lr.kurtosis():.2f}\n"
             f"JB p={jb_p:.2e}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))

    ax2 = axes[i, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(lr, dist="norm")
    ax2.scatter(osm, osr, color=COLOR_MAP[key], s=1.0, alpha=0.3)
    xl = np.array([osm[0], osm[-1]])
    ax2.plot(xl, slope * xl + intercept, "k-", linewidth=1.2)
    ax2.set_title(f"{key}: Normal Q-Q Plot")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")

    print(f"{key:<6} {jb:>14.1f} {jb_p:>12.4e} "
          f"{lr.skew():>8.4f} {lr.kurtosis():>9.4f} "
          f"{'No' if jb_p < 0.05 else 'Yes':>8}")

print("\nH0: normally distributed. p < 0.05 -> reject H0.")
plt.tight_layout()
_save("figA2_return_distributions.png")

# =========================================================================
# figA3  ROLLING VOLATILITY + ARCH-LM TEST
# =========================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax_i, window in enumerate([21, 63]):
    ax = axes[ax_i]
    for key in POOL:
        rv = dfs[key]["LogReturn"].rolling(window).std() * np.sqrt(252) * 100
        ax.plot(dfs[key].index, rv, color=COLOR_MAP[key],
                linewidth=0.8, label=key, alpha=0.85)
    ax.set_title(f"{window}-Day Rolling Annualised Volatility")
    ax.set_ylabel("Volatility (% p.a.)")
    ax.legend(loc="upper right")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(axes[-1].get_xticklabels(), rotation=30)
fig.suptitle("Rolling Annualised Volatility: Five-Commodity Pool",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("figA3_rolling_volatility.png")

print("\n" + "=" * 70)
print("  A6: ARCH-LM TEST -- Volatility Clustering")
print("=" * 70)
print(f"\n{'Code':<6} {'Lags':>6} {'LM Stat':>10} {'p-value':>12} {'ARCH?':>8}")
print("-" * 48)
for key in POOL:
    lr = dfs[key]["LogReturn"].dropna()
    for lags in [5, 10, 20]:
        lm, pv, _, _ = het_arch(lr, nlags=lags)
        print(f"{key:<6} {lags:>6} {lm:>10.3f} {pv:>12.4e} "
              f"{'Yes' if pv < 0.05 else 'No':>8}")
print("\nH0: no ARCH effects. p < 0.05 -> volatility clustering present.")

# =========================================================================
# A7  ADF STATIONARITY TEST
# =========================================================================
print("\n" + "=" * 70)
print("  A7: ADF STATIONARITY TEST")
print("=" * 70)

print("\nLog Price Levels:")
for key in POOL:
    _adf(dfs[key]["LogPrice"], f"{key} log price")

print("\nLog Returns:")
for key in POOL:
    _adf(dfs[key]["LogReturn"], f"{key} log return")

# =========================================================================
# B1  INTER-COMMODITY CORRELATION
# =========================================================================
print("\n" + "=" * 70)
print("  B1: INTER-COMMODITY CORRELATION ANALYSIS")
print("=" * 70)

price_mat  = pd.DataFrame({k: dfs[k]["LogPrice"]  for k in POOL}).dropna()
return_mat = pd.DataFrame({k: dfs[k]["LogReturn"] for k in POOL}).dropna()
corr_price  = price_mat.corr()
corr_return = return_mat.corr()

print("\nLog Price Correlation Matrix:")
print(corr_price.round(4).to_string())
print("\nLog Return Correlation Matrix:")
print(corr_return.round(4).to_string())

print("\nPairwise Return Correlations (sorted descending):")
print(f"  {'Pair':<12} {'Return Corr':>13} {'Sector Relation'}")
print("  " + "-" * 50)
pair_corrs = [(a, b, corr_return.loc[a, b],
               "Intra-sector" if SECTORS[a]==SECTORS[b] else "Inter-sector")
              for a, b in PAIRS]
for a, b, r, rel in sorted(pair_corrs, key=lambda x: -abs(x[2])):
    print(f"  {a}-{b:<9} {r:>13.4f}  {rel}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, matrix, title in zip(
    axes,
    [corr_price, corr_return],
    ["Log Price Correlation", "Log Return Correlation"]
):
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 10})
    ax.set_title(title, fontsize=11, fontweight="bold")

sector_patches = [
    mpatches.Patch(color=COLS5[0], label="Energy: CL, NG"),
    mpatches.Patch(color=COLS5[2], label="Metals: HG"),
    mpatches.Patch(color=COLS5[3], label="Agriculture: S, SM"),
]
fig.legend(handles=sector_patches, loc="lower center", ncol=3,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Inter-Commodity Correlation: Log Prices & Log Returns",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("figB1_correlation_heatmap.png")

# =========================================================================
# B2  FOUR-CRITERION PAIR SCREENING
# =========================================================================
print("\n" + "=" * 70)
print("  B2: FOUR-CRITERION PAIR SCREENING  (10 candidate pairs)")
print("=" * 70)
print(f"\n  Thresholds:")
print(f"    i.   Engle-Granger p   < {TH_EG_PVAL}")
print(f"    ii.  OU half-life      < {TH_HALF_LIFE} trading days")
print(f"    iii. Hurst exponent    < {TH_HURST}")
print(f"    iv.  Hedge-ratio CV    < {TH_HEDGECV*100:.0f}%")
print()

screening_results = []
for a, b in PAIRS:
    lp_a = dfs[a]["LogPrice"].dropna()
    lp_b = dfs[b]["LogPrice"].dropna()
    idx  = lp_a.index.intersection(lp_b.index)
    res  = screen_pair(lp_a.loc[idx].values, lp_b.loc[idx].values)
    screening_results.append({
        "Pair":       f"{a}-{b}",
        "A": a, "B": b,
        "Relation":   "Intra" if SECTORS[a]==SECTORS[b] else "Inter",
        "spread_idx": idx,
        **res,
    })

# Print screening table
print(f"{'Pair':<9} {'EG p':>8} {'HL(d)':>8} {'Hurst':>7} "
      f"{'CV':>8} {'EG':>4} {'HL':>4} {'H':>4} {'CV':>4} {'ALL':>5}")
print("-" * 68)
for r in screening_results:
    hl_s = f"{r['half_life']:.1f}" if np.isfinite(r["half_life"]) else "inf"
    hstr = f"{r['hurst']:.4f}"     if np.isfinite(r["hurst"])     else "nan"
    cstr = f"{r['hedge_cv']:.4f}"  if np.isfinite(r["hedge_cv"])  else "inf"
    print(f"{r['Pair']:<9} {r['eg_p']:>8.4f} {hl_s:>8} {hstr:>7} {cstr:>8} "
          f"{'Y' if r['pass_eg'] else 'N':>4} "
          f"{'Y' if r['pass_hl'] else 'N':>4} "
          f"{'Y' if r['pass_h']  else 'N':>4} "
          f"{'Y' if r['pass_cv'] else 'N':>4} "
          f"{'PASS' if r['all_pass'] else 'fail':>5}")

# =========================================================================
# B3  RANKING & SELECTION
# =========================================================================
print("\n" + "=" * 70)
print("  B3: PAIR RANKING & SELECTION")
print("=" * 70)

survivors = sorted([r for r in screening_results if r["all_pass"]],
                   key=lambda r: r["half_life"])
print(f"\n{len(survivors)} pair(s) passed all four criteria.")

if survivors:
    print(f"\n{'Rank':<6} {'Pair':<9} {'Half-Life':>11} {'EG p':>8} "
          f"{'Hurst':>7} {'CV':>8} {'Relation':>9}")
    print("-" * 58)
    for rank, r in enumerate(survivors, 1):
        print(f"{rank:<6} {r['Pair']:<9} {r['half_life']:>11.1f} "
              f"{r['eg_p']:>8.4f} {r['hurst']:>7.4f} "
              f"{r['hedge_cv']:>8.4f} {r['Relation']:>9}")
    k = min(3, len(survivors))
    print(f"\nSelected top-{k} pairs (shortest half-life):")
    for r in survivors[:k]:
        print(f"  {r['Pair']}  HL={r['half_life']:.1f}d  {r['Relation']}")
else:
    print("\nNone survived full-sample screening.")
    print("-> This motivates the walk-forward / dynamic approach.")
    print("   See pair_screening_walkforward.py for rolling-window results.")
    print("\nPairs failing only ONE criterion:")
    for r in screening_results:
        n_fail = sum([not r["pass_eg"], not r["pass_hl"],
                      not r["pass_h"],  not r["pass_cv"]])
        if n_fail == 1:
            failed = [c for c in ["pass_eg","pass_hl","pass_h","pass_cv"]
                      if not r[c]]
            print(f"  {r['Pair']}: failed {failed[0]}")

# figB2 -- screening summary heatmap
fig, ax = plt.subplots(figsize=(10, 6))
labels   = [r["Pair"] for r in screening_results]
criteria = ["Cointegration\n(EG p<0.05)", "Half-Life\n(<40d)",
            "Hurst\n(<0.50)",             "Hedge CV\n(<50%)"]
heat = np.array([[int(r["pass_eg"]), int(r["pass_hl"]),
                  int(r["pass_h"]),  int(r["pass_cv"])]
                 for r in screening_results], dtype=float)

im = ax.imshow(heat, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4));  ax.set_xticklabels(criteria, fontsize=9)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(4):
        txt = "PASS" if heat[i, j] else "FAIL"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black" if heat[i, j] else "white")
for i, r in enumerate(screening_results):
    if r["all_pass"]:
        for j in range(4):
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                         fill=False, edgecolor="gold", lw=2.5))
ax.set_title("Four-Criterion Pair Screening: All 10 Candidate Pairs\n"
             "(Gold = passed all criteria)", fontsize=11, fontweight="bold")
plt.tight_layout()
_save("figB2_pair_screening.png")

# figB3 -- all 10 pair spreads
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle("Cointegrated Spread Series: All 10 Candidate Pairs",
             fontsize=13, fontweight="bold")
for i, r in enumerate(screening_results):
    ax  = axes.flatten()[i]
    sp  = pd.Series(r["spread"], index=r["spread_idx"])
    col = "green" if r["all_pass"] else "gray"
    ax.plot(sp.index, sp, color=col, linewidth=0.7, alpha=0.85)
    ax.axhline(sp.mean(), color="red", linestyle="--", linewidth=0.9)
    ax.fill_between(sp.index, sp.mean() + sp.std(),
                               sp.mean() - sp.std(),
                    alpha=0.12, color=col)
    hl_s = (f"{r['half_life']:.1f}d"
            if np.isfinite(r["half_life"]) else "inf")
    ax.set_title(
        f"{r['Pair']}  [{'PASS' if r['all_pass'] else 'fail'}]  "
        f"HL={hl_s}  H={r['hurst']:.3f}",
        fontsize=9, color="green" if r["all_pass"] else "black"
    )
    ax.set_ylabel("Spread")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
plt.tight_layout()
_save("figB3_all_pair_spreads.png")

# =========================================================================
# PART C  CRUSH SPREAD DEEP DIVE (S vs SM)
# =========================================================================
print("\n" + "=" * 70)
print("  PART C: CRUSH SPREAD DEEP DIVE  (S vs SM)")
print("=" * 70)

sm_res     = next(r for r in screening_results
                  if {r["A"], r["B"]} == {"S", "SM"})
crush_sp   = pd.Series(sm_res["spread"], index=sm_res["spread_idx"])
crush_roll = crush_sp.rolling(63).mean()
idx_c      = sm_res["spread_idx"]

print(f"\n  S-SM Spread Statistics:")
print(f"    Mean        : {crush_sp.mean():.4f}")
print(f"    Std         : {crush_sp.std():.4f}")
print(f"    Hurst       : {sm_res['hurst']:.4f}")
print(f"    Half-life   : {sm_res['half_life']:.1f} trading days")
print(f"    EG p-value  : {sm_res['eg_p']:.4f}")
print(f"    Hedge ratio : {sm_res['beta']:.4f}")
print(f"    Hedge CV    : {sm_res['hedge_cv']:.4f} ({sm_res['hedge_cv']*100:.1f}%)")
print(f"    Screening   : {'PASS' if sm_res['all_pass'] else 'FAIL'}")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
for key, lbl in [("S","Soybeans"), ("SM","Soybean Meal")]:
    p = dfs[key]["Close"].loc[idx_c]
    ax.plot(idx_c, p / p.iloc[0] * 100,
            color=COLOR_MAP[key], linewidth=0.9, label=lbl)
ax.set_title("Crush Spread Components: Soybeans vs. Soybean Meal (Normalised)")
ax.set_ylabel("Index (Base=100)")
ax.legend()

ax = axes[1]
mean_sp, std_sp = crush_sp.mean(), crush_sp.std()
ax.fill_between(crush_sp.index,
                mean_sp + 2*std_sp, mean_sp - 2*std_sp,
                alpha=0.10, color=COLS5[4], label="+/- 2SD")
ax.fill_between(crush_sp.index,
                mean_sp + std_sp, mean_sp - std_sp,
                alpha=0.18, color=COLS5[4], label="+/- 1SD")
ax.plot(crush_sp.index, crush_sp,  color=COLS5[4], linewidth=0.7,
        alpha=0.75, label="Spread")
ax.plot(crush_roll.index, crush_roll, "k-", linewidth=1.2, label="63-day MA")
ax.axhline(mean_sp, color="red", linestyle="--", linewidth=1.0, label="Mean")
ax.set_title(f"S-SM Cointegrated Spread  "
             f"(HL={sm_res['half_life']:.1f}d, H={sm_res['hurst']:.3f})")
ax.set_ylabel("Spread")
ax.legend(fontsize=8, loc="upper right")

ax = axes[2]
ret_s  = dfs["S"]["LogReturn"].reindex(idx_c).dropna()
ret_sm = dfs["SM"]["LogReturn"].reindex(idx_c).dropna()
cm_idx = ret_s.index.intersection(ret_sm.index)
ax.scatter(ret_s.loc[cm_idx], ret_sm.loc[cm_idx],
           s=1.5, alpha=0.25, color=COLS5[4])
r_val = ret_s.loc[cm_idx].corr(ret_sm.loc[cm_idx])
xl    = np.linspace(ret_s.loc[cm_idx].min(), ret_s.loc[cm_idx].max(), 200)
fit   = np.polyfit(ret_s.loc[cm_idx], ret_sm.loc[cm_idx], 1)
ax.plot(xl, np.polyval(fit, xl), "r-", linewidth=1.2)
ax.set_title(f"S vs SM: Daily Log Return Scatter  (r = {r_val:.4f})")
ax.set_xlabel("Soybeans Log Return")
ax.set_ylabel("Soybean Meal Log Return")

fig.suptitle("Crush Spread Deep Dive: Soybeans (S) vs. Soybean Meal (SM)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("figC1_crush_spread.png")

# =========================================================================
# PART D  CALENDAR SPREAD COMPARISON
# =========================================================================
print("\n" + "=" * 70)
print("  PART D: CALENDAR SPREAD COMPARISON  (CL1-CL2 as contrast)")
print("=" * 70)

cal_idx = dfs["CL"].index.intersection(raw_cl2.index)
lp_cl1  = dfs["CL"]["LogPrice"].reindex(cal_idx).dropna()
lp_cl2  = raw_cl2["LogPrice"].reindex(cal_idx).dropna()
cal_idx  = lp_cl1.index.intersection(lp_cl2.index)

cal_res = screen_pair(lp_cl1.loc[cal_idx].values,
                      lp_cl2.loc[cal_idx].values)
cal_sp  = pd.Series(cal_res["spread"], index=cal_idx)
cal_ret_corr = (dfs["CL"]["LogReturn"].reindex(cal_idx)
                .corr(raw_cl2["LogReturn"].reindex(cal_idx)))

print(f"\n  CL1-CL2 Calendar Spread (for comparison):")
print(f"    Log Return Correlation : {cal_ret_corr:.4f}")
print(f"    EG p-value             : {cal_res['eg_p']:.6f}")
print(f"    OU half-life           : {cal_res['half_life']:.2f} trading days")
print(f"    Hurst exponent         : {cal_res['hurst']:.4f}")
print(f"    Hedge-ratio CV         : {cal_res['hedge_cv']:.4f} ({cal_res['hedge_cv']*100:.1f}%)")

print(f"\n  Return correlation comparison:")
print(f"    CL1-CL2  (calendar)  : r = {cal_ret_corr:.4f}  <- extremely high")
for a, b, r, rel in sorted(pair_corrs, key=lambda x: -x[2]):
    print(f"    {a}-{b:<5} ({rel:<5}): r = {r:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: return correlation bar chart
ax = axes[0]
all_corrs = ([(f"CL1-CL2\n(calendar)", cal_ret_corr, "gray")] +
             [(f"{a}-{b}", r, COLOR_MAP[a])
              for a, b, r, _ in sorted(pair_corrs, key=lambda x: x[2],
                                        reverse=True)])
bars = ax.barh([x[0] for x in all_corrs],
               [x[1] for x in all_corrs],
               color=[x[2] for x in all_corrs], alpha=0.75, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Log Return Correlation")
ax.set_title("Return Correlation: Calendar vs. Inter-Commodity Pairs")
ax.set_xlim(-0.3, 1.1)
for bar, (_, v, _) in zip(bars, all_corrs):
    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=8)

# Right: standardised spread comparison
ax = axes[1]
best_r = survivors[0] if survivors else sm_res
best_sp = pd.Series(best_r["spread"], index=best_r["spread_idx"])

def _std(s):
    return (s - s.mean()) / s.std()

ax.plot(cal_idx, _std(cal_sp),
        color="gray", linewidth=0.75,
        label=f"CL1-CL2 Calendar (HL={cal_res['half_life']:.1f}d)")
ax.plot(best_r["spread_idx"], _std(best_sp),
        color=COLOR_MAP[best_r["A"]], linewidth=0.75, alpha=0.85,
        label=f"{best_r['Pair']} (HL={best_r['half_life']:.1f}d)")
ax.set_title("Spread Comparison (Standardised): Calendar vs. Best Inter-Commodity")
ax.set_ylabel("Standardised Spread")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
plt.setp(ax.get_xticklabels(), rotation=30)

fig.suptitle("Calendar Spread vs. Inter-Commodity Spread: Key Differences",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save("figD1_calendar_vs_intercommodity.png")

# =========================================================================
# EXPORT
# =========================================================================
res_export = pd.DataFrame(
    [{k: v for k, v in r.items()
      if k not in ("spread", "spread_idx")}
     for r in screening_results]
).round(4)

with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "analysis_summary.xlsx")) as writer:
    for key in POOL:
        df = dfs[key]
        for field in ["Close", "LogReturn"]:
            pass
        df[["Close","LogReturn"]].describe().round(6).to_excel(
            writer, sheet_name=f"Stats_{key}")
    res_export.to_excel(writer, sheet_name="PairScreening", index=False)
    corr_return.round(4).to_excel(writer, sheet_name="ReturnCorr")

print("\n[saved] analysis_summary.xlsx")

# =========================================================================
print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 70)

_log.close()
sys.stdout = sys.__stdout__
print(f"\nDone. Output -> {OUTPUT_DIR}")
