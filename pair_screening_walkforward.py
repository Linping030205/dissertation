"""
pair_screening_walkforward.py
==============================
Walk-forward rolling-window pair screening for the inter-commodity
spread trading research.

At each evaluation date we take the previous FORMATION_WINDOW trading
days of log-price data and run all four screening criteria on every
candidate pair.  Results are aggregated to show how the viability of
each pair evolves over time.

Key parameters
--------------
FORMATION_WINDOW : 252  (1 calendar year of trading days)
STEP_SIZE        : 21   (re-evaluate monthly; set to 63 for quarterly)
TOP_K            : 3    (max pairs selected at any given date)

Output
------
  figWF1_rolling_halflife.png      -- rolling half-life for each pair
  figWF2_rolling_hurst.png         -- rolling Hurst exponent
  figWF3_screening_heatmap.png     -- criteria-pass heatmap (pair x time)
  figWF4_pass_counts.png           -- how many pairs pass each criterion
  figWF5_selected_pairs.png        -- which pairs are selected (top-k)
  walkforward_results.xlsx         -- full numeric results
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commodity_utils import (
    DATA_FILES, FULL_NAMES, SECTORS, POOL, PAIRS,
    TH_EG_PVAL, TH_HALF_LIFE, TH_HURST, TH_HEDGECV,
    load_commodities, screen_pair,
)

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIG
# =========================================================================
DATA_DIR         = str(Path(__file__).resolve().parent)
OUTPUT_DIR       = os.path.join(DATA_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORMATION_WINDOW = 252   # trading days (1 year)
STEP_SIZE        = 21    # re-evaluate every ~1 month  (change to 63 for quarterly)
TOP_K            = 3     # max selected pairs per date

PAIR_LABELS = [f"{a}-{b}" for a, b in PAIRS]
COLS10 = [
    "#1565C0","#1E88E5","#C62828","#E53935",
    "#FF8F00","#2E7D32","#43A047","#6A1B9A","#8E24AA","#BF360C",
]
COLOR_PAIR = dict(zip(PAIR_LABELS, COLS10))


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


_log = open(os.path.join(OUTPUT_DIR, "walkforward_report.txt"), "w", encoding="utf-8")
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
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =========================================================================
# LOAD & ALIGN DATA
# =========================================================================
print("=" * 70)
print("  WALK-FORWARD PAIR SCREENING")
print(f"  Formation window : {FORMATION_WINDOW} trading days")
print(f"  Step size        : {STEP_SIZE} trading days (re-evaluate monthly)")
print(f"  Top-k selection  : {TOP_K}")
print("=" * 70)

dfs, common_idx, _ = load_commodities(DATA_DIR)

# Build aligned log-price matrix (5 columns)
log_price_mat = pd.DataFrame(
    {k: dfs[k]["LogPrice"] for k in POOL}
).dropna()

# Evaluation dates: end of each formation window, stepping forward
eval_positions = range(FORMATION_WINDOW, len(log_price_mat), STEP_SIZE)
eval_dates     = log_price_mat.index[list(eval_positions)]

print(f"\nTotal aligned trading days  : {len(log_price_mat)}")
print(f"Evaluation points           : {len(eval_dates)}")
print(f"First evaluation date       : {eval_dates[0].date()}")
print(f"Last  evaluation date       : {eval_dates[-1].date()}\n")

# =========================================================================
# WALK-FORWARD LOOP
# =========================================================================
print("Running walk-forward screening...")
records = []

for eval_date in eval_dates:
    t_pos        = log_price_mat.index.get_loc(eval_date)
    window_start = t_pos - FORMATION_WINDOW
    window       = log_price_mat.iloc[window_start:t_pos]

    for a, b in PAIRS:
        pair_label = f"{a}-{b}"
        res = screen_pair(window[a].values, window[b].values)
        records.append({
            "eval_date":  eval_date,
            "pair":       pair_label,
            "A": a, "B": b,
            "relation":   "Intra" if SECTORS[a]==SECTORS[b] else "Inter",
            "eg_p":       res["eg_p"],
            "half_life":  res["half_life"],
            "hurst":      res["hurst"],
            "hedge_cv":   res["hedge_cv"],
            "pass_eg":    int(res["pass_eg"]),
            "pass_hl":    int(res["pass_hl"]),
            "pass_h":     int(res["pass_h"]),
            "pass_cv":    int(res["pass_cv"]),
            "all_pass":   int(res["all_pass"]),
            "n_pass":     sum([res["pass_eg"], res["pass_hl"],
                               res["pass_cv"]]),   # 3-criterion count; Hurst excluded
        })

df_wf = pd.DataFrame(records)
print(f"Done.  {len(df_wf)} records ({len(eval_dates)} dates x {len(PAIRS)} pairs)\n")

# =========================================================================
# SUMMARY STATISTICS
# =========================================================================
print("=" * 70)
print("  WALK-FORWARD SUMMARY STATISTICS")
print("=" * 70)

print(f"\nOverall criterion pass rates ({len(eval_dates)} evaluation dates):")
print(f"  {'Pair':<10} {'EG%':>6} {'HL%':>6} {'H%':>6} {'CV%':>6} {'ALL%':>6}")
print("  " + "-" * 40)
for pair_lbl in PAIR_LABELS:
    sub = df_wf[df_wf["pair"] == pair_lbl]
    eg  = sub["pass_eg"].mean() * 100
    hl  = sub["pass_hl"].mean() * 100
    h   = sub["pass_h"].mean()  * 100
    cv  = sub["pass_cv"].mean() * 100
    ap  = sub["all_pass"].mean()* 100
    print(f"  {pair_lbl:<10} {eg:>6.1f} {hl:>6.1f} {h:>6.1f} {cv:>6.1f} {ap:>6.1f}")

# Dates when at least one pair passes all criteria
dates_with_pass = (df_wf[df_wf["all_pass"]==1]
                   .groupby("eval_date")["pair"].apply(list))
pass_pct = len(dates_with_pass) / len(eval_dates) * 100
print(f"\nDates with >= 1 pair passing all criteria: "
      f"{len(dates_with_pass)} / {len(eval_dates)} ({pass_pct:.1f}%)")

if len(dates_with_pass) > 0:
    print("\nSample of dates with qualifying pairs:")
    for dt, pairs in list(dates_with_pass.items())[:10]:
        print(f"  {dt.date()}  ->  {pairs}")

# Top-k selection history
print(f"\nTop-{TOP_K} selection history (dates with all_pass > 0):")
selected_history = []
for eval_date in eval_dates:
    day_res = df_wf[df_wf["eval_date"] == eval_date]
    passers = day_res[day_res["all_pass"] == 1].copy()
    if len(passers) == 0:
        selected_history.append({"eval_date": eval_date, "selected": []})
        continue
    # Sort by half_life ascending; take top-k
    passers = passers.sort_values("half_life")
    top     = passers.head(TOP_K)
    selected_history.append({
        "eval_date": eval_date,
        "selected":  top["pair"].tolist(),
        "half_lives": top["half_life"].tolist(),
    })

n_selected = sum(1 for s in selected_history if len(s["selected"]) > 0)
print(f"  {n_selected} / {len(eval_dates)} dates have at least one selected pair")

# =========================================================================
# figWF1  ROLLING HALF-LIFE
# =========================================================================
fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
fig.suptitle(f"Walk-Forward Rolling Half-Life ({FORMATION_WINDOW}-day window)\n"
             f"Red dashed line = threshold ({TH_HALF_LIFE} trading days)",
             fontsize=12, fontweight="bold")

for idx_p, (pair_lbl, ax) in enumerate(zip(PAIR_LABELS, axes.flatten())):
    sub = df_wf[df_wf["pair"] == pair_lbl].set_index("eval_date")
    hl  = sub["half_life"].replace(np.inf, np.nan)
    ax.plot(hl.index, hl, color=COLOR_PAIR[pair_lbl], linewidth=0.85)
    ax.axhline(TH_HALF_LIFE, color="red", linestyle="--", linewidth=0.9)
    pct = (hl < TH_HALF_LIFE).mean() * 100
    ax.set_title(f"{pair_lbl}  (below threshold {pct:.1f}% of time)",
                 fontsize=9)
    ax.set_ylabel("Half-life (days)")
    ax.set_ylim(0, min(hl.quantile(0.98) * 1.2, 400)
                if hl.notna().any() else 200)

for ax in axes[-1]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.get_xticklabels(), rotation=30)

plt.tight_layout()
_save("figWF1_rolling_halflife.png")

# =========================================================================
# figWF2  ROLLING HURST EXPONENT
# =========================================================================
fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
fig.suptitle(f"Walk-Forward Rolling Hurst Exponent ({FORMATION_WINDOW}-day window)\n"
             f"[Reference only: R/S estimator has severe upward bias at n=252; H<0.5 is "
             f"excluded from hard screening and is shown here for information only]",
             fontsize=12, fontweight="bold")

for pair_lbl, ax in zip(PAIR_LABELS, axes.flatten()):
    sub = df_wf[df_wf["pair"] == pair_lbl].set_index("eval_date")
    H   = sub["hurst"]
    ax.plot(H.index, H, color=COLOR_PAIR[pair_lbl], linewidth=0.85)
    ax.axhline(TH_HURST, color="red", linestyle="--", linewidth=0.9)
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.7, alpha=0.7)
    pct = (H < TH_HURST).mean() * 100
    ax.set_title(f"{pair_lbl}  (below 0.5 in {pct:.1f}% of windows)", fontsize=9)
    ax.set_ylabel("Hurst H")
    ax.set_ylim(0.3, 1.1)

for ax in axes[-1]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.setp(ax.get_xticklabels(), rotation=30)

plt.tight_layout()
_save("figWF2_rolling_hurst.png")

# =========================================================================
# figWF3  SCREENING HEATMAP  (pair × time, colour = criteria passed)
# =========================================================================
# Pivot: rows = pairs, columns = eval_date, value = n_pass (0-4)
pivot = df_wf.pivot(index="pair", columns="eval_date", values="n_pass")
pivot = pivot.reindex(PAIR_LABELS)

# Downsample columns for readability (take every 3rd date for display)
display_dates = pivot.columns[::3]
pivot_disp    = pivot[display_dates]

fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(pivot_disp.values, aspect="auto",
               cmap="RdYlGn", vmin=0, vmax=3, interpolation="nearest")

ax.set_yticks(range(len(PAIR_LABELS)))
ax.set_yticklabels(PAIR_LABELS, fontsize=9)

n_ticks = min(12, len(display_dates))
tick_pos = np.linspace(0, len(display_dates)-1, n_ticks, dtype=int)
ax.set_xticks(tick_pos)
ax.set_xticklabels([display_dates[i].strftime("%Y-%m") for i in tick_pos],
                   rotation=45, ha="right", fontsize=8)

cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
cbar.set_label("Active Criteria Passed (0-3)", fontsize=9)
cbar.set_ticks([0, 1, 2, 3])

ax.set_title(
    f"Walk-Forward Screening: Active Criteria Passed Over Time\n"
    f"(Green=3=PASS [EG+HL+CV], Red=0=fail  |  {FORMATION_WINDOW}-day window, "
    f"step={STEP_SIZE} days  |  Hurst excluded: R/S bias at n=252)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
_save("figWF3_screening_heatmap.png")

# =========================================================================
# figWF4  HOW MANY PAIRS PASS EACH CRITERION OVER TIME
# =========================================================================
criteria_cols = ["pass_eg", "pass_hl", "pass_cv", "all_pass"]
criteria_lbls = ["Cointegration\n(EG p<0.05)", "Half-Life\n(<40d)",
                 "Hedge CV\n(<50%)", "All 3\nCriteria"]
crit_colors   = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]

# Count pairs passing each criterion per date
pass_counts = (df_wf.groupby("eval_date")[criteria_cols].sum()
               .rolling(3).mean())   # light smoothing

fig, ax = plt.subplots(figsize=(14, 6))
for col, lbl, col_c in zip(criteria_cols, criteria_lbls, crit_colors):
    ax.plot(pass_counts.index, pass_counts[col],
            label=lbl.replace("\n", " "), color=col_c, linewidth=1.0)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_title(f"Number of Pairs (out of 10) Passing Each Active Criterion Over Time\n"
             f"(3-point smoothed, {FORMATION_WINDOW}-day formation window  |  "
             f"Hurst excluded: reported in figWF2 only)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Number of Pairs Passing")
ax.set_ylim(-0.2, 10.5)
ax.legend(loc="upper right", ncol=5, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
_save("figWF4_pass_counts.png")

# =========================================================================
# figWF5  WHICH PAIRS ARE SELECTED (top-k) OVER TIME
# =========================================================================
# Build a DataFrame: rows = eval_dates, columns = pair_labels,
# value = 1 if this pair is selected (top-k), 0 otherwise
select_mat = pd.DataFrame(0, index=eval_dates, columns=PAIR_LABELS)

for entry in selected_history:
    for pair_lbl in entry["selected"]:
        select_mat.loc[entry["eval_date"], pair_lbl] = 1

fig, ax = plt.subplots(figsize=(14, 5))
for pair_lbl in PAIR_LABELS:
    series = select_mat[pair_lbl]
    if series.sum() > 0:   # only plot pairs that were ever selected
        # Show as dots where selected = 1
        sel_dates = series[series == 1].index
        ax.scatter(sel_dates,
                   [PAIR_LABELS.index(pair_lbl)] * len(sel_dates),
                   color=COLOR_PAIR[pair_lbl], s=20, marker="s",
                   label=pair_lbl, zorder=5)

ax.set_yticks(range(len(PAIR_LABELS)))
ax.set_yticklabels(PAIR_LABELS, fontsize=9)
ax.set_title(
    f"Walk-Forward Top-{TOP_K} Selection History\n"
    f"(Each square = this pair was selected at that evaluation date)",
    fontsize=11, fontweight="bold"
)
ax.set_xlabel("Evaluation Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax.get_xticklabels(), rotation=30)
if select_mat.sum().sum() > 0:
    ax.legend(loc="upper right", fontsize=8, ncol=3)
else:
    ax.text(0.5, 0.5, "No pairs selected in any window.\n"
            "Consider relaxing thresholds or shortening formation window.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="red")
plt.tight_layout()
_save("figWF5_selected_pairs.png")

# =========================================================================
# EXPORT RESULTS TO EXCEL
# =========================================================================
with pd.ExcelWriter(
    os.path.join(OUTPUT_DIR, "walkforward_results.xlsx")
) as writer:
    # Full record table
    df_wf.to_excel(writer, sheet_name="AllRecords", index=False)

    # Per-pair summary: pass rates across time
    summary_rows = []
    for pair_lbl in PAIR_LABELS:
        sub = df_wf[df_wf["pair"] == pair_lbl]
        a, b = pair_lbl.split("-")
        summary_rows.append({
            "Pair":          pair_lbl,
            "A":             sub["A"].iloc[0],
            "B":             sub["B"].iloc[0],
            "Relation":      sub["relation"].iloc[0],
            "EG_pass_%":     sub["pass_eg"].mean() * 100,
            "HL_pass_%":     sub["pass_hl"].mean() * 100,
            "Hurst_pass_%":  sub["pass_h"].mean()  * 100,
            "CV_pass_%":     sub["pass_cv"].mean()  * 100,
            "All_pass_%":    sub["all_pass"].mean() * 100,
            "Median_HL":     sub["half_life"].replace(np.inf, np.nan).median(),
            "Median_Hurst":  sub["hurst"].median(),
            "Median_CV":     sub["hedge_cv"].replace(np.inf, np.nan).median(),
        })
    pd.DataFrame(summary_rows).round(2).to_excel(
        writer, sheet_name="PairSummary", index=False)

    # Selection history
    sel_rows = []
    for entry in selected_history:
        sel_rows.append({
            "eval_date": entry["eval_date"],
            "n_selected": len(entry["selected"]),
            "selected_pairs": ", ".join(entry["selected"]) or "none",
        })
    pd.DataFrame(sel_rows).to_excel(
        writer, sheet_name="SelectionHistory", index=False)

print("\n[saved] walkforward_results.xlsx")

# =========================================================================
print("\n" + "=" * 70)
print("  WALK-FORWARD SCREENING COMPLETE")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 70)

_log.close()
sys.stdout = sys.__stdout__
print(f"\nDone. Output -> {OUTPUT_DIR}")
