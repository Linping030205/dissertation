"""
evaluate.py
===========
Out-of-sample evaluation and ablation matrix for the S-SM RL strategy.

Usage
-----
    python evaluate.py                   # evaluate default CFG checkpoint
    python evaluate.py --ablation        # run full 5-variant ablation matrix

Outputs (in analysis_output/evaluation/)
    equity_curves.png
    metrics_summary.xlsx
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from rl_env   import CrushSpreadEnv, FEATURES_PATH, OUTPUT_DIR, TRAIN_END
from rl_agent import make_agent
from train    import CFG as DEFAULT_CFG, _run_name, _eval_episode, train
from commodity_utils import spread_trade_cost, file_sha256

EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Serif", "font.size": 10,
})


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(pnls: np.ndarray, label: str = "",
                    positions: np.ndarray = None,
                    path_dependent: bool = True) -> dict:
    """
    Annualised performance metrics from a daily P&L array.
    Assumes 252 trading days per year.

    Units: `pnls` is position-weighted SpreadReturn (log-return units on a
    one-unit position), not a percentage-of-capital portfolio return -- no
    margin/capital base is defined in this dataset. CumReturn/AnnReturn/
    MaxDrawdown are therefore cumulative/annualised SUMS of this P&L series
    (additive, not compounded), and should be read as such rather than as
    standard capital-normalised portfolio return percentages.
    """
    r = np.array(pnls, dtype=np.float64)
    n = len(r)
    if n == 0:
        return {}

    cum_ret      = float(r.sum())
    ann_ret      = float(r.mean() * 252)
    ann_vol      = float(r.std() * np.sqrt(252))
    sharpe       = float(r.mean() / (r.std() + 1e-10) * np.sqrt(252))

    # Standard downside deviation: dispersion below the MAR (=0), taken over
    # ALL N observations (not just the negative subset, which would use the
    # wrong denominator and centre on the negative-subset mean instead of 0).
    downside_dev = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    sortino      = float(r.mean() / (downside_dev + 1e-10) * np.sqrt(252)) if downside_dev > 0 else np.inf

    max_dd = None
    calmar = None
    if path_dependent:
        # Include the initial zero-P&L equity point.  Without E_0 = 0, an
        # immediate loss is incorrectly treated as the first running peak and
        # the opening drawdown is understated.
        equity       = np.r_[0.0, np.cumsum(r)]
        rolling_max  = np.maximum.accumulate(equity)
        drawdowns    = equity - rolling_max
        max_dd       = float(drawdowns.min())
        calmar       = (float(ann_ret / (-max_dd + 1e-10))
                        if max_dd < 0 else np.inf)

    # Number of days the discrete position changed value. NOT a count of
    # complete round-trip trades: a direct reversal (-1 -> +1) counts once
    # here even though it economically closes one leg and opens another.
    n_position_changes = None
    if path_dependent and positions is not None:
        p = np.asarray(positions)
        if len(p) == n:
            n_position_changes = int(np.sum(np.diff(np.r_[0, p]) != 0))
    pct_positive = float((r > 0).mean() * 100)

    return {
        "Label":        label,
        "CumReturn":    round(cum_ret, 6),
        "AnnReturn":    round(ann_ret, 6),
        "AnnVol":       round(ann_vol, 6),
        "Sharpe":       round(sharpe, 4),
        "Sortino":      round(sortino, 4),
        "MaxDrawdown":  None if max_dd is None else round(max_dd, 6),
        "Calmar":       None if calmar is None else round(calmar, 4),
        "PctPositive":  round(pct_positive, 2),
        "NPositionChanges": n_position_changes,
        "N_days":       n,
    }


def stability_tertile_bounds(features_path: str = FEATURES_PATH,
                             train_end: str = TRAIN_END) -> tuple:
    """
    33rd/67th percentile of RegimeStabilityScore on the TRAINING period
    only (no test-period information used to define the cut points) --
    i.e. training-anchored stability bands, not a tertile split of the
    test period itself. The binary RegimeStable split leaves only ~17
    "Stable" days in the test period (not enough for any meaningful
    per-group statistic); these bands are a somewhat better alternative
    (the continuous score has real spread throughout the test period),
    but because the test period's stability distribution sits mostly
    below the training period's (see spread_features_report.txt), the
    resulting test-period group sizes are NOT balanced -- typically
    Low >> Mid > High, and High may still be a small-sample group.
    """
    data = pd.read_excel(features_path, sheet_name="Features",
                         index_col=0, parse_dates=True)
    train_scores = data.loc[:train_end, "RegimeStabilityScore"].dropna()
    if train_scores.empty:
        raise ValueError(
            f"No RegimeStabilityScore observations on or before {train_end}."
        )
    return (float(np.percentile(train_scores, 33)),
            float(np.percentile(train_scores, 67)))


def regime_split_metrics(pnls: np.ndarray,
                         regimes: np.ndarray,
                         dates: np.ndarray,
                         positions: np.ndarray = None,
                         label: str = "",
                         stability_scores: np.ndarray = None,
                         stability_bounds: tuple = None) -> dict:
    """Metrics by indicator regime, continuous-stability tertile, and the
    fixed 2021-2025 calendar episode."""
    dates = pd.DatetimeIndex(dates)
    positions = None if positions is None else np.asarray(positions)
    stable_mask   = regimes == 1
    unstable_mask = regimes == 0
    calendar_mask = ((dates >= pd.Timestamp("2021-01-01")) &
                     (dates <= pd.Timestamp("2025-12-31")))

    def _pos(mask=None):
        if positions is None:
            return None
        return positions if mask is None else positions[mask]

    m  = compute_metrics(pnls, label, _pos())
    # Stable/Unstable masks generally select non-contiguous dates. Path-dependent
    # metrics on the compressed subsequences would not describe a real equity
    # path, so MaxDrawdown, Calmar and NPositionChanges are intentionally left blank.
    ms = compute_metrics(pnls[stable_mask], label + " [Stable]",
                         _pos(stable_mask), path_dependent=False)
    mu = compute_metrics(pnls[unstable_mask], label + " [Unstable]",
                         _pos(unstable_mask), path_dependent=False)
    mc = compute_metrics(pnls[calendar_mask], label + " [2021-2025]",
                         _pos(calendar_mask))
    result = {"overall": m, "stable": ms, "unstable": mu,
              "calendar_2021_2025": mc}

    if stability_scores is not None and stability_bounds is not None:
        lo_cut, hi_cut = stability_bounds
        low_mask  = stability_scores <= lo_cut
        mid_mask  = (stability_scores > lo_cut) & (stability_scores <= hi_cut)
        high_mask = stability_scores > hi_cut
        result["low_stability"] = compute_metrics(
            pnls[low_mask], label + " [LowStability]", _pos(low_mask), path_dependent=False)
        result["mid_stability"] = compute_metrics(
            pnls[mid_mask], label + " [MidStability]", _pos(mid_mask), path_dependent=False)
        result["high_stability"] = compute_metrics(
            pnls[high_mask], label + " [HighStability]", _pos(high_mask), path_dependent=False)

    return result


# =========================================================================
# EVALUATION OF ONE CONFIG
# =========================================================================

def evaluate_config(cfg: dict, ckpt_path: str = None) -> dict:
    """
    Load checkpoint and run full test episode.
    Returns dict with metrics and per-day info.
    """
    reward_mode = "regime" if cfg["regime_in_reward"] else "base"
    env_kwargs  = dict(
        features_path   = cfg["features_path"],
        lookback        = cfg["lookback"],
        episode_len     = cfg["episode_len"],
        regime_in_state = cfg["regime_in_state"],
        reward_mode     = reward_mode,
        lambda_cost     = cfg["lambda_cost"],
        penalty_factor  = cfg["penalty_factor"],
        cost_factor     = cfg["cost_factor"],
        train_end       = cfg.get("train_end", "2019-12-31"),
        val_start       = cfg.get("val_start", "2020-01-01"),
        val_end         = cfg.get("val_end", "2020-12-31"),
        test_start      = cfg.get("test_start", "2021-01-01"),
        test_end        = cfg.get("test_end"),
    )
    test_env = CrushSpreadEnv(split="test", **env_kwargs)
    obs_dim  = test_env.obs_dim

    agent = make_agent(cfg["agent_type"], obs_dim, cfg)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'. Refusing to silently evaluate "
            f"random weights -- train the model first or pass a valid ckpt_path."
        )
    agent.load(ckpt_path)
    print(f"  Loaded: {ckpt_path}")

    # ---- run test episode ----
    obs, _ = test_env.reset()
    pnls, gross_pnls, costs = [], [], []
    regimes, stability_scores, signal_dates, return_dates, positions = [], [], [], [], []
    done = False
    while not done:
        action = agent.act(obs, epsilon=0.0, deterministic=True)
        obs, _, done, _, info = test_env.step(action)
        pnls.append(info["net_pnl"])
        gross_pnls.append(info["gross_pnl"])
        costs.append(info["transaction_cost"])
        regimes.append(info["regime"])
        stability_scores.append(info["stability_score"])
        signal_dates.append(info["signal_date"])
        return_dates.append(info["return_date"])
        positions.append(info["position"])

    pnls    = np.array(pnls,    dtype=np.float64)
    regimes = np.array(regimes, dtype=np.float64)
    stability_scores = np.array(stability_scores, dtype=np.float64)
    signal_dates = pd.to_datetime(signal_dates)
    return_dates = pd.to_datetime(return_dates)
    positions = np.array(positions)
    split_m = regime_split_metrics(
        pnls, regimes, return_dates, positions, label=_run_name(cfg),
        stability_scores=stability_scores,
        stability_bounds=stability_tertile_bounds(
            cfg["features_path"], cfg.get("train_end", TRAIN_END)
        ),
    )

    return {
        "cfg":       cfg,
        "pnls":      pnls,
        "gross_pnls": np.asarray(gross_pnls, dtype=np.float64),
        "costs":      np.asarray(costs, dtype=np.float64),
        "regimes":   regimes,
        "dates":     return_dates,
        "signal_dates": signal_dates,
        "return_dates": return_dates,
        "positions": positions,
        "metrics":   split_m,
        "audit": {
            "pnl_basis": "net",
            "signal_first": str(signal_dates[0].date()),
            "signal_last": str(signal_dates[-1].date()),
            "return_first": str(return_dates[0].date()),
            "return_last": str(return_dates[-1].date()),
            "n_observations": len(pnls),
        },
    }


# =========================================================================
# BASELINE: Z-SCORE RULE
# =========================================================================

def _train_only_entry_threshold(features_path: str = FEATURES_PATH,
                                 pct: float = 95.0) -> float:
    """
    Empirical |Z|-score entry threshold, computed ONLY from the training
    period (<= TRAIN_END). Using the full-sample percentile (as before)
    would leak information from the 2021+ test period into a baseline
    parameter that is then evaluated on that same test period.
    """
    data = pd.read_excel(features_path, sheet_name="Features",
                         index_col=0, parse_dates=True)
    train_z = data.loc[:TRAIN_END, "ZScore20"].dropna()
    return float(np.percentile(np.abs(train_z), pct))


def zscore_baseline(features_path: str = FEATURES_PATH,
                    entry: float = None,
                    exit_: float = 0.5,
                    lambda_cost: float = 0.0005) -> dict:
    """
    Classic pairs trading rule:
      |Z| > entry  → open position (short spread if Z > +entry, long if Z < -entry)
      |Z| < exit_  → close position

    `entry` defaults to the empirical 95th percentile of |Z| on the
    TRAINING period only (see _train_only_entry_threshold) -- no test-period
    information is used to pick this parameter.
    """
    if entry is None:
        entry = _train_only_entry_threshold(features_path)
        print(f"  [zscore_baseline] train-only entry threshold = {entry:.4f}")

    data = pd.read_excel(features_path, sheet_name="Features",
                         index_col=0, parse_dates=True)
    data = data.loc["2021-01-01":]   # test period only

    z       = data["ZScore20"].values
    regime  = data["RegimeStable"].values
    stab    = data["RegimeStabilityScore"].values
    beta    = data["RollingBeta"].values
    dates   = data.index

    pos   = 0
    pnls  = []
    gross_pnls = []
    costs = []
    regs  = []
    stabs = []
    signal_dts = []
    return_dts = []
    positions = []
    prev_pos  = 0
    last_i    = None

    for i in range(len(z) - 1):
        if np.isnan(z[i]):
            continue
        # Entry
        if pos == 0:
            if z[i] > entry:
                pos = -1   # short spread
            elif z[i] < -entry:
                pos = 1    # long spread
        # Exit
        elif abs(z[i]) < exit_:
            pos = 0

        sr_next = float(data["SpreadReturn"].iloc[i + 1])
        if np.isnan(sr_next):
            sr_next = 0.0
        gross_pnl = pos * sr_next
        # Same cost model as the RL environment (see rl_env.py / spread_trade_cost):
        # priced on both legs, and charges for hedge-ratio rebalancing even
        # when the discrete position is unchanged.
        beta_t    = beta[i]
        beta_prev = beta[i - 1] if i > 0 else beta_t
        cost = spread_trade_cost(lambda_cost, pos, prev_pos, beta_t, beta_prev)
        pnls.append(gross_pnl - cost)
        gross_pnls.append(gross_pnl)
        costs.append(cost)
        regs.append(regime[i])
        stabs.append(stab[i])
        signal_dts.append(dates[i])
        return_dts.append(dates[i + 1])
        positions.append(pos)
        prev_pos = pos
        last_i   = i

    # Forced close-out: unwind any open position at the end of the test
    # window, same as the RL environment does at episode/test termination.
    if last_i is not None and pos != 0:
        beta_last = beta[last_i]
        close_cost = spread_trade_cost(lambda_cost, 0.0, pos, beta_last, beta_last)
        pnls[-1]  -= close_cost
        costs[-1] += close_cost

    pnls  = np.array(pnls)
    regs  = np.array(regs)
    stabs = np.array(stabs)
    signal_dts = pd.to_datetime(signal_dts)
    return_dts = pd.to_datetime(return_dts)
    positions = np.array(positions)
    split_m = regime_split_metrics(
        pnls, regs, return_dts, positions, "Z-score Rule",
        stability_scores=stabs,
        stability_bounds=stability_tertile_bounds(features_path),
    )
    return {"pnls": pnls, "regimes": regs, "dates": return_dts,
            "signal_dates": signal_dts, "return_dates": return_dts,
            "positions": positions,
            "gross_pnls": np.asarray(gross_pnls),
            "costs": np.asarray(costs),
            "metrics": split_m, "label": "Z-score Rule (baseline)",
            "audit": {
                "pnl_basis": "net",
                "signal_first": str(signal_dts[0].date()),
                "signal_last": str(signal_dts[-1].date()),
                "return_first": str(return_dts[0].date()),
                "return_last": str(return_dts[-1].date()),
                "n_observations": len(pnls),
            }}


# =========================================================================
# ABLATION MATRIX
# =========================================================================

ABLATION_VARIANTS = [
    # (label, regime_in_state, regime_in_reward, agent_type)
    ("Full IQN",              True,  True,  "IQN"),
    ("w/o Regime-in-State",   False, True,  "IQN"),
    ("w/o Regime-in-Reward",  True,  False, "IQN"),
    ("No Explicit Regime IQN", False, False, "IQN"),
    ("Parameter-Matched Double DQN", True, True, "DQN"),
]


def _checkpoint_is_stale(run_dir: str, cfg: dict) -> bool:
    """
    True if run_dir has no config.json, its saved config doesn't match
    `cfg`, or the features file has been regenerated (different content)
    since training. _run_name() only encodes agent_type/regime
    flags/pipeline_version -- it does NOT encode seed, lr, cost params,
    etc. -- so two different configs can map to the same run directory.
    A plain cfg-dict comparison also can't see a features file that was
    regenerated in place (same path, different content, e.g. after a
    feature-construction fix) -- hence the separate SHA-256 check.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return True
    with open(cfg_path) as f:
        saved = json.load(f)
    for k, v in cfg.items():
        if k in ("output_dir",):   # path, not a modelling hyperparameter
            continue
        if saved.get(k) != v:
            return True
    features_path = cfg.get("features_path")
    if features_path and os.path.isfile(features_path):
        if saved.get("features_file_sha256") != file_sha256(features_path):
            return True
    return False


def run_ablation(base_cfg: dict, retrain: bool = False) -> list[dict]:
    """
    Train (if retrain=True) and evaluate all ablation variants.
    If checkpoints already exist AND match the current config, load them
    without re-training; otherwise retrain (see _checkpoint_is_stale).
    """
    results = []
    for label, reg_s, reg_r, algo in ABLATION_VARIANTS:
        cfg = dict(base_cfg)
        cfg["regime_in_state"]  = reg_s
        cfg["regime_in_reward"] = reg_r
        cfg["agent_type"]       = algo

        run_dir  = os.path.join(base_cfg["output_dir"],
                                "checkpoints", _run_name(cfg))
        ckpt     = os.path.join(run_dir, "best_model.pt")

        stale = _checkpoint_is_stale(run_dir, cfg)
        if retrain or not os.path.isfile(ckpt) or stale:
            if stale and os.path.isfile(ckpt) and not retrain:
                print(f"  [!] {label}: existing checkpoint's config.json doesn't "
                      f"match the current config -- retraining instead of reusing it")
            print(f"\n{'='*50}\n  Training: {label}\n{'='*50}")
            train(cfg)

        print(f"\n  Evaluating: {label}")
        res = evaluate_config(cfg, ckpt)
        res["label"] = label
        results.append(res)

    return results


# =========================================================================
# PLOTTING & REPORT
# =========================================================================

def plot_equity_curves(results: list[dict],
                       baseline: dict = None,
                       title: str = "Test Set Equity Curves (2021–2026)") -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    colors = ["#1565C0", "#C62828", "#2E7D32", "#FF8F00", "#6A1B9A",
              "#BF360C", "#37474F"]
    lines  = []

    ax = axes[0]
    for res, col in zip(results, colors):
        label   = res.get("label", _run_name(res["cfg"]))
        equity  = np.cumsum(res["pnls"])
        dates   = res["dates"]
        sharpe  = res["metrics"]["overall"]["Sharpe"]
        l, = ax.plot(dates, equity, color=col, linewidth=1.0,
                     label=f"{label}  (Sharpe {sharpe:+.2f})")
        lines.append(l)

    if baseline is not None:
        eq  = np.cumsum(baseline["pnls"])
        sh  = baseline["metrics"]["overall"]["Sharpe"]
        l, = ax.plot(baseline["dates"], eq, color="black",
                     linewidth=1.0, linestyle="--",
                     label=f"{baseline.get('label', 'Z-score Rule')}  "
                           f"(Sharpe {sh:+.2f})")
        lines.append(l)

    ax.axhline(0, color="grey", linewidth=0.5)

    # Shade 2021-2025 breakdown region
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"),
               color="#FFCCCC", alpha=0.25, lw=0, label="Breakdown period")
    ax.set_ylabel("Cumulative Spread Return")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    # RegimeStable bar chart (use first result's regime series)
    ax2 = axes[1]
    r0  = results[0]
    ax2.fill_between(r0["dates"], r0["regimes"], step="post",
                     color="#2E7D32", alpha=0.4, label="RegimeStable")
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("RegimeStable")
    ax2.set_title("Regime Stability (walk-forward screening)", fontsize=9)

    for axi in axes:
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axi.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(axi.get_xticklabels(), rotation=30)

    plt.tight_layout()
    out = os.path.join(EVAL_DIR, "equity_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] equity_curves.png")
    return out


def save_metrics_xlsx(results: list[dict], baseline: dict = None) -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    rows_overall   = []
    rows_stable    = []
    rows_unstable  = []
    rows_calendar  = []
    rows_low_stab  = []
    rows_mid_stab  = []
    rows_high_stab = []
    rows_audit     = []

    all_results = list(results)
    if baseline:
        all_results.append({**baseline,
                            "label": baseline.get("label", "Z-score Rule"),
                            "cfg":   {}})

    for res in all_results:
        label = res.get("label", "?")
        m     = res["metrics"]
        rows_overall.append({"variant": label, **m["overall"]})
        rows_stable.append({"variant": label, **m["stable"]})
        rows_unstable.append({"variant": label, **m["unstable"]})
        rows_calendar.append({"variant": label, **m["calendar_2021_2025"]})
        if "low_stability" in m:
            rows_low_stab.append({"variant": label, **m["low_stability"]})
            rows_mid_stab.append({"variant": label, **m["mid_stability"]})
            rows_high_stab.append({"variant": label, **m["high_stability"]})
        rows_audit.append({"variant": label, **res.get("audit", {})})

    out = os.path.join(EVAL_DIR, "metrics_summary.xlsx")
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame(rows_overall).to_excel(writer,  sheet_name="Overall",  index=False)
        pd.DataFrame(rows_stable).to_excel(writer,   sheet_name="Stable",   index=False)
        pd.DataFrame(rows_unstable).to_excel(writer, sheet_name="Unstable", index=False)
        pd.DataFrame(rows_calendar).to_excel(
            writer, sheet_name="Calendar_2021_2025", index=False)
        if rows_low_stab:
            # Training-anchored stability bands (see stability_tertile_bounds).
            # Not a tertile split of the test period itself -- group sizes can
            # be highly imbalanced under the train/test distribution shift
            # (typically Low >> Mid > High); High may still be small-sample.
            pd.DataFrame(rows_low_stab).to_excel(
                writer, sheet_name="Low_Stability", index=False)
            pd.DataFrame(rows_mid_stab).to_excel(
                writer, sheet_name="Mid_Stability", index=False)
            pd.DataFrame(rows_high_stab).to_excel(
                writer, sheet_name="High_Stability", index=False)
        pd.DataFrame(rows_audit).to_excel(
            writer, sheet_name="SampleAudit", index=False)
    print(f"[saved] metrics_summary.xlsx")
    return out


def print_summary(results: list[dict], baseline: dict = None):
    print(f"\n{'='*75}")
    print(f"  {'Variant':<30} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Sharpe[U]':>10}")
    print(f"  {'-'*75}")
    for res in results:
        label = res.get("label", _run_name(res["cfg"]))
        m  = res["metrics"]["overall"]
        mu = res["metrics"]["unstable"]
        print(f"  {label:<30} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
              f"{m['MaxDrawdown']:>8.4f} {m['Calmar']:>8.3f} "
              f"{mu['Sharpe']:>10.3f}")
    if baseline:
        m  = baseline["metrics"]["overall"]
        mu = baseline["metrics"]["unstable"]
        label = baseline.get("label", "Z-score Rule")
        print(f"  {'-'*75}")
        print(f"  {label:<30} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
              f"{m['MaxDrawdown']:>8.4f} {m['Calmar']:>8.3f} "
              f"{mu['Sharpe']:>10.3f}")
    print(f"{'='*75}")
    print("  [U] = all test days with RegimeStable=0; "
          "calendar 2021-2025 metrics are exported separately.\n")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation matrix (trains all variants)")
    parser.add_argument("--retrain",  action="store_true",
                        help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    if args.ablation:
        print("\nRunning full ablation matrix...")
        results = run_ablation(DEFAULT_CFG, retrain=args.retrain)
    else:
        # Single eval of the default config
        run_dir  = os.path.join(DEFAULT_CFG["output_dir"],
                                "checkpoints", _run_name(DEFAULT_CFG))
        ckpt     = os.path.join(run_dir, "best_model.pt")

        if not os.path.isfile(ckpt) or _checkpoint_is_stale(run_dir, DEFAULT_CFG):
            print("No checkpoint found (or its config is stale) — training first...")
            train(DEFAULT_CFG)

        print("\nEvaluating default config on test set...")
        res     = evaluate_config(DEFAULT_CFG, ckpt)
        res["label"] = "Full IQN"
        results = [res]

    # Z-score rule baseline
    print("\nComputing Z-score rule baseline...")
    baseline = zscore_baseline(DEFAULT_CFG["features_path"],
                               lambda_cost=DEFAULT_CFG["lambda_cost"])

    # Outputs
    print_summary(results, baseline)
    plot_equity_curves(results, baseline)
    save_metrics_xlsx(results, baseline)

    print(f"\nAll outputs → {EVAL_DIR}")
