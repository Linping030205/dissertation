"""
commodity_utils.py
==================
Shared configuration, data loading, and statistical functions
for the inter-commodity spread trading research.

Imported by:
  data_analysis.py              -- static full-sample analysis
  pair_screening_walkforward.py -- rolling-window pair screening
  rl_env.py  (future)           -- RL trading environment
"""

from itertools import combinations
import hashlib
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# =========================================================================
# SHARED CONFIGURATION
# =========================================================================

# Raw Bloomberg Excel column order (encoding-safe positional rename)
EXCEL_COLS = [
    "Date", "Close", "Change", "PctChange",
    "Open", "Low", "High", "Volume", "OpenInterest", "Bid", "Ask",
]

DATA_FILES = {
    "CL": "CL1.xlsx",
    "NG": "NGc1.xlsx",
    "HG": "HGc1.xlsx",
    "S":  "Sc1.xlsx",
    "SM": "SMc1.xlsx",
}

FULL_NAMES = {
    "CL": "WTI Crude Oil",
    "NG": "Natural Gas",
    "HG": "Copper",
    "S":  "Soybeans",
    "SM": "Soybean Meal",
}

SECTORS = {
    "CL": "Energy", "NG": "Energy",
    "HG": "Metals",
    "S":  "Agriculture", "SM": "Agriculture",
}

UNITS = {
    "CL": "USD/barrel", "NG": "USD/MMBtu",
    "HG": "USD/lb",
    "S":  "USD/bushel",  "SM": "USD/short ton",
}

POOL  = list(DATA_FILES.keys())           # ['CL','NG','HG','S','SM']
PAIRS = list(combinations(POOL, 2))       # 10 candidate pairs

# Pair-screening thresholds (reference paper, Table 4/6)
TH_EG_PVAL   = 0.05   # Engle-Granger ADF p-value on residuals
TH_HALF_LIFE = 40     # OU half-life in trading days
TH_HURST     = 0.50   # R/S Hurst exponent
TH_HEDGECV   = 0.50   # CV of OLS hedge ratio across rolling sub-windows
                       # Kept at the reference-paper value. A Monte-Carlo
                       # null test (cv_noise_floor_analysis.py) suggests this
                       # threshold sits close to the sampling-noise floor for
                       # S-SM (~42% false-positive rate under a true-constant
                       # -beta null) -- worth flagging as a limitation/
                       # discussion point, but not a reason to depart from
                       # the cited standard for the main pipeline.

# Supplementary files used for comparison only (not in main pool)
_SUPPLEMENTARY_FILES = {
    "CL2": "CL2.xlsx",
    "CL3": "CL3.xlsx",
}


# =========================================================================
# DATA LOADING
# =========================================================================

def load_commodities(data_dir, supplementary=("CL2",)):
    """
    Load the five pool commodities from Excel and inner-join on common dates.

    Parameters
    ----------
    data_dir : str
        Directory containing the Excel files.
    supplementary : tuple[str]
        Extra files to load outside the pool (e.g. 'CL2' for calendar-spread
        comparison). NOT included in the inner-join alignment.

    Returns
    -------
    dfs : dict[str -> pd.DataFrame]
        Aligned DataFrames for pool assets, indexed by their common dates.
    common_idx : pd.DatetimeIndex
        The inner-joined set of trading days.
    extras : dict[str -> pd.DataFrame]
        Supplementary DataFrames (raw, not aligned).
    """
    raw = {}
    for key, fname in DATA_FILES.items():
        df = _read_excel(os.path.join(data_dir, fname))
        raw[key] = df

    # Inner-join: keep only dates present in all five series
    common_idx = raw[POOL[0]].index
    for key in POOL[1:]:
        common_idx = common_idx.intersection(raw[key].index)
    common_idx = common_idx.sort_values()

    dfs = {k: v.loc[common_idx] for k, v in raw.items()}

    extras = {}
    for key in supplementary:
        if key in _SUPPLEMENTARY_FILES:
            extras[key] = _read_excel(
                os.path.join(data_dir, _SUPPLEMENTARY_FILES[key])
            )

    return dfs, common_idx, extras


def _read_excel(path):
    df = pd.read_excel(path, header=0)
    df.columns = EXCEL_COLS
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["LogPrice"]  = np.log(df["Close"])
    return df


# =========================================================================
# STATISTICAL FUNCTIONS
# =========================================================================

def eg_test_single(log_y, log_x):
    """
    Single-direction Engle-Granger test.

    Steps:
      1. OLS: log_y = alpha + beta * log_x + epsilon
      2. Cointegration test on the OLS residuals, using MacKinnon's
         cointegration-specific p-value (statsmodels.tsa.stattools.coint),
         NOT a plain adfuller() p-value.

    Why coint() and not adfuller(resid) directly: adfuller()'s p-value is
    calibrated for testing a raw/exogenous series. Residuals from an
    estimated cointegrating regression are "super-consistent" (OLS chose
    beta specifically to make them look as stationary as possible), so
    reusing adfuller()'s own null distribution is anti-conservative --
    verified empirically on this dataset: across the 258 S-SM walk-forward
    windows, plain adfuller(resid) passes EG on 95 windows vs. 45 with
    coint()'s MacKinnon p-value (all_pass: 77 vs 34). coint() runs the same
    OLS+ADF procedure internally but returns the correctly-sized p-value.

    Returns
    -------
    p         : float    MacKinnon cointegration p-value
    beta      : float    OLS slope (hedge ratio)
    intercept : float    OLS intercept
    resid     : ndarray  OLS residuals
    """
    y   = np.asarray(log_y, dtype=float)
    x   = np.asarray(log_x, dtype=float)
    X   = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    _, p, _ = coint(y, x, trend="c", autolag="AIC")
    return p, fit.params[1], fit.params[0], fit.resid


def eg_test(log_a, log_b):
    """
    Bidirectional Engle-Granger: test A->B and B->A, return the
    direction with the lower (better) p-value.

    Testing both directions and keeping the smaller p-value is itself a
    (mild) multiple-comparisons problem -- it inflates the chance of a
    spuriously low p-value relative to testing one pre-specified direction.
    Applying a standard Bonferroni correction (double the p-value, capped
    at 1.0) for the 2 comparisons keeps the reported p-value conservative.

    Returns
    -------
    p         : float    Bonferroni-corrected MacKinnon p-value
    direction : str    'A->B' or 'B->A'
    beta      : float
    intercept : float
    resid     : ndarray
    """
    p1, b1, a1, r1 = eg_test_single(log_a, log_b)
    p2, b2, a2, r2 = eg_test_single(log_b, log_a)
    if p1 <= p2:
        p_corrected, direction, beta, intercept, resid = p1, "A->B", b1, a1, r1
    else:
        p_corrected, direction, beta, intercept, resid = p2, "B->A", b2, a2, r2
    p_corrected = min(1.0, p_corrected * 2.0)
    return p_corrected, direction, beta, intercept, resid


def ou_half_life(spread):
    """
    Ornstein-Uhlenbeck half-life via AR(1) regression.

    Model: delta_s_t = alpha + beta * s_{t-1} + epsilon_t
    Half-life = -ln(2) / beta

    Returns inf when beta >= 0 (spread is not mean-reverting).
    """
    s    = np.asarray(spread, dtype=float)
    ds   = np.diff(s)
    X    = sm.add_constant(s[:-1])
    beta = sm.OLS(ds, X).fit().params[1]
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


def hurst_rs(series, min_lag=20, n_lags=20):
    """
    R/S (rescaled-range) Hurst exponent.

    Interpretation:
      H < 0.5  anti-persistent / mean-reverting   <- what we want
      H = 0.5  random walk
      H > 0.5  trending / persistent

    Parameters
    ----------
    series  : array-like  Time series (log prices or spread values).
    min_lag : int         Minimum window size for R/S computation.
    n_lags  : int         Number of log-spaced lag sizes to evaluate.

    Returns
    -------
    H : float  (nan if insufficient data)
    """
    ts      = np.asarray(series, dtype=float)
    n       = len(ts)
    max_lag = max(n // 4, min_lag + 1)
    lags    = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), n_lags).astype(int)
    )
    rs_vals, used_lags = [], []
    for lag in lags:
        num_w = n // lag
        if num_w < 2:
            continue
        rs_w = []
        for w in range(num_w):
            chunk = ts[w * lag:(w + 1) * lag]
            if len(chunk) < 4:
                continue
            m   = np.mean(chunk)
            dev = np.cumsum(chunk - m)
            R   = dev.max() - dev.min()
            S   = np.std(chunk, ddof=1)
            if S > 0:
                rs_w.append(R / S)
        if rs_w:
            rs_vals.append(np.mean(rs_w))
            used_lags.append(lag)
    if len(used_lags) < 2:
        return np.nan
    return float(np.polyfit(np.log(used_lags), np.log(rs_vals), 1)[0])


def hedge_ratio_cv(log_y, log_x, n_windows=8):
    """
    Coefficient of variation of the OLS hedge ratio across n_windows
    rolling sub-windows.

    CV = std(betas) / |mean(betas)|

    Returns inf when fewer than 2 valid windows are available.
    """
    y   = np.asarray(log_y, dtype=float)
    x   = np.asarray(log_x, dtype=float)
    n   = len(y)
    wsz = n // n_windows
    betas = []
    for i in range(n_windows):
        s, e = i * wsz, (i + 1) * wsz
        if e - s < 30:
            continue
        X = sm.add_constant(x[s:e])
        betas.append(sm.OLS(y[s:e], X).fit().params[1])
    if len(betas) < 2:
        return np.inf
    betas = np.array(betas)
    return float(np.std(betas) / np.abs(np.mean(betas)))


def screen_pair(lp_a, lp_b,
                th_eg=TH_EG_PVAL, th_hl=TH_HALF_LIFE,
                th_hurst=TH_HURST, th_cv=TH_HEDGECV):
    """
    Run all four screening criteria on a pair of log-price arrays.

    Parameters
    ----------
    lp_a, lp_b : array-like   Log prices for asset A and B (same length).
    th_*        : float        Threshold overrides (default = module-level constants).

    Returns
    -------
    dict with keys:
      eg_p, direction, beta, intercept,
      half_life, hurst, hedge_cv,
      pass_eg, pass_hl, pass_h, pass_cv, all_pass,
      spread  (ndarray)
    """
    a = np.asarray(lp_a, dtype=float)
    b = np.asarray(lp_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    _nan = float("nan")
    if len(a) < 60:
        return dict(eg_p=_nan, direction="", beta=_nan, intercept=_nan,
                    half_life=np.inf, hurst=_nan, hedge_cv=_nan,
                    pass_eg=False, pass_hl=False, pass_h=False, pass_cv=False,
                    all_pass=False, spread=np.array([]))

    eg_p, direction, beta, intercept, _ = eg_test(a, b)

    if direction == "A->B":
        spread = a - beta * b - intercept
        cv     = hedge_ratio_cv(a, b)
    else:
        spread = b - beta * a - intercept
        cv     = hedge_ratio_cv(b, a)

    hl = ou_half_life(spread)
    H  = hurst_rs(spread)

    pass_eg = eg_p < th_eg
    pass_hl = np.isfinite(hl) and hl < th_hl
    pass_h  = np.isfinite(H)  and H  < th_hurst   # reported but NOT used in all_pass
    pass_cv = np.isfinite(cv) and cv < th_cv

    # Hurst (R/S) excluded from hard screening: diagnostic tests confirm severe
    # upward bias (+0.48) at n=252, making H<0.5 effectively unachievable even
    # for known mean-reverting OU processes.  Pass_h is retained for reporting.
    all_pass = all([pass_eg, pass_hl, pass_cv])

    return dict(
        eg_p=eg_p, direction=direction, beta=beta, intercept=intercept,
        half_life=hl, hurst=H, hedge_cv=cv,
        pass_eg=pass_eg, pass_hl=pass_hl, pass_h=pass_h, pass_cv=pass_cv,
        all_pass=all_pass,
        spread=spread,
    )


# =========================================================================
# SHARED TRADE-COST MODEL
# =========================================================================

def spread_trade_cost(lambda_cost: float,
                      new_pos: float, old_pos: float,
                      beta_new: float, beta_old: float) -> float:
    """
    Turnover-cost proxy for a 2-leg spread trade (one unit of the primary
    asset plus |beta| units of the hedge asset). Used by BOTH the RL
    environment and the Z-score baseline so the two are priced identically
    -- comparing a strategy's net P&L is only meaningful if both sides pay
    for turnover under the same cost model.

    Charges for:
      - a change in the discrete spread position (|new_pos - old_pos|,
        the primary-asset leg), and
      - hedge-leg rebalancing (|new_pos*beta_new - old_pos*beta_old|),
        which is nonzero even when the position itself doesn't change,
        because beta drifts day to day and the hedge leg must be
        re-sized to track it.

    This remains a lambda_cost-scaled proxy, not a dollar-notional cost:
    real contract multipliers, margin, bid/ask spread and roll costs are
    not modelled (the raw Bid/Ask columns in the source data were checked
    and found too inconsistent to use -- Close falls outside [Bid,Ask] on
    23-35% of days).
    """
    b_new = 0.0 if not np.isfinite(beta_new) else beta_new
    b_old = 0.0 if not np.isfinite(beta_old) else beta_old
    turnover = abs(new_pos - old_pos) + abs(new_pos * b_new - old_pos * b_old)
    return lambda_cost * turnover


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, for detecting a data file that was
    regenerated at the same path with different content (e.g. features
    re-run after a fix) even when nothing about the referencing config
    dict itself changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
