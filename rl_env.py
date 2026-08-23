"""
rl_env.py
=========
Gymnasium-compatible trading environment for the S-SM crush spread.

Observation (flat vector)
    lookback × n_features  +  2 scalars (position, days_in_position)
    All variants use 9 × 10 + 2 = 92 inputs. When regime_in_state=False,
    the three standardised regime channels are fixed at zero, so the
    ablation does not change network capacity.

Actions
    0 = Short Spread  (position -1)
    1 = Flat          (position  0)
    2 = Long Spread   (position +1)

Reward modes
    'base'   R_t = c_t × SpreadReturn_{t+1} − spread_trade_cost(...)
             (cost prices both legs of the spread trade -- one S contract
             plus |beta_t| SM contracts -- and includes hedge-ratio
             rebalancing cost even when the position itself is unchanged;
             a forced close-out cost is also charged if a position remains
             open when an episode/the test window ends. See
             commodity_utils.spread_trade_cost, shared with the Z-score
             baseline in evaluate.py so the two are cost-comparable.)
    'regime' Same base, but losses are amplified and transaction costs are
             increased in proportion to (1 - RegimeStabilityScore), a
             continuous [0,1] statistic (not the binary RegimeStable flag,
             which is ~0% for most of 2021-2025 and would otherwise
             collapse this to an almost-constant multiplier) — mirrors
             SAPT (Lu et al. 2022) while using a fully interpretable
             statistical regime signal.

Train/Val/Test splits
    Train : 2004-start → 2019-12-31  (random 60-day episode starts)
    Val   : 2020-01-01 → 2020-12-31  (one continuous episode)
    Test  : 2021-01-01 → end-of-data (one continuous episode)

Normalisation
    All features standardised with training-set mean/std (no look-ahead).
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from commodity_utils import spread_trade_cost

DATA_DIR   = str(Path(__file__).resolve().parent)
OUTPUT_DIR = str(Path(DATA_DIR) / "analysis_output")

FEATURES_PATH = str(Path(OUTPUT_DIR) / "spread_features_SSM.xlsx")

TRAIN_END  = "2019-12-31"
VAL_START  = "2020-01-01"
VAL_END    = "2020-12-31"
TEST_START = "2021-01-01"

_BASE_FEATS   = ["ZScore20", "SpreadReturn", "RollStd20",
                 "DeltaBeta5d", "VolS_roll20", "VolSM_roll20"]
_REGIME_FEAT  = "RegimeStable"   # kept for info dict / stable-vs-unstable eval split
# Continuous regime diagnostics used as STATE features when regime_in_state=True.
# The binary RegimeStable flag is ~0% for most of 2021-2025 (see
# spread_features_report.txt), so it alone carries almost no information
# during that stretch; these three keep varying day to day throughout.
_REGIME_STATE_FEATS  = ["RegimeEGScore", "RegimeHLScore", "RegimeCVScore"]
# Continuous stability score used for REWARD shaping (reward_mode="regime"),
# in place of the binary RegimeStable -- same reasoning as above.
_REGIME_REWARD_FEAT  = "RegimeStabilityScore"


class CrushSpreadEnv(gym.Env):
    """S-SM crush spread trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_path: str = FEATURES_PATH,
        split: str = "train",
        lookback: int = 10,
        episode_len: int = 60,
        regime_in_state: bool = True,
        reward_mode: str = "regime",
        lambda_cost: float = 0.0005,
        penalty_factor: float = 0.5,
        cost_factor: float = 0.5,
        seed: int = None,
        train_end: str = TRAIN_END,
        val_start: str = VAL_START,
        val_end: str = VAL_END,
        test_start: str = TEST_START,
        test_end: str = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        assert reward_mode in ("base", "regime"), f"Unknown reward_mode '{reward_mode}'"

        self.split          = split
        self.lookback       = lookback
        self.episode_len    = episode_len
        self.regime_in_state = regime_in_state
        self.reward_mode    = reward_mode
        self.lambda_cost    = lambda_cost
        self.penalty_factor = penalty_factor
        self.cost_factor    = cost_factor
        self.train_end      = train_end
        self.val_start      = val_start
        self.val_end        = val_end
        self.test_start     = test_start
        self.test_end       = test_end

        # ---- load features ----
        self.data = pd.read_excel(features_path, sheet_name="Features",
                                  index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)

        # Keep the observation layout and network capacity identical across
        # the state ablation. Disabled regime channels are masked in
        # _get_obs() after normalisation rather than removed here.
        self.feature_cols = list(_BASE_FEATS) + list(_REGIME_STATE_FEATS)
        self.n_features = len(self.feature_cols)

        # Always 92 at the default lookback: 9 features * 10 days + 2 scalars.
        self.obs_dim = self.n_features * lookback + 2

        # ---- split positions & normalisation ----
        self._setup_splits()
        self._compute_norm()

        # ---- spaces ----
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=Short, 1=Flat, 2=Long
        # action_space has its own internal RNG, separate from self._rng /
        # the base gym.Env seeding -- without seeding it explicitly here,
        # action_space.sample() (used for warm-up exploration in train.py)
        # is NOT reproducible across runs with the same nominal seed.
        self.action_space.seed(seed)

        # ---- episode state ----
        self.current_pos     = 0
        self.step_count      = 0
        self.max_steps       = 0
        self.position        = 0
        self.days_in_pos     = 0

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _setup_splits(self):
        idx = self.data.index

        def _start_pos(date_str):
            loc = idx.searchsorted(pd.Timestamp(date_str))
            return int(min(loc, len(idx) - 1))

        def _end_pos(date_str):
            loc = idx.searchsorted(pd.Timestamp(date_str), side="right") - 1
            return int(max(0, min(loc, len(idx) - 1)))

        # First position where ALL required features are non-NaN
        need = list(self.feature_cols) + [_REGIME_FEAT, _REGIME_REWARD_FEAT]
        need = list(dict.fromkeys(need))   # deduplicate, preserve order
        valid = self.data[need].notna().all(axis=1).values
        first_valid = int(np.argmax(valid))   # first True index

        # Need `lookback` additional steps of valid data before the episode start
        self.first_usable  = first_valid + self.lookback

        self.train_end_pos  = _end_pos(self.train_end)
        self.val_start_pos  = _start_pos(self.val_start)
        self.val_end_pos    = _end_pos(self.val_end)
        self.test_start_pos = _start_pos(self.test_start)
        # Last available return date. step() is called with t up to len(idx)-2
        # and reads the realised return at t+1, including the final data row.
        self.data_end_pos   = (_end_pos(self.test_end)
                               if self.test_end else len(idx) - 1)

        # Train episode sampling bounds
        self.min_ep_start = self.first_usable
        self.max_ep_start = self.train_end_pos - self.episode_len
        if self.max_ep_start < self.min_ep_start:
            raise ValueError("Training window is too short for lookback + episode_len")
        if self.val_end_pos <= self.val_start_pos:
            raise ValueError("Validation window must contain at least two observations")
        if self.data_end_pos <= self.test_start_pos:
            raise ValueError("Test window must contain at least two observations")

    def _compute_norm(self):
        """Training-set mean/std for each feature column."""
        train_slice = self.data[self.feature_cols].iloc[
            self.first_usable : self.train_end_pos + 1
        ]
        mean = train_slice.mean().values.astype(np.float32)
        std  = train_slice.std().values.astype(np.float32)
        std[std < 1e-8] = 1.0
        self._feat_mean = mean   # (n_features,)
        self._feat_std  = std

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        super().reset(seed=seed)

        if self.split == "train":
            start = int(self._rng.integers(self.min_ep_start,
                                            self.max_ep_start + 1))
            self.max_steps = self.episode_len
        elif self.split == "val":
            start = self.val_start_pos
            self.max_steps = self.val_end_pos - self.val_start_pos
        else:   # test
            start = self.test_start_pos
            self.max_steps = self.data_end_pos - self.test_start_pos

        self.current_pos = start
        self.step_count  = 0
        self.position    = 0
        self.days_in_pos = 0

        return self._get_obs(), {}

    def step(self, action: int):
        new_position = int(action) - 1    # {0,1,2} → {-1,0,+1}
        t = self.current_pos

        # ---- spread return at t+1 ----
        sr_raw = float(self.data["SpreadReturn"].iloc[t + 1])
        spread_return = 0.0 if np.isnan(sr_raw) else sr_raw

        # ---- regime at t ----
        # `regime` (binary RegimeStable) is kept for the info dict / the
        # stable-vs-unstable evaluation split in evaluate.py. Reward
        # shaping below uses the continuous RegimeStabilityScore instead,
        # since the binary flag is ~0% for most of 2021-2025 and would
        # otherwise collapse the penalty to an almost-constant multiplier.
        regime           = float(self.data[_REGIME_FEAT].iloc[t])
        stability_score  = float(self.data[_REGIME_REWARD_FEAT].iloc[t])
        if np.isnan(stability_score):
            stability_score = 0.0

        # ---- hedge-ratio-weighted turnover cost ----
        # A spread trade is two legs: one S contract plus |beta| SM
        # contracts, and the SM leg must be re-sized whenever beta drifts
        # even if the discrete position itself doesn't change. Priced via
        # the shared spread_trade_cost() (same formula used by the
        # Z-score baseline in evaluate.py, so the two are cost-comparable).
        beta_t    = float(self.data["RollingBeta"].iloc[t])
        beta_prev = float(self.data["RollingBeta"].iloc[t - 1]) if t > 0 else beta_t

        # ---- P&L ----
        gross_pnl      = new_position * spread_return
        turnover_cost  = spread_trade_cost(self.lambda_cost, new_position, self.position,
                                           beta_t, beta_prev)

        # ---- update position tracking ----
        if new_position != 0:
            self.days_in_pos = (self.days_in_pos + 1
                                if new_position == self.position else 1)
        else:
            self.days_in_pos = 0
        self.position = new_position

        # ---- advance ----
        self.current_pos += 1
        self.step_count  += 1
        terminated = (self.step_count >= self.max_steps or
                      self.current_pos + 1 >= len(self.data))
        truncated  = False

        # Forced close-out: a real trader must unwind any open position by
        # the end of the evaluation window. Without this, a terminal open
        # position's remaining exposure is neither charged nor realised.
        closing_cost = (spread_trade_cost(self.lambda_cost, 0.0, new_position, beta_t, beta_t)
                        if terminated and new_position != 0 else 0.0)
        base_cost = turnover_cost + closing_cost

        pnl  = gross_pnl
        cost = base_cost
        if self.reward_mode == "regime":
            if pnl < 0.0:
                pnl  *= (1.0 + (1.0 - stability_score) * self.penalty_factor)
            cost *= (1.0 + (1.0 - stability_score) * self.cost_factor)

        reward = float(pnl - cost)

        obs  = (self._get_obs() if not terminated
                else np.zeros(self.obs_dim, dtype=np.float32))
        info = {
            "raw_pnl":      gross_pnl,  # backwards-compatible alias
            "gross_pnl":    gross_pnl,
            "transaction_cost": base_cost,
            "net_pnl":      gross_pnl - base_cost,
            "reward":       reward,
            "position":     self.position,
            "regime":       regime,
            "stability_score": stability_score,
            "spread_return": spread_return,
            "signal_date":  self.data.index[t].strftime("%Y-%m-%d"),
            "return_date":  self.data.index[t + 1].strftime("%Y-%m-%d"),
            # Backwards-compatible alias; realised P&L belongs to return_date.
            "date":         self.data.index[t + 1].strftime("%Y-%m-%d"),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        t     = self.current_pos
        start = t - self.lookback + 1
        window = self.data[self.feature_cols].iloc[start : t + 1].values  # (L, n_feat)
        norm   = (window - self._feat_mean) / self._feat_std
        if not self.regime_in_state:
            regime_start = len(_BASE_FEATS)
            norm[:, regime_start:] = 0.0
        obs = np.concatenate([
            norm.flatten(),
            [float(self.position),
             min(float(self.days_in_pos) / max(self.episode_len, 1), 1.0)],
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        # Clip to the declared observation_space bounds: normalised features
        # can occasionally exceed +-10 (e.g. DeltaBeta5d, SpreadReturn spikes),
        # which otherwise silently violates the Box(-10,10) contract.
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        pass

    # Backwards-compatible signature; v5 keeps capacity fixed for ablations.
    @staticmethod
    def obs_dim_for(regime_in_state: bool, lookback: int = 10) -> int:
        n_feat = len(_BASE_FEATS) + len(_REGIME_STATE_FEATS)
        return n_feat * lookback + 2
