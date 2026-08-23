"""
train.py
========
Training loop for the S-SM crush spread RL agent.

All hyperparameters and ablation switches are controlled by the CFG dict.
Run from the data directory:

    python train.py

Outputs (in analysis_output/checkpoints/<run_name>/):
    best_model.pt     – checkpoint saved at peak validation Sharpe
    training_log.csv  – step-level loss and epsilon
    val_log.csv       – Sharpe / returns at every evaluation point
"""

import os
import sys
import time
import csv
import json
import random

import numpy as np
import torch

from rl_env   import CrushSpreadEnv, FEATURES_PATH, OUTPUT_DIR
from rl_agent import ReplayBuffer, make_agent
from commodity_utils import file_sha256

# =========================================================================
# MASTER CONFIG  ← change any of these to run a different ablation variant
# =========================================================================
CFG = {
    "pipeline_version": 6,      # v6: expanding-window split support, automatic device
                                  # selection, and parameter-matched DQN configuration.
                                  # v5: fixed 92-dim observations for every ablation;
                                  # disabled regime-state history is masked to zero.
                                  # v4: corrected Engle-Granger p-value (coint() with
                                  # MacKinnon p-value + Bonferroni-corrected bidirectional
                                  # test, replacing a biased plain-adfuller(resid) p-value)
                                  # -- changes RegimeStable / regime scores.
                                  # v3: tradeable SpreadReturn, same-checkpoint block-avg
                                  # selection, leg/rebalance-weighted costs + close-out,
                                  # continuous regime state+reward, obs clipping
                                  # (v2 checkpoints are 72/62-dim; v3-v4 are 92/62-dim;
                                  # v5 is always 92-dim and uses different run dirs)
    # --- agent ---
    "agent_type":      "IQN",    # "IQN" | "QRDQN" | "DQN"
    "hidden_dim":      128,
    "dqn_hidden_dim":  224,      # 82,051 params vs IQN's 81,923 (0.16% difference)
    "n_quantiles":        64,    # N for IQN training (target uses N//2)
    "qrdqn_n_quantiles":  32,    # N for QR-DQN (separate from IQN to match literature)
    "n_cos":              64,    # cosine embedding size (IQN only)
    "cvar_alpha":      1.0,      # 1.0 = RISK-NEUTRAL (mean over the full quantile
                                  # distribution); <1.0 = CVaR risk-averse (mean over
                                  # only the bottom cvar_alpha fraction of quantiles).
                                  # At the current default, "IQN" here means
                                  # "distributional-RL value representation", NOT
                                  # "risk-averse" -- don't describe it as risk-averse
                                  # in writeups unless this is actually set < 1.0.

    # --- ablation switches ---
    "regime_in_state":  True,
    "regime_in_reward": True,

    # --- environment ---
    "lookback":        10,
    "episode_len":     60,
    "lambda_cost":     0.0005,
    "penalty_factor":  0.5,
    "cost_factor":     0.5,

    # --- optimiser / RL ---
    "lr":              3e-4,
    "gamma":           0.99,
    "batch_size":      64,
    "buffer_capacity": 100_000,

    # --- training schedule ---
    "warmup_steps":       2_000,
    "target_update_freq": 500,
    "eval_freq":          500,
    "patience":           20,       # eval rounds without val-Sharpe improvement
    "val_n_blocks":       4,        # split the validation year into this many
                                     # contiguous blocks and average their Sharpes
                                     # -- a same-checkpoint, multi-window criterion
                                     # (see _eval_episode). Replaces the earlier
                                     # cross-training-step smoothing, which averaged
                                     # results from different policy parameters and
                                     # did not reduce any single checkpoint's noise.
    "min_steps_before_stopping": 100_000,   # don't early-stop during high-epsilon phase
    "min_eps_for_ckpt":   0.30,    # only save checkpoint when policy is ≥70% deterministic
    "max_steps":          500_000,

    # --- exploration ---
    "eps_start": 1.0,
    "eps_end":   0.05,
    "eps_decay": 50_000,
    "seed":      42,
    "device":    "auto",        # CUDA when available, otherwise CPU

    # --- time split (overridden by walk-forward folds) ---
    "train_end":  "2019-12-31",
    "val_start":  "2020-01-01",
    "val_end":    "2020-12-31",
    "test_start": "2021-01-01",
    "test_end":   None,
    "fold_id":    None,

    # --- paths ---
    "features_path": FEATURES_PATH,
    "output_dir":    OUTPUT_DIR,
}


def _run_name(cfg: dict) -> str:
    """Generate a short run identifier from the key ablation flags."""
    parts = [cfg["agent_type"]]
    parts.append("regS" if cfg["regime_in_state"]  else "noRegS")
    parts.append("regR" if cfg["regime_in_reward"] else "noRegR")
    if cfg.get("cvar_alpha", 1.0) < 1.0:
        parts.append(f"CVaR{cfg['cvar_alpha']}")
    if cfg.get("fold_id"):
        parts.append(str(cfg["fold_id"]))
    parts.append(f"v{cfg.get('pipeline_version', 1)}")
    return "_".join(parts)


def _epsilon(step: int, cfg: dict) -> float:
    e0, e1, decay = cfg["eps_start"], cfg["eps_end"], cfg["eps_decay"]
    return e1 + (e0 - e1) * np.exp(-step / decay)


def _eval_episode(env: CrushSpreadEnv, agent,
                  n_blocks: int = 4) -> tuple[float, float, np.ndarray]:
    """
    Run ONE deterministic pass of the current checkpoint through the
    validation episode and return:
      (whole-episode Sharpe, block-averaged Sharpe, net-P&L array)

    block-averaged Sharpe splits the SAME rollout into `n_blocks`
    contiguous sub-periods and averages their individual Sharpes. This is
    a same-checkpoint, multi-window noise reduction (no extra env cost,
    since it reuses the single rollout) -- unlike averaging val_sharpe
    across successive training steps, which mixes results from different
    policy parameters and doesn't estimate any one checkpoint's quality.
    """
    obs, _ = env.reset()
    pnls, done = [], False
    while not done:
        action = agent.act(obs, epsilon=0.0, deterministic=True)
        obs, _, done, _, info = env.step(action)
        pnls.append(info["net_pnl"])
    arr = np.array(pnls, dtype=np.float64)
    sharpe = (arr.mean() / (arr.std() + 1e-10)) * np.sqrt(252)

    blocks = np.array_split(arr, n_blocks)
    block_sharpes = [
        (b.mean() / (b.std() + 1e-10)) * np.sqrt(252)
        for b in blocks if len(b) >= 5
    ]
    block_avg = float(np.mean(block_sharpes)) if block_sharpes else float(sharpe)

    return float(sharpe), block_avg, arr


def train(cfg: dict = CFG):
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name  = _run_name(cfg)
    ckpt_dir  = os.path.join(cfg["output_dir"], "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build (but do NOT yet write) the config.json payload. `risk_profile`
    # is a derived, human-readable field so a run's risk stance is
    # unambiguous without having to know the cvar_alpha<1.0-means-CVaR
    # convention: agent_type != "IQN" has no quantile-truncation mechanism
    # at all, so it is risk-neutral by construction regardless of cvar_alpha.
    # `features_file_sha256` lets evaluate.py's staleness guard detect a
    # features file that was regenerated at the same path with different
    # content, which a plain cfg-dict comparison can't see.
    cvar_alpha = cfg.get("cvar_alpha", 1.0)
    if cfg["agent_type"] != "IQN":
        risk_profile = "risk-neutral (agent has no CVaR mechanism)"
    elif cvar_alpha >= 1.0:
        risk_profile = "risk-neutral (cvar_alpha=1.0: mean over full quantile range)"
    else:
        risk_profile = f"CVaR risk-averse (cvar_alpha={cvar_alpha})"
    cfg_out = dict(cfg, risk_profile=risk_profile,
                   features_file_sha256=file_sha256(cfg["features_path"]))
    config_path = os.path.join(ckpt_dir, "config.json")

    def _save_checkpoint(agent_obj, sharpe_value):
        # Write best_model.pt and config.json TOGETHER, only once a
        # checkpoint is actually produced. If training crashes before this
        # ever runs, neither file is touched -- the directory keeps
        # whatever it had before (old config.json + old best_model.pt,
        # consistently stale together), so a later staleness check still
        # correctly triggers a retrain instead of pairing a fresh config
        # with a stale model.
        agent_obj.save(os.path.join(ckpt_dir, "best_model.pt"))
        with open(config_path, "w") as f:
            json.dump(dict(cfg_out, best_block_avg_val_sharpe=sharpe_value), f, indent=2)

    reward_mode = "regime" if cfg["regime_in_reward"] else "base"

    # ---- environments ----
    env_kwargs = dict(
        features_path  = cfg["features_path"],
        lookback       = cfg["lookback"],
        episode_len    = cfg["episode_len"],
        regime_in_state = cfg["regime_in_state"],
        reward_mode    = reward_mode,
        lambda_cost    = cfg["lambda_cost"],
        penalty_factor = cfg["penalty_factor"],
        cost_factor    = cfg["cost_factor"],
        train_end      = cfg.get("train_end", "2019-12-31"),
        val_start      = cfg.get("val_start", "2020-01-01"),
        val_end        = cfg.get("val_end", "2020-12-31"),
        test_start     = cfg.get("test_start", "2021-01-01"),
        test_end       = cfg.get("test_end"),
    )
    train_env = CrushSpreadEnv(split="train", seed=seed, **env_kwargs)
    val_env   = CrushSpreadEnv(split="val",   **env_kwargs)

    obs_dim = train_env.obs_dim
    print(f"\n{'='*60}")
    print(f"  Run   : {run_name}")
    print(f"  obs   : {obs_dim}  |  actions: 3  |  steps: {cfg['max_steps']:,}")
    print(f"{'='*60}\n")

    # ---- agent & buffer ----
    agent  = make_agent(cfg["agent_type"], obs_dim, cfg)
    model_parameter_count = sum(p.numel() for p in agent.online.parameters()
                                if p.requires_grad)
    cfg_out["model_parameter_count"] = model_parameter_count
    cfg_out["resolved_device"] = str(next(agent.online.parameters()).device)
    print(f"  trainable parameters: {model_parameter_count:,}")
    print(f"  device: {cfg_out['resolved_device']}")
    buffer = ReplayBuffer(cfg["buffer_capacity"], obs_dim,
                          device=next(agent.online.parameters()).device)

    # ---- logging ----
    train_log_path = os.path.join(ckpt_dir, "training_log.csv")
    val_log_path   = os.path.join(ckpt_dir, "val_log.csv")

    with open(train_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "epsilon"])
    with open(val_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "val_sharpe", "val_sharpe_blockavg"])

    # ---- warm-up: fill buffer with random actions ----
    print(f"Warm-up: filling replay buffer ({cfg['warmup_steps']} steps)...")
    obs, _ = train_env.reset()
    for _ in range(cfg["warmup_steps"]):
        action  = train_env.action_space.sample()
        nobs, r, done, _, info = train_env.step(action)
        buffer.push(obs, action, r, nobs, done)
        obs = nobs
        if done:
            obs, _ = train_env.reset()
    print(f"Buffer size after warm-up: {len(buffer)}\n")

    # ---- main training loop ----
    best_val_sharpe = -np.inf
    patience_count  = 0
    total_steps     = 0
    episode_reward  = 0.0
    episode_num     = 0
    obs, _          = train_env.reset()
    t0              = time.time()

    while total_steps < cfg["max_steps"]:
        eps    = _epsilon(total_steps, cfg)
        action = agent.act(obs, epsilon=eps)
        nobs, r, done, _, info = train_env.step(action)
        buffer.push(obs, action, r, nobs, done)
        episode_reward += r
        obs = nobs
        total_steps += 1

        if done:
            episode_num += 1
            obs, _ = train_env.reset()
            episode_reward = 0.0

        # ---- update ----
        if len(buffer) >= cfg["batch_size"]:
            batch = buffer.sample(cfg["batch_size"])
            loss  = agent.update(batch)

            if total_steps % 200 == 0:
                with open(train_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([total_steps, f"{loss:.6f}", f"{eps:.4f}"])

        # ---- target network sync ----
        if total_steps % cfg["target_update_freq"] == 0:
            agent.sync_target()

        # ---- validation ----
        if total_steps % cfg["eval_freq"] == 0:
            # block-averaged Sharpe: SAME checkpoint evaluated on n_blocks
            # contiguous sub-periods of the validation year, averaged. This
            # replaces the earlier (incorrect) approach of averaging
            # val_sharpe across successive training steps, which mixed
            # results from different policy parameters rather than
            # reducing the noise of evaluating any single checkpoint.
            val_sharpe, val_blockavg, _ = _eval_episode(
                val_env, agent, n_blocks=cfg.get("val_n_blocks", 4))
            elapsed = time.time() - t0
            print(f"step {total_steps:>7,} | eps {eps:.3f} | "
                  f"val Sharpe {val_sharpe:+.3f} (block-avg {val_blockavg:+.3f}) | "
                  f"{elapsed/60:.1f} min elapsed")

            with open(val_log_path, "a", newline="") as f:
                csv.writer(f).writerow([total_steps, f"{val_sharpe:.6f}", f"{val_blockavg:.6f}"])

            ckpt_eligible = eps <= cfg.get("min_eps_for_ckpt", 1.0)
            if ckpt_eligible:
                if val_blockavg > best_val_sharpe:
                    best_val_sharpe = val_blockavg
                    patience_count  = 0
                    _save_checkpoint(agent, best_val_sharpe)
                    print(f"  *** new best block-avg val Sharpe {best_val_sharpe:+.4f} → saved ***")
                else:
                    patience_count += 1
                    min_stop = cfg.get("min_steps_before_stopping", 0)
                    if patience_count >= cfg["patience"] and total_steps >= min_stop:
                        print(f"\nEarly stopping at step {total_steps:,} "
                              f"(no improvement for {cfg['patience']} eval rounds)")
                        break

    print(f"\nTraining done.  Best block-avg val Sharpe = {best_val_sharpe:+.4f}")
    print(f"Checkpoint: {os.path.join(ckpt_dir, 'best_model.pt')}")
    return ckpt_dir


if __name__ == "__main__":
    train(CFG)
