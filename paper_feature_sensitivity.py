"""Grouped out-of-sample permutation sensitivity for the final v6 agents.

This is a post-training, fixed-checkpoint diagnostic.  It does not retrain an
agent, alter the formal 95-task results, or estimate a causal feature effect.
For each intervention, the selected feature's complete 10-day history is
replaced by a history drawn from a shuffled sequence of contiguous test-period
blocks.  Position state, realised returns, costs, reward accounting and every
unselected input remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate import compute_metrics
from rl_agent import make_agent
from rl_env import CrushSpreadEnv, _BASE_FEATS, _REGIME_STATE_FEATS


ROOT = Path(__file__).resolve().parent
WF_ROOT = ROOT / "analysis_output" / "walkforward_v6"
OUT_DIR = WF_ROOT / "paper_analysis" / "feature_sensitivity"

VARIANTS = {
    "Full IQN": "IQN_regS_regR",
    "w/o Regime-in-Reward": "IQN_regS_noRegR",
}
SEEDS = (42, 123, 2024)
YEARS = (2021, 2022, 2023, 2024, 2025)
FEATURES = list(_BASE_FEATS) + list(_REGIME_STATE_FEATS)
GROUPS = {
    **{name: (name,) for name in FEATURES},
    "EG + Half-life": ("RegimeEGScore", "RegimeHLScore"),
    "All Regime": tuple(_REGIME_STATE_FEATS),
    "All Base": tuple(_BASE_FEATS),
}


def _paths(prefix: str, seed: int, year: int) -> tuple[Path, Path, Path]:
    run_name = f"{prefix}_test{year}_v6"
    run_dir = WF_ROOT / f"seed_{seed}" / "checkpoints" / run_name
    result = WF_ROOT / "task_results" / f"{run_name}_seed{seed}.json"
    return run_dir / "config.json", run_dir / "best_model.pt", result


def _make_env(cfg: dict) -> CrushSpreadEnv:
    return CrushSpreadEnv(
        features_path=cfg["features_path"], split="test",
        lookback=cfg["lookback"], episode_len=cfg["episode_len"],
        regime_in_state=cfg["regime_in_state"],
        reward_mode="regime" if cfg["regime_in_reward"] else "base",
        lambda_cost=cfg["lambda_cost"],
        penalty_factor=cfg["penalty_factor"], cost_factor=cfg["cost_factor"],
        train_end=cfg["train_end"], val_start=cfg["val_start"],
        val_end=cfg["val_end"], test_start=cfg["test_start"],
        test_end=cfg.get("test_end"),
    )


def _exogenous_histories(env: CrushSpreadEnv) -> np.ndarray:
    """Reproduce the first lookback*n_features entries of env._get_obs()."""
    rows = []
    for offset in range(env.max_steps):
        t = env.test_start_pos + offset
        start = t - env.lookback + 1
        window = env.data[env.feature_cols].iloc[start:t + 1].to_numpy()
        norm = (window - env._feat_mean) / env._feat_std
        norm = np.nan_to_num(norm, nan=0.0, posinf=5.0, neginf=-5.0)
        rows.append(np.clip(norm.flatten(), -10.0, 10.0).astype(np.float32))
    return np.asarray(rows)


def _install_fast_observation(env: CrushSpreadEnv, histories: np.ndarray) -> None:
    """Avoid repeated pandas slicing during thousands of diagnostic rollouts."""
    def fast_obs() -> np.ndarray:
        offset = env.current_pos - env.test_start_pos
        exogenous = histories[offset]
        endogenous = np.asarray([
            float(env.position),
            min(float(env.days_in_pos) / max(env.episode_len, 1), 1.0),
        ], dtype=np.float32)
        return np.concatenate([exogenous, endogenous])
    env._get_obs = fast_obs


def _block_permutation(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous source blocks while preserving order within blocks."""
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def _feature_indices(names: tuple[str, ...], lookback: int) -> np.ndarray:
    selected = {FEATURES.index(name) for name in names}
    return np.asarray([
        lag * len(FEATURES) + feat
        for lag in range(lookback) for feat in sorted(selected)
    ], dtype=int)


def _rollout(env: CrushSpreadEnv, agent, histories: np.ndarray,
             selected: tuple[str, ...] | None = None,
             source: np.ndarray | None = None) -> dict:
    obs, _ = env.reset()
    pnls, positions, actions, dates = [], [], [], []
    indices = None if selected is None else _feature_indices(selected, env.lookback)
    done = False
    step = 0
    while not done:
        policy_obs = obs.copy()
        if indices is not None:
            policy_obs[indices] = histories[source[step], indices]
        action = agent.act(policy_obs, epsilon=0.0, deterministic=True)
        obs, _, done, _, info = env.step(action)
        actions.append(action)
        positions.append(info["position"])
        pnls.append(info["net_pnl"])
        dates.append(info["return_date"])
        step += 1
    return {
        "pnls": np.asarray(pnls, dtype=float),
        "positions": np.asarray(positions, dtype=int),
        "actions": np.asarray(actions, dtype=int),
        "dates": dates,
    }


def _path_stats(path: dict) -> dict:
    metrics = compute_metrics(path["pnls"], positions=path["positions"])
    return {
        "sharpe": metrics["Sharpe"],
        "net_pnl": metrics["CumReturn"],
        "position_changes": metrics["NPositionChanges"],
        "flat_pct": float(np.mean(path["positions"] == 0) * 100.0),
    }


def _load_task(label: str, prefix: str, seed: int, year: int):
    cfg_path, ckpt_path, result_path = _paths(prefix, seed, year)
    for path in (cfg_path, ckpt_path, result_path):
        if not path.exists():
            raise FileNotFoundError(path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["device"] = "cpu"
    env = _make_env(cfg)
    agent = make_agent(cfg["agent_type"], env.obs_dim, cfg)
    agent.load(str(ckpt_path))
    agent.online.eval()
    env.reset()
    histories = _exogenous_histories(env)
    _install_fast_observation(env, histories)
    baseline = _rollout(env, agent, histories)
    saved = json.loads(result_path.read_text(encoding="utf-8"))["daily"]
    saved_pnl = np.asarray(saved["pnls"], dtype=float)
    saved_pos = np.asarray(saved["positions"], dtype=int)
    if (baseline["dates"] != saved["dates"]
            or not np.allclose(baseline["pnls"], saved_pnl, atol=1e-12, rtol=0)
            or not np.array_equal(baseline["positions"], saved_pos)):
        raise AssertionError(f"Baseline reproduction failed: {label}, seed={seed}, {year}")
    return env, agent, histories, baseline


def summarise(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average permutation repeats within a checkpoint, then across tasks."""
    value_cols = [
        "sharpe_loss", "net_pnl_loss", "action_disagreement_pct",
        "delta_position_changes", "delta_flat_pct",
    ]
    task_means = (tasks.groupby(
        ["variant", "seed", "test_year", "group", "features"], sort=False
    )[value_cols].mean().reset_index())
    summary = (task_means.groupby(["variant", "group", "features"], sort=False)
               .agg(
                   n_tasks=("sharpe_loss", "size"),
                   mean_sharpe_loss=("sharpe_loss", "mean"),
                   sd_sharpe_loss=("sharpe_loss", "std"),
                   median_sharpe_loss=("sharpe_loss", "median"),
                   q025_sharpe_loss=("sharpe_loss", lambda x: np.quantile(x, .025)),
                   q975_sharpe_loss=("sharpe_loss", lambda x: np.quantile(x, .975)),
                   positive_sharpe_loss_pct=("sharpe_loss", lambda x: np.mean(x > 0) * 100),
                   mean_net_pnl_loss=("net_pnl_loss", "mean"),
                   mean_action_disagreement_pct=("action_disagreement_pct", "mean"),
                   mean_delta_position_changes=("delta_position_changes", "mean"),
                   mean_delta_flat_pct=("delta_flat_pct", "mean"),
               ).reset_index())
    return task_means, summary


def run(repeats: int, block: int, smoke: bool = False) -> pd.DataFrame:
    task_rows = []
    labels = list(VARIANTS.items())[:1] if smoke else list(VARIANTS.items())
    seeds = SEEDS[:1] if smoke else SEEDS
    years = YEARS[:1] if smoke else YEARS
    groups = list(GROUPS.items())[:2] if smoke else list(GROUPS.items())
    repeats = min(repeats, 2) if smoke else repeats

    for label, prefix in labels:
        for seed in seeds:
            for year in years:
                print(f"Loading {label}, seed {seed}, test {year} ...", flush=True)
                env, agent, histories, baseline = _load_task(label, prefix, seed, year)
                base_stats = _path_stats(baseline)
                for group_index, (group, names) in enumerate(groups):
                    for repeat in range(repeats):
                        # Mapping is shared across variants for the same fold/group/repeat.
                        rng = np.random.default_rng(
                            9173 + year * 1009 + seed * 17 + group_index * 101 + repeat
                        )
                        source = _block_permutation(len(baseline["pnls"]), block, rng)
                        perturbed = _rollout(env, agent, histories, names, source)
                        pert_stats = _path_stats(perturbed)
                        task_rows.append({
                            "variant": label, "seed": seed, "test_year": year,
                            "group": group, "features": " + ".join(names),
                            "repeat": repeat, "block_length": block,
                            "n_days": len(baseline["pnls"]),
                            "baseline_sharpe": base_stats["sharpe"],
                            "perturbed_sharpe": pert_stats["sharpe"],
                            "sharpe_loss": base_stats["sharpe"] - pert_stats["sharpe"],
                            "baseline_net_pnl": base_stats["net_pnl"],
                            "perturbed_net_pnl": pert_stats["net_pnl"],
                            "net_pnl_loss": base_stats["net_pnl"] - pert_stats["net_pnl"],
                            "action_disagreement_pct": float(
                                np.mean(baseline["actions"] != perturbed["actions"]) * 100.0
                            ),
                            "delta_position_changes": (
                                pert_stats["position_changes"] - base_stats["position_changes"]
                            ),
                            "delta_flat_pct": pert_stats["flat_pct"] - base_stats["flat_pct"],
                        })

    tasks = pd.DataFrame(task_rows)
    return tasks


def _plot(summary: pd.DataFrame, path: Path) -> None:
    variants = list(VARIANTS)
    groups = list(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, variant in zip(axes, variants):
        part = summary.set_index(["variant", "group"]).loc[variant].reindex(groups)
        values = part["mean_sharpe_loss"].to_numpy()
        colours = ["#d95f02" if g in ("EG + Half-life", "All Regime", "All Base")
                   else "#2c7fb8" for g in groups]
        ax.barh(groups, values, color=colours, alpha=.9)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(variant)
        ax.set_xlabel("Mean Sharpe loss (baseline - perturbed)")
        ax.grid(axis="x", alpha=.25)
    fig.suptitle("Grouped out-of-sample permutation sensitivity (20-day blocks)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    # These networks are evaluated one state at a time; large thread pools add
    # overhead without useful matrix-level parallelism.
    torch.set_num_threads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    task_path = OUT_DIR / f"feature_sensitivity_task_results{suffix}.csv"
    if args.summarize_only:
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        tasks = pd.read_csv(task_path)
    else:
        tasks = run(args.repeats, args.block_length, args.smoke)
        tasks.to_csv(task_path, index=False)
    task_means, summary = summarise(tasks)
    task_means.to_csv(OUT_DIR / f"feature_sensitivity_checkpoint_means{suffix}.csv", index=False)
    summary.to_csv(OUT_DIR / f"feature_sensitivity_summary{suffix}.csv", index=False)
    if not args.smoke:
        _plot(summary, OUT_DIR / "figure_feature_sensitivity.png")
        audit = (
            "Grouped out-of-sample permutation sensitivity\n"
            "Fixed final v6 checkpoints; no retraining or model selection.\n"
            f"Variants: {', '.join(VARIANTS)}\nSeeds: {SEEDS}\nYears: {YEARS}\n"
            f"Block length: {args.block_length}; repeats: {args.repeats}\n"
            "Each selected feature is replaced as a complete 10-day history.\n"
            "All unselected features, endogenous position state, realised returns, "
            "cost accounting and reward setting remain unchanged.\n"
            "Baseline daily P&L, positions and dates were exactly reproduced from "
            "the audited formal task JSON before each intervention.\n"
            "Interpretation: descriptive fixed-policy sensitivity, not causal "
            "feature attribution under retraining.\n"
        )
        (OUT_DIR / "feature_sensitivity_audit.txt").write_text(audit, encoding="utf-8")
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
