"""Expanding-window walk-forward experiment for all five RL variants.

Default design: five complete test years (2021--2025), the immediately
preceding year for validation, all earlier observations from 2004 onward for
training, and seeds 42/123/2024 for all variants. Full IQN and the
parameter-matched Double DQN additionally use seeds 7/999. The incomplete
2026 year is deliberately excluded from the main pooled estimates.

Usage:
    python run_walkforward.py
    python run_walkforward.py --test-years 2021 --seeds 42 --max-steps 100000
    python run_walkforward.py --retrain
"""

import argparse
import copy
import os
import time

import pandas as pd

from train import CFG, _run_name, train
from evaluate import ABLATION_VARIANTS, _checkpoint_is_stale, evaluate_config


DEFAULT_TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_SEEDS = [42, 123, 2024]
DEFAULT_CORE_EXTRA_SEEDS = [7, 999]
CORE_VARIANTS = {"Full IQN", "Parameter-Matched Double DQN"}


def fold_overrides(test_year: int) -> dict:
    val_year = test_year - 1
    return {
        "fold_id": f"test{test_year}",
        "train_end": f"{test_year - 2}-12-31",
        "val_start": f"{val_year}-01-01",
        "val_end": f"{val_year}-12-31",
        "test_start": f"{test_year}-01-01",
        "test_end": f"{test_year}-12-31",
    }


def run(test_years, seeds, core_extra_seeds=None,
        retrain=False, max_steps=None):
    root = os.path.join(CFG["output_dir"], "walkforward_v6")
    os.makedirs(root, exist_ok=True)
    rows = []
    core_extra_seeds = [s for s in (core_extra_seeds or []) if s not in seeds]
    schedule = []
    for test_year in test_years:
        for seed in seeds:
            for variant in ABLATION_VARIANTS:
                schedule.append((test_year, seed, variant))
        for seed in core_extra_seeds:
            for variant in ABLATION_VARIANTS:
                if variant[0] in CORE_VARIANTS:
                    schedule.append((test_year, seed, variant))

    total = len(schedule)
    job = 0

    for test_year, seed, variant in schedule:
        label, reg_s, reg_r, algo = variant
        job += 1
        cfg = copy.deepcopy(CFG)
        cfg.update(fold_overrides(test_year))
        cfg.update({
            "seed": seed,
            "agent_type": algo,
            "regime_in_state": reg_s,
            "regime_in_reward": reg_r,
            "output_dir": os.path.join(root, f"seed_{seed}"),
        })
        if max_steps is not None:
            cfg["max_steps"] = max_steps
            # Make short screening runs checkpoint-producing while
            # preserving the same rule across every variant.
            cfg["min_eps_for_ckpt"] = 1.0
            cfg["min_steps_before_stopping"] = max_steps
            cfg["eval_freq"] = min(cfg["eval_freq"], max_steps)

        run_name = _run_name(cfg)
        run_dir = os.path.join(cfg["output_dir"], "checkpoints", run_name)
        ckpt = os.path.join(run_dir, "best_model.pt")
        print(f"\n[{job}/{total}] {label} | test={test_year} | seed={seed}")
        t0 = time.time()
        if (retrain or not os.path.isfile(ckpt)
                or _checkpoint_is_stale(run_dir, cfg)):
            train(cfg)
        result = evaluate_config(cfg, ckpt)
        metrics = result["metrics"]["overall"]
        rows.append({
            "test_year": test_year,
            "seed": seed,
            "variant": label,
            "run_name": run_name,
            "Sharpe": metrics["Sharpe"],
            "Sortino": metrics["Sortino"],
            "MaxDrawdown": metrics["MaxDrawdown"],
            "Calmar": metrics["Calmar"],
            "CumReturn": metrics["CumReturn"],
            "AnnReturn": metrics["AnnReturn"],
            "NPositionChanges": metrics["NPositionChanges"],
            "elapsed_minutes": (time.time() - t0) / 60.0,
        })
        pd.DataFrame(rows).to_csv(
            os.path.join(root, "walkforward_results.csv"), index=False)

    detail = pd.DataFrame(rows)
    metric_cols = ["Sharpe", "Sortino", "MaxDrawdown", "Calmar",
                   "CumReturn", "AnnReturn", "NPositionChanges"]
    summary = detail.groupby("variant")[metric_cols].agg(["mean", "std"])
    seed_counts = detail.groupby("variant")["seed"].nunique().rename("N_seeds")
    fold_counts = detail.groupby("variant")["test_year"].nunique().rename("N_folds")
    summary.insert(0, ("Design", "N_folds"), fold_counts)
    summary.insert(1, ("Design", "N_seeds"), seed_counts)
    with pd.ExcelWriter(os.path.join(root, "walkforward_summary.xlsx")) as writer:
        detail.to_excel(writer, sheet_name="FoldSeedResults", index=False)
        summary.to_excel(writer, sheet_name="Mean_SD")
    print(f"\nCompleted {len(detail)} runs. Outputs: {root}")
    return detail, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-years", nargs="+", type=int,
                        default=DEFAULT_TEST_YEARS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--core-extra-seeds", nargs="+", type=int,
                        default=DEFAULT_CORE_EXTRA_SEEDS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    run(args.test_years, args.seeds, args.core_extra_seeds,
        args.retrain, args.max_steps)
