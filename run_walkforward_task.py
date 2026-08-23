"""Run one recoverable walk-forward task from the 95-task manifest."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

from evaluate import ABLATION_VARIANTS, _checkpoint_is_stale, evaluate_config
from commodity_utils import file_sha256
from run_walkforward import fold_overrides
from train import CFG, _run_name, train


def _variant_by_label(label):
    matches = [v for v in ABLATION_VARIANTS if v[0] == label]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous variant: {label!r}")
    return matches[0]


def task_from_manifest(path, task_id):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if task_id < 1 or task_id > len(rows):
        raise ValueError(f"task_id must be between 1 and {len(rows)}")
    return rows[task_id - 1]


def run_one(test_year, seed, label, project_root, retrain=False,
            max_steps=None):
    _, reg_s, reg_r, algo = _variant_by_label(label)
    project_root = Path(project_root).resolve()
    output_root = project_root / "analysis_output" / "walkforward_v6"
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(CFG)
    cfg.update(fold_overrides(test_year))
    cfg.update({
        "seed": seed,
        "agent_type": algo,
        "regime_in_state": reg_s,
        "regime_in_reward": reg_r,
        "features_path": str(project_root / "analysis_output" /
                             "spread_features_SSM.xlsx"),
        "output_dir": str(output_root / f"seed_{seed}"),
        "device": "auto",
    })
    if max_steps is not None:
        cfg["max_steps"] = max_steps
        cfg["min_eps_for_ckpt"] = 1.0
        cfg["min_steps_before_stopping"] = max_steps
        cfg["eval_freq"] = min(cfg["eval_freq"], max_steps)

    run_name = _run_name(cfg)
    run_dir = Path(cfg["output_dir"]) / "checkpoints" / run_name
    ckpt = run_dir / "best_model.pt"
    result_dir = output_root / "task_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{run_name}_seed{seed}.json"

    checkpoint_is_current = (ckpt.exists()
                             and not _checkpoint_is_stale(str(run_dir), cfg))
    result_is_current = False
    if result_path.exists() and checkpoint_is_current and not retrain:
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            audit = old.get("audit", {})
            result_is_current = (
                int(old.get("test_year")) == int(test_year)
                and int(old.get("seed")) == int(seed)
                and old.get("variant") == label
                and old.get("run_name") == run_name
                and audit.get("checkpoint_sha256") == file_sha256(str(ckpt))
                and audit.get("features_file_sha256")
                    == file_sha256(cfg["features_path"])
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            result_is_current = False
    if result_is_current:
        print(f"Complete, hash-verified result already exists; skipping: {result_path}")
        return result_path
    if retrain or not checkpoint_is_current:
        train(cfg)
    result = evaluate_config(cfg, str(ckpt))
    payload = {
        "test_year": test_year,
        "seed": seed,
        "variant": label,
        "run_name": run_name,
        "metrics": result["metrics"],
        "audit": {
            **result["audit"],
            "checkpoint_sha256": file_sha256(str(ckpt)),
            "features_file_sha256": file_sha256(cfg["features_path"]),
        },
        # Retain daily out-of-sample paths so annual folds can be joined into
        # one genuine 2021-2025 walk-forward path for each model/seed.
        "daily": {
            "dates": [str(x.date()) for x in result["dates"]],
            "pnls": result["pnls"].tolist(),
            "positions": result["positions"].tolist(),
        },
    }
    temp_path = result_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, result_path)
    print(f"Saved completed task: {result_path}")
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parent)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    task = task_from_manifest(args.manifest, args.task_id)
    run_one(int(task["test_year"]), int(task["seed"]), task["variant"],
            args.project_root, args.retrain, args.max_steps)
