"""Run formal walk-forward tasks sequentially on the local CPU.

Phases follow deadline priority:
  phase1: all five variants, seed 42, all five folds (25 tasks)
  phase2: all five variants, seeds 123/2024, all folds (50 tasks)
  phase3: Full IQN and matched DQN, seeds 7/999, all folds (20 tasks)

Completed, configuration-matching tasks are skipped by run_one(), so the
queue is safe to restart after interruption.
"""

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

from collect_walkforward_results import collect
from run_walkforward import CORE_VARIANTS
from run_walkforward_task import run_one


PHASE_SEEDS = {
    "phase1": {42},
    "phase2": {123, 2024},
    "phase3": {7, 999},
}


def load_tasks(manifest, phase):
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if phase == "all":
        return rows
    seeds = PHASE_SEEDS[phase]
    selected = [r for r in rows if int(r["seed"]) in seeds]
    if phase == "phase3":
        selected = [r for r in selected if r["variant"] in CORE_VARIANTS]
    return selected


def run_queue(project_root, manifest, phase):
    project_root = Path(project_root).resolve()
    output_root = project_root / "analysis_output" / "walkforward_v6"
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / f"local_queue_{phase}_status.json"
    lock_path = output_root / f"local_queue_{phase}.lock"

    if lock_path.exists():
        raise RuntimeError(
            f"Queue lock exists: {lock_path}. Check whether another queue is "
            "running before removing a stale lock.")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    tasks = load_tasks(manifest, phase)
    status = {"phase": phase, "pid": os.getpid(), "total": len(tasks),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "completed": [], "failed": []}
    try:
        for index, task in enumerate(tasks, 1):
            key = {"task_id": int(task["task_id"]),
                   "test_year": int(task["test_year"]),
                   "seed": int(task["seed"]),
                   "variant": task["variant"]}
            print(f"\nLOCAL QUEUE {phase} [{index}/{len(tasks)}] {key}", flush=True)
            t0 = time.time()
            try:
                result = run_one(key["test_year"], key["seed"], key["variant"],
                                 project_root, retrain=False)
                status["completed"].append(
                    {**key, "result": str(result),
                     "elapsed_minutes": (time.time() - t0) / 60.0})
            except Exception as exc:
                traceback.print_exc()
                status["failed"].append(
                    {**key, "error": repr(exc),
                     "elapsed_minutes": (time.time() - t0) / 60.0})
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        collect(project_root)
        print(f"Queue complete: {len(status['completed'])} succeeded, "
              f"{len(status['failed'])} failed", flush=True)
    finally:
        if lock_path.exists() and lock_path.read_text(encoding="utf-8") == str(os.getpid()):
            lock_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3", "all"],
                        default="phase1")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", default=Path(__file__).resolve().parent /
                        "walkforward_manifest.tsv")
    args = parser.parse_args()
    run_queue(args.project_root, args.manifest, args.phase)
