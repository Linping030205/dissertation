"""Audit and aggregate completed Myriad walk-forward task JSON files."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import compute_metrics
from make_walkforward_manifest import build_rows


METRICS = ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "Sortino",
           "MaxDrawdown", "Calmar", "PctPositive", "NPositionChanges",
           "N_days"]


def collect(project_root):
    root = Path(project_root).resolve() / "analysis_output" / "walkforward_v6"
    result_dir = root / "task_results"
    payloads = []
    for path in sorted(result_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            item = json.load(f)
        item["_path"] = str(path)
        payloads.append(item)

    expected = {(int(r["test_year"]), int(r["seed"]), r["variant"])
                for r in build_rows()}
    task_keys = [(int(r["test_year"]), int(r["seed"]), r["variant"])
                 for r in payloads]
    key_counts = Counter(task_keys)
    duplicates = sorted(k for k, count in key_counts.items() if count > 1)
    found = set(task_keys)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    audit_path = root / "completion_audit.txt"
    with audit_path.open("w", encoding="utf-8") as f:
        f.write(f"Expected tasks: {len(expected)}\n")
        f.write(f"Completed tasks: {len(found & expected)}\n")
        f.write(f"Missing tasks: {len(missing)}\n")
        for row in missing:
            f.write(f"MISSING\t{row}\n")
        for row in unexpected:
            f.write(f"UNEXPECTED\t{row}\n")
        for row in duplicates:
            f.write(f"DUPLICATE\t{row}\tcount={key_counts[row]}\n")

    fold_rows = []
    for r in payloads:
        overall = r["metrics"]["overall"]
        fold_rows.append({"test_year": r["test_year"], "seed": r["seed"],
                          "variant": r["variant"], **overall})
    fold_df = pd.DataFrame(fold_rows)

    seed_rows = []
    if payloads:
        keys = sorted({(r["variant"], int(r["seed"])) for r in payloads})
        for variant, seed in keys:
            selected = sorted(
                [r for r in payloads
                 if r["variant"] == variant and int(r["seed"]) == seed],
                key=lambda r: int(r["test_year"]))
            pnls = np.concatenate([np.asarray(r["daily"]["pnls"], dtype=float)
                                   for r in selected])
            positions = np.concatenate(
                [np.asarray(r["daily"]["positions"], dtype=int)
                 for r in selected])
            metrics = compute_metrics(pnls, variant, positions)
            seed_rows.append({"variant": variant, "seed": seed,
                              "N_folds": len(selected), **metrics})
    seed_df = pd.DataFrame(seed_rows)

    if seed_df.empty:
        summary = pd.DataFrame()
    else:
        usable = [m for m in METRICS if m in seed_df.columns]
        summary = seed_df.groupby("variant")[usable].agg(["mean", "std"])
        summary.insert(0, ("Design", "N_seeds"),
                       seed_df.groupby("variant")["seed"].nunique())
        summary.insert(1, ("Design", "Min_folds_per_seed"),
                       seed_df.groupby("variant")["N_folds"].min())

    fold_df.to_csv(root / "fold_results.csv", index=False)
    seed_df.to_csv(root / "pooled_seed_results.csv", index=False)
    with pd.ExcelWriter(root / "walkforward_summary.xlsx") as writer:
        fold_df.to_excel(writer, sheet_name="AnnualFolds", index=False)
        seed_df.to_excel(writer, sheet_name="PooledBySeed", index=False)
        summary.to_excel(writer, sheet_name="Mean_SD")
        pd.DataFrame(missing, columns=["test_year", "seed", "variant"]).to_excel(
            writer, sheet_name="MissingTasks", index=False)
    print(f"Completed {len(found & expected)}/{len(expected)} expected tasks")
    print(f"Audit: {audit_path}")
    return missing, unexpected, duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    missing, unexpected, duplicates = collect(args.project_root)
    raise SystemExit(1 if (missing or unexpected or duplicates) else 0)
