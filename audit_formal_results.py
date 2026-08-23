"""Strict, read-only integrity audit for the formal walk-forward experiment."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_utils import file_sha256
from evaluate import compute_metrics
from make_walkforward_manifest import build_rows


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"


def main():
    errors, warnings = [], []
    expected = {(int(r["test_year"]), int(r["seed"]), r["variant"])
                for r in build_rows()}
    files = sorted((WF / "task_results").glob("*.json"))
    payloads = []
    for path in files:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["_path"] = path
            payloads.append(item)
        except Exception as exc:
            errors.append(f"Unreadable JSON {path.name}: {exc}")

    keys = [(int(x["test_year"]), int(x["seed"]), x["variant"])
            for x in payloads]
    counts = Counter(keys)
    found = set(keys)
    if expected - found:
        errors.append(f"Missing tasks: {sorted(expected - found)}")
    if found - expected:
        errors.append(f"Unexpected tasks: {sorted(found - expected)}")
    if any(n != 1 for n in counts.values()):
        errors.append(f"Duplicate task keys: {[k for k,n in counts.items() if n != 1]}")

    feature_hash = file_sha256(str(FEATURES))
    by_model_seed = defaultdict(list)
    for x in payloads:
        key = (int(x["test_year"]), int(x["seed"]), x["variant"])
        year, seed, variant = key
        d = x.get("daily", {})
        dates = pd.to_datetime(d.get("dates", []))
        pnls = np.asarray(d.get("pnls", []), dtype=float)
        pos = np.asarray(d.get("positions", []), dtype=float)
        if not (len(dates) == len(pnls) == len(pos) > 0):
            errors.append(f"{key}: unequal/empty daily arrays")
            continue
        if not (np.isfinite(pnls).all() and np.isfinite(pos).all()):
            errors.append(f"{key}: non-finite daily values")
        if not set(np.unique(pos)).issubset({-1.0, 0.0, 1.0}):
            errors.append(f"{key}: invalid positions {np.unique(pos)}")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            errors.append(f"{key}: dates duplicated or not increasing")
        if not ((dates.year == year).all()):
            errors.append(f"{key}: return dates outside test year")
        recomputed = compute_metrics(pnls, x["run_name"], pos)
        reported = x["metrics"]["overall"]
        for field, value in recomputed.items():
            if field not in reported:
                errors.append(f"{key}: missing overall metric {field}")
            elif value != reported[field]:
                errors.append(f"{key}: {field} reported={reported[field]} recomputed={value}")
        by_model_seed[(variant, seed)].append((year, dates, pnls, pos))

        run_dir = WF / f"seed_{seed}" / "checkpoints" / x["run_name"]
        cfg_path, ckpt = run_dir / "config.json", run_dir / "best_model.pt"
        if not cfg_path.exists() or not ckpt.exists():
            errors.append(f"{key}: checkpoint/config missing")
        else:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            expected_dates = {
                "train_end": f"{year-2}-12-31", "val_start": f"{year-1}-01-01",
                "val_end": f"{year-1}-12-31", "test_start": f"{year}-01-01",
                "test_end": f"{year}-12-31", "seed": seed,
            }
            for field, value in expected_dates.items():
                if cfg.get(field) != value:
                    errors.append(f"{key}: config {field}={cfg.get(field)!r}, expected {value!r}")
            if cfg.get("features_file_sha256") != feature_hash:
                errors.append(f"{key}: feature hash mismatch")
            if x.get("audit", {}).get("features_file_sha256") != feature_hash:
                errors.append(f"{key}: result feature hash missing/mismatched")
            if x.get("audit", {}).get("checkpoint_sha256") != file_sha256(str(ckpt)):
                errors.append(f"{key}: result checkpoint hash missing/mismatched")
            if cfg.get("pipeline_version") != 6:
                errors.append(f"{key}: pipeline version is not 6")
            if cfg.get("cvar_alpha") != 1.0:
                errors.append(f"{key}: cvar_alpha is not risk-neutral 1.0")
            if cfg.get("lookback") != 10:
                errors.append(f"{key}: unexpected lookback")
            if cfg.get("agent_type") == "DQN" and cfg.get("model_parameter_count") != 82051:
                errors.append(f"{key}: DQN parameter count mismatch")
            if cfg.get("agent_type") == "IQN" and cfg.get("model_parameter_count") != 81923:
                errors.append(f"{key}: IQN parameter count mismatch")

    for key, folds in by_model_seed.items():
        years = sorted(y for y, *_ in folds)
        if years != [2021, 2022, 2023, 2024, 2025]:
            errors.append(f"{key}: pooled fold years {years}")

    pooled = pd.read_csv(WF / "pooled_seed_results.csv")
    for (variant, seed), folds in by_model_seed.items():
        folds.sort(key=lambda z: z[0])
        pnls = np.concatenate([z[2] for z in folds])
        pos = np.concatenate([z[3] for z in folds])
        calc = compute_metrics(pnls, variant, pos)
        row = pooled[(pooled.variant == variant) & (pooled.seed == seed)]
        if len(row) != 1:
            errors.append(f"{(variant, seed)}: pooled row count {len(row)}")
            continue
        for field in ("CumReturn", "AnnReturn", "AnnVol", "Sharpe", "Sortino",
                      "MaxDrawdown", "Calmar", "PctPositive", "NPositionChanges", "N_days"):
            if not np.isclose(float(row.iloc[0][field]), float(calc[field]),
                              rtol=0, atol=1e-10, equal_nan=True):
                errors.append(f"{(variant, seed)}: pooled {field} mismatch")

    report = ["FORMAL RESULTS INTEGRITY AUDIT", "", f"JSON files: {len(files)}",
              f"Expected task keys: {len(expected)}", f"Feature SHA256: {feature_hash}",
              f"Errors: {len(errors)}", f"Warnings: {len(warnings)}", ""]
    report += [f"ERROR: {x}" for x in errors]
    report += [f"WARNING: {x}" for x in warnings]
    report += ["PASS" if not errors else "FAIL"]
    out = WF / "formal_integrity_audit.txt"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
