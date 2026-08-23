"""Verify the public dissertation result bundle using only the standard library."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
TASKS = WF / "task_results"


def metrics(pnls, positions):
    r = [float(x) for x in pnls]
    n = len(r)
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / n
    std = math.sqrt(var)
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in r) / n)
    equity = [0.0]
    for x in r:
        equity.append(equity[-1] + x)
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = min(max_dd, value - peak)
    ann_ret = mean * 252
    changes = sum(a != b for a, b in zip([0] + list(positions[:-1]), positions))
    return {
        "CumReturn": round(sum(r), 6),
        "AnnReturn": round(ann_ret, 6),
        "AnnVol": round(std * math.sqrt(252), 6),
        "Sharpe": round(mean / (std + 1e-10) * math.sqrt(252), 4),
        "Sortino": round(mean / (downside + 1e-10) * math.sqrt(252), 4),
        "MaxDrawdown": round(max_dd, 6),
        "Calmar": round(ann_ret / (-max_dd + 1e-10), 4) if max_dd < 0 else math.inf,
        "PctPositive": round(sum(x > 0 for x in r) / n * 100, 2),
        "NPositionChanges": changes,
        "N_days": n,
    }


def close(a, b, tol=1e-8):
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main():
    errors = []
    files = sorted(TASKS.glob("*.json"))
    records = []
    keys = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        records.append(item)
        keys.append((item["test_year"], item["seed"], item["variant"]))
        daily = item["daily"]
        dates, pnls, positions = daily["dates"], daily["pnls"], daily["positions"]
        if not (len(dates) == len(pnls) == len(positions)):
            errors.append(f"{path.name}: unequal daily array lengths")
            continue
        rebuilt = metrics(pnls, positions)
        saved = item["metrics"]["overall"]
        for name, value in rebuilt.items():
            if not close(value, saved[name]):
                errors.append(f"{path.name}: {name} saved={saved[name]} rebuilt={value}")
        if item.get("audit", {}).get("drawdown_equity_origin") != "E_0=0":
            errors.append(f"{path.name}: missing corrected drawdown provenance")

    if len(files) != 95:
        errors.append(f"expected 95 task files, found {len(files)}")
    if len(set(keys)) != len(keys):
        errors.append("duplicate (test_year, seed, variant) task keys")
    if {r["test_year"] for r in records} != {2021, 2022, 2023, 2024, 2025}:
        errors.append("test-year coverage is not 2021-2025")

    counts = Counter(r["variant"] for r in records)
    expected = {
        "Full IQN": 25,
        "Parameter-Matched Double DQN": 25,
        "w/o Regime-in-State": 15,
        "w/o Regime-in-Reward": 15,
        "No Explicit Regime IQN": 15,
    }
    if counts != Counter(expected):
        errors.append(f"unexpected variant counts: {dict(counts)}")

    headline_path = WF / "paper_analysis" / "model_summary_uncertainty.csv"
    with headline_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["strategy"]: row for row in csv.DictReader(handle)}
    locked = {
        "Full IQN": (-0.51662, -0.8051352),
        "Parameter-Matched Double DQN": (-0.1735, -0.5672422),
        "w/o Regime-in-State": (-0.3419, -0.5800643333333334),
        "w/o Regime-in-Reward": (-0.4640666666666667, -0.8112563333333332),
        "No Explicit Regime IQN": (-0.5921333333333334, -0.8435203333333333),
    }
    for name, (sharpe, drawdown) in locked.items():
        if name not in rows:
            errors.append(f"headline table missing {name}")
            continue
        if not close(rows[name]["Sharpe_Mean"], sharpe):
            errors.append(f"headline Sharpe mismatch for {name}")
        if not close(rows[name]["MaxDrawdown_Mean"], drawdown):
            errors.append(f"headline drawdown mismatch for {name}")

    print(f"Task files: {len(files)}")
    print(f"Unique task keys: {len(set(keys))}")
    for name in expected:
        print(f"{name}: {counts[name]}")
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()

