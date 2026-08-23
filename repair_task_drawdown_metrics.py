"""Recompute only MaxDrawdown and Calmar in formal task JSON files.

The daily P&L paths, positions, checkpoints, and all non-drawdown metrics are
left unchanged.  This repair aligns legacy task JSON with the standard equity
path used by ``evaluate.compute_metrics``, which now includes E_0 = 0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from evaluate import compute_metrics


ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "analysis_output" / "walkforward_v6" / "task_results"
DRAW_FIELDS = ("MaxDrawdown", "Calmar")


def main() -> None:
    paths = sorted(TASK_DIR.glob("*.json"))
    changed_files = 0
    changed_fields = 0

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        daily = payload["daily"]
        pnls = np.asarray(daily["pnls"], dtype=float)
        positions = np.asarray(daily["positions"], dtype=float)
        repaired = compute_metrics(pnls, payload["run_name"], positions)
        reported = payload["metrics"]["overall"]

        # Prove that this migration cannot silently alter another metric.
        for field, value in repaired.items():
            if field not in DRAW_FIELDS and reported.get(field) != value:
                raise AssertionError(
                    f"Non-drawdown metric changed in {path.name}: "
                    f"{field}={reported.get(field)!r}, recomputed={value!r}"
                )

        file_changed = False
        for section in ("overall", "calendar_2021_2025"):
            metrics = payload.get("metrics", {}).get(section, {})
            if not metrics:
                continue
            for field in DRAW_FIELDS:
                value = repaired[field]
                if metrics.get(field) != value:
                    metrics[field] = value
                    changed_fields += 1
                    file_changed = True

        payload.setdefault("audit", {})["drawdown_equity_origin"] = "E_0=0"
        if file_changed:
            changed_files += 1
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        os.replace(temp, path)

    print(
        f"Checked {len(paths)} task JSON files; updated {changed_fields} "
        f"drawdown fields across {changed_files} files."
    )


if __name__ == "__main__":
    main()
