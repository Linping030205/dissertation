"""Repair legacy per-regime diagnostics in task JSON; never changes daily paths."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import regime_split_metrics
from commodity_utils import file_sha256


ROOT = Path(__file__).resolve().parent
WF = ROOT / "analysis_output" / "walkforward_v6"
FEATURES = ROOT / "analysis_output" / "spread_features_SSM.xlsx"


def main():
    features = pd.read_excel(FEATURES, sheet_name="Features",
                             index_col=0, parse_dates=True).sort_index()
    paths = sorted((WF / "task_results").glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        year = int(payload["test_year"])
        dates = pd.to_datetime(payload["daily"]["dates"])
        pnls = np.asarray(payload["daily"]["pnls"], dtype=float)
        positions = np.asarray(payload["daily"]["positions"], dtype=int)

        return_locs = features.index.get_indexer(dates)
        if (return_locs <= 0).any():
            raise ValueError(f"Cannot map return dates to preceding signals: {path.name}")
        signal_rows = features.iloc[return_locs - 1]
        scores = signal_rows["RegimeStabilityScore"].to_numpy(dtype=float)
        regimes = signal_rows["RegimeStable"].to_numpy(dtype=float)
        train_end = f"{year - 2}-12-31"
        train_scores = features.loc[:train_end, "RegimeStabilityScore"].dropna()
        bounds = (float(np.percentile(train_scores, 33)),
                  float(np.percentile(train_scores, 67)))
        label = payload["run_name"]
        seed = int(payload["seed"])
        checkpoint = (WF / f"seed_{seed}" / "checkpoints" / label /
                      "best_model.pt")
        repaired = regime_split_metrics(
            pnls, regimes, dates, positions, label,
            stability_scores=scores, stability_bounds=bounds,
        )
        if repaired["overall"] != payload["metrics"]["overall"]:
            raise AssertionError(f"Overall metrics unexpectedly changed: {path.name}")
        payload["metrics"] = repaired
        payload.setdefault("audit", {}).update({
            "regime_signal_alignment": "previous feature row relative to return_date",
            "regime_bounds_train_end": train_end,
            "regime_stability_bounds_p33_p67": list(bounds),
            "checkpoint_sha256": file_sha256(str(checkpoint)),
            "features_file_sha256": file_sha256(str(FEATURES)),
        })
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        os.replace(temp, path)
    print(f"Repaired and annotated {len(paths)} task JSON files.")


if __name__ == "__main__":
    main()
