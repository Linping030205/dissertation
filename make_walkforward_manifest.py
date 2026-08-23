"""Generate the deterministic 95-task formal experiment manifest."""

import csv
from pathlib import Path

from evaluate import ABLATION_VARIANTS
from run_walkforward import (CORE_VARIANTS, DEFAULT_CORE_EXTRA_SEEDS,
                             DEFAULT_SEEDS, DEFAULT_TEST_YEARS)


def build_rows():
    rows = []
    task_id = 0
    for year in DEFAULT_TEST_YEARS:
        schedule = [(seed, variant) for seed in DEFAULT_SEEDS
                    for variant in ABLATION_VARIANTS]
        schedule += [(seed, variant) for seed in DEFAULT_CORE_EXTRA_SEEDS
                     for variant in ABLATION_VARIANTS
                     if variant[0] in CORE_VARIANTS]
        for seed, (label, reg_s, reg_r, algo) in schedule:
            task_id += 1
            rows.append({
                "task_id": task_id,
                "test_year": year,
                "seed": seed,
                "variant": label,
                "agent_type": algo,
                "regime_in_state": int(reg_s),
                "regime_in_reward": int(reg_r),
            })
    assert len(rows) == 95
    return rows


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "walkforward_manifest.tsv"
    rows = build_rows()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {out}")

