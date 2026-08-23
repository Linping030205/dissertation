"""
Reproducible small-sample diagnostic for the classical R/S Hurst estimator.

The experiment compares a stationary AR(1) discretisation of an
Ornstein-Uhlenbeck process with a 20-trading-day half-life against a random
walk. Both series have the same length as the walk-forward formation window.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from commodity_utils import hurst_rs


SEED = 42
N_OBS = 252
N_SIMULATIONS = 2_000
OU_HALF_LIFE = 20.0
OU_PHI = float(np.exp(-np.log(2.0) / OU_HALF_LIFE))

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_ou(rng: np.random.Generator) -> np.ndarray:
    innovations = rng.normal(size=N_OBS)
    series = np.zeros(N_OBS, dtype=float)
    for t in range(1, N_OBS):
        series[t] = OU_PHI * series[t - 1] + innovations[t]
    return series


def simulate_random_walk(rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.normal(size=N_OBS))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for simulation in range(N_SIMULATIONS):
        rows.append({
            "simulation": simulation + 1,
            "hurst_ou": hurst_rs(simulate_ou(rng)),
            "hurst_random_walk": hurst_rs(simulate_random_walk(rng)),
        })

    results = pd.DataFrame(rows)
    results["difference_ou_minus_rw"] = (
        results["hurst_ou"] - results["hurst_random_walk"]
    )

    summary = results[
        ["hurst_ou", "hurst_random_walk", "difference_ou_minus_rw"]
    ].agg(["count", "mean", "std", "median", "min", "max"]).T
    summary["q025"] = results.quantile(numeric_only=True, q=0.025)
    summary["q975"] = results.quantile(numeric_only=True, q=0.975)

    metadata = pd.DataFrame({
        "parameter": [
            "seed", "n_observations", "n_simulations",
            "ou_half_life", "ou_phi",
        ],
        "value": [
            SEED, N_OBS, N_SIMULATIONS, OU_HALF_LIFE, OU_PHI,
        ],
    })

    output = OUTPUT_DIR / "hurst_diagnostic_results.xlsx"
    with pd.ExcelWriter(output) as writer:
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        summary.to_excel(writer, sheet_name="Summary")
        results.to_excel(writer, sheet_name="Simulations", index=False)

    report = OUTPUT_DIR / "hurst_diagnostic_report.txt"
    report.write_text(
        "R/S HURST SMALL-SAMPLE DIAGNOSTIC\n"
        f"Seed: {SEED}\n"
        f"Observations per series: {N_OBS}\n"
        f"Simulations: {N_SIMULATIONS}\n"
        f"OU half-life: {OU_HALF_LIFE:.1f}\n"
        f"OU phi: {OU_PHI:.8f}\n\n"
        + summary.to_string(float_format=lambda x: f"{x:.6f}")
        + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {output}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
