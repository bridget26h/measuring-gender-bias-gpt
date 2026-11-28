
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from priming_significance import correlation_changes

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
N_BOOTSTRAP = 10000
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        default="associative_priming"
    )

    args = parser.parse_args()
    condition = args.condition
    df = pd.read_csv(
        RESULTS_DIR / f"{condition}_occupation_comparison.csv"
    )

    baseline = df["male_proportion_baseline"].to_numpy()
    primed = df["male_proportion_primed"].to_numpy()
    bls = df["bls_baseline"].to_numpy()
    gsr = df["gsr_baseline"].to_numpy()

    # number of binary classified stories per occupation
    n_baseline = (
        df["male_baseline"] + df["female_baseline"]
    ).to_numpy(dtype=int)

    n_primed = (
        df["male_primed"] + df["female_primed"]
    ).to_numpy(dtype=int)

    if np.any(n_baseline <= 0) or np.any(n_primed <= 0):
        raise ValueError("Empty occupation found")

    # check consistency with the original proportions
    if not np.allclose(
        baseline, df["male_baseline"] / n_baseline
    ):
        raise ValueError("Baseline proportions do not match")

    if not np.allclose(
        primed, df["male_primed"] / n_primed
    ):
        raise ValueError("Primed proportions do not match")

    observed = np.concatenate([
        [(primed - baseline).mean()],
        correlation_changes(baseline, primed, bls, gsr)
    ])

    rng = np.random.default_rng(SEED)
    n = len(df)

    distribution = np.full(
        (N_BOOTSTRAP, 4), np.nan
    )

    for i in range(N_BOOTSTRAP):
        # resample occupations
        indices = rng.integers(0, n, size=n)

        # resample stories within each occupation
        b = rng.binomial(
            n_baseline[indices],
            baseline[indices]
        ) / n_baseline[indices]

        p = rng.binomial(
            n_primed[indices],
            primed[indices]
        ) / n_primed[indices]

        # calculate changes in representation
        distribution[i, 0] = (p - b).mean()

        # calculate changes in rank correlations
        distribution[i, 1:] = correlation_changes(
            b, p, bls[indices], gsr[indices]
        )

    metric_names = [
        "mean_male_proportion_change",
        "tau_bls_change",
        "tau_gsr_change",
        "tau_difference_change"
    ]

    results = []

    for i, metric in enumerate(metric_names):
        values = distribution[:, i]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError("No valid bootstrap estimates")

        lower, upper = np.percentile(
            values, [2.5, 97.5]
        )

        results.append({
            "condition": condition,
            "metric": metric,
            "effect": observed[i],
            "bootstrap_mean": values.mean(),
            "ci_lower": lower,
            "ci_upper": upper,
            "excludes_zero": lower > 0 or upper < 0,
            "valid_iterations": len(values)
        })

    results = pd.DataFrame(results)

    results.to_csv(
        RESULTS_DIR / f"{condition}_hierarchical_bootstrap.csv",
        index=False
    )

    pd.DataFrame(
        distribution,
        columns=metric_names
    ).to_csv(
        RESULTS_DIR / f"{condition}_hierarchical_distribution.csv",
        index=False
    )

    print("\nHIERARCHICAL BOOTSTRAP")
    print("-" * 60)
    print(results.to_string(index=False))

    print("\nresults saved!")


if __name__ == "__main__":
    main()
