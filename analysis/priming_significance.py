
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon, kendalltau
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

N_BOOTSTRAP = 10000
SEED = 42


# calculate changes in the rank correlations
def correlation_changes(baseline, primed, bls, gsr):
    tau_bls_baseline = kendalltau(baseline, bls).statistic
    tau_bls_primed = kendalltau(primed, bls).statistic

    tau_gsr_baseline = kendalltau(baseline, gsr).statistic
    tau_gsr_primed = kendalltau(primed, gsr).statistic

    delta_bls = tau_bls_primed - tau_bls_baseline
    delta_gsr = tau_gsr_primed - tau_gsr_baseline

    return np.array([
        delta_bls,
        delta_gsr,
        delta_gsr - delta_bls
    ])


# bootstrap by resampling the same occupations
# across both experimental conditions
def bootstrap_changes(baseline, primed, bls, gsr):
    rng = np.random.default_rng(SEED)
    n = len(baseline)

    distribution = np.full((N_BOOTSTRAP, 4), np.nan)

    for i in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)

        b = baseline[indices]
        p = primed[indices]

        distribution[i, 0] = np.mean(p - b)

        distribution[i, 1:] = correlation_changes(
            b, p, bls[indices], gsr[indices]
        )

    return distribution


# calculate 95% bootstrap confidence intervals
def confidence_interval(distribution):
    distribution = distribution[np.isfinite(distribution)]

    if len(distribution) == 0:
        raise ValueError("No valid bootstrap estimates")

    return np.percentile(distribution, [2.5, 97.5])


# approximate two sided p-value using
# the centered bootstrap distribution
def bootstrap_p_value(distribution, observed):
    distribution = distribution[np.isfinite(distribution)]

    if not np.isfinite(observed) or len(distribution) == 0:
        raise ValueError("Invalid bootstrap test")

    centered = distribution - observed

    count = np.sum(
        np.abs(centered) >= np.abs(observed)
    )

    return (count + 1) / (len(distribution) + 1)


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

    if not np.isfinite(
        np.column_stack([baseline, primed, bls, gsr])
    ).all():
        raise ValueError("Missing or invalid proportions")

    delta = primed - baseline

    # paired tests for changes in gender representation
    if np.all(delta == 0):
        t_p_value = 1.0
        wilcoxon_p_value = 1.0
    else:
        t_p_value = ttest_rel(primed, baseline).pvalue
        wilcoxon_p_value = wilcoxon(delta).pvalue

    # observed effects
    observed = np.concatenate([
        [delta.mean()],
        correlation_changes(baseline, primed, bls, gsr)
    ])

    # bootstrap distributions
    distribution = bootstrap_changes(
        baseline, primed, bls, gsr
    )

    test_names = [
        "mean_male_proportion_change",
        "tau_bls_change",
        "tau_gsr_change",
        "tau_difference_change"
    ]

    results = []

    for i, name in enumerate(test_names):
        lower, upper = confidence_interval(
            distribution[:, i]
        )

        if i == 0:
            p_value = t_p_value
            test = "paired_t_test"
        else:
            p_value = bootstrap_p_value(
                distribution[:, i], observed[i]
            )
            test = "paired_bootstrap"

        results.append({
            "condition": condition,
            "metric": name,
            "test": test,
            "effect": observed[i],
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_value
        })

    results = pd.DataFrame(results)

    # correct for the four primary hypothesis tests
    results["p_adjusted"] = multipletests(
        results["p_value"],
        method="holm"
    )[1]

    results["significant"] = (
        results["p_adjusted"] < 0.05
    )

    results.to_csv(
        RESULTS_DIR / f"{condition}_significance.csv",
        index=False
    )

    pd.DataFrame(
        distribution,
        columns=test_names
    ).to_csv(
        RESULTS_DIR / f"{condition}_bootstrap_distribution.csv",
        index=False
    )

    print("\nPRIMING SIGNIFICANCE TESTS")
    print("-" * 60)
    print(results.to_string(index=False))
    print("\nROBUSTNESS CHECK")
    print("-" * 60)
    print(f"Wilcoxon p-value: {wilcoxon_p_value:.6g}")
    print("\nresults saved!")


if __name__ == "__main__":
    main()
