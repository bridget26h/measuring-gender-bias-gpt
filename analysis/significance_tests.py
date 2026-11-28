
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, kendalltau
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
ALPHA = 0.05


# calculate 95% bootstrap confidence interval
def confidence_interval(values):
    return np.percentile(
        values,
        [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
    )


# bootstrap mean difference between two paired datasets
def bootstrap_mean_difference(model_values, benchmark_values):
    rng = np.random.default_rng(RANDOM_SEED)
    model_values = np.asarray(model_values)
    benchmark_values = np.asarray(benchmark_values)

    differences = model_values - benchmark_values
    n = len(differences)
    bootstrap_results = []

    for _ in range(N_BOOTSTRAP):
        # resample occupations with replacement
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_results.append(np.mean(differences[indices]))

    lower, upper = confidence_interval(bootstrap_results)
    return lower, upper


# test if model representation differs from benchmark
def test_representation(model_values, benchmark_values, benchmark_name):
    model_values = np.asarray(model_values)
    benchmark_values = np.asarray(benchmark_values)

    differences = model_values - benchmark_values

    # paired t-test since comparing same occupations
    t_stat, p_value = ttest_rel(
        model_values,
        benchmark_values
    )

    # nonparametric robustness check
    w_stat, w_p_value = wilcoxon(
        model_values,
        benchmark_values
    )

    # bootstrap confidence interval for mean difference
    lower, upper = bootstrap_mean_difference(
        model_values,
        benchmark_values
    )

    return {
        "test": f"representation_vs_{benchmark_name}",
        "effect_size": np.mean(differences),
        "ci_lower": lower,
        "ci_upper": upper,
        "test_statistic": t_stat,
        "p_value": p_value,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p_value
    }


# bootstrap difference between GSR and BLS correlations
def bootstrap_tau_difference(model_values, gsr, bls):
    rng = np.random.default_rng(RANDOM_SEED)
    model_values = np.asarray(model_values)
    gsr = np.asarray(gsr)
    bls = np.asarray(bls)
    n = len(model_values)

    # calculate observed correlation difference
    tau_gsr = kendalltau(
        model_values, gsr, variant="b"
    ).statistic

    tau_bls = kendalltau(
        model_values, bls, variant="b"
    ).statistic

    observed_difference = tau_gsr - tau_bls
    bootstrap_results = []

    for _ in range(N_BOOTSTRAP):
        # use same sampled indices for all three arrays
        # to preserve dependence between the correlations
        indices = rng.choice(n, size=n, replace=True)

        boot_model = model_values[indices]
        boot_gsr = gsr[indices]
        boot_bls = bls[indices]

        boot_tau_gsr = kendalltau(
            boot_model, boot_gsr, variant="b"
        ).statistic

        boot_tau_bls = kendalltau(
            boot_model, boot_bls, variant="b"
        ).statistic

        difference = boot_tau_gsr - boot_tau_bls
        if np.isfinite(difference):
            bootstrap_results.append(difference)

    bootstrap_results = np.asarray(bootstrap_results)
    lower, upper = confidence_interval(bootstrap_results)

    # center bootstrap differences around the null hypothesis
    centered = bootstrap_results - observed_difference

    # approximate two-sided bootstrap p-value
    p_value = (
        np.sum(
            np.abs(centered) >= abs(observed_difference)
        ) + 1
    ) / (len(centered) + 1)

    # save bootstrap distribution for visualisation
    pd.DataFrame({
        "tau_difference": bootstrap_results
    }).to_csv(
        RESULTS_DIR / "bootstrap_tau_distribution.csv",
        index=False
    )

    return {
        "test": "tau_gsr_vs_bls",
        "effect_size": observed_difference,
        "ci_lower": lower,
        "ci_upper": upper,
        "test_statistic": observed_difference,
        "p_value": p_value
    }


def main():
    # load occupation proportions and benchmark values
    df = pd.read_csv(
        RESULTS_DIR / "occupation_proportions.csv"
    )

    model = df["male_proportion"]
    gsr = df["gsr"]
    bls = df["bls"]

    # check if all observations are valid
    if df[["male_proportion", "gsr", "bls"]].isna().any().any():
        raise ValueError("Missing values in analysis dataset")

    results = []
    # test representation relative to BLS
    results.append(
        test_representation(model, bls, "bls")
    )
    # relative to GSR
    results.append(
        test_representation(model, gsr, "gsr")
    )

    # test difference in benchmark correlations
    results.append(
        bootstrap_tau_difference(model, gsr, bls)
    )

    results = pd.DataFrame(results)

    reject, adjusted_p, _, _ = multipletests(
        results["p_value"],
        alpha=ALPHA,
        method="holm"
    )

    results["adjusted_p_value"] = adjusted_p
    results["significant"] = reject

    results.insert(0, "model", "gpt-5.6-luna")
    results.insert(1, "condition", "baseline")

    # save results
    results.to_csv(
        RESULTS_DIR / "significance_results.csv",
        index=False
    )

    print("\nSTATISTICAL SIGNIFICANCE RESULTS")
    print("-" * 60)

    for _, row in results.iterrows():
        print(f"\nTest: {row['test']}")
        print(f"Effect size: {row['effect_size']:.6f}")

        print(
            f"95% CI: [{row['ci_lower']:.6f}, "
            f"{row['ci_upper']:.6f}]"
        )

        print(f"P-value: {row['p_value']:.6g}")

        print(
            f"Adjusted p-value: "
            f"{row['adjusted_p_value']:.6g}"
        )

        print(f"Significant: {row['significant']}")

    print("\nResults saved!")


if __name__ == "__main__":
    main()
