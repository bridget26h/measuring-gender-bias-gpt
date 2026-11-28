
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
N_BOOTSTRAP = 10000
SEED = 42

# bootstrap the mean paired difference
def bootstrap_ci(delta):
    rng = np.random.default_rng(SEED)

    indices = rng.integers(
        0, len(delta),
        size=(N_BOOTSTRAP, len(delta))
    )

    distribution = delta[indices].mean(axis=1)
    return np.percentile(distribution, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        default="associative_priming"
    )

    args = parser.parse_args()
    condition = args.condition

    baseline = pd.read_csv(
        RESULTS_DIR / "baseline_occupation_diversity.csv"
    )
    primed = pd.read_csv(
        RESULTS_DIR / f"{condition}_occupation_diversity.csv"
    )

    # match occupations and name types
    keys = ["profession", "name_type"]
    if set(map(tuple, baseline[keys].values)) != set(
        map(tuple, primed[keys].values)
    ):
        raise ValueError("Occupations or name types do not match")

    merged = baseline.merge(
        primed,
        on=keys,
        suffixes=("_baseline", "_primed"),
        validate="one_to_one"
    )

    results = []
    for name_type in ["first_name", "full_name"]:
        subset = merged[
            merged["name_type"] == name_type
        ]

        for metric in [
            "simpson_diversity",
            "shannon_entropy"
        ]:
            b = subset[f"{metric}_baseline"].to_numpy()
            p = subset[f"{metric}_primed"].to_numpy()

            if not np.isfinite(
                np.column_stack([b, p])
            ).all():
                raise ValueError("Invalid diversity values")

            delta = p - b

            if np.all(delta == 0):
                t_p = 1.0
                w_p = 1.0
            else:
                t_p = ttest_rel(p, b).pvalue
                w_p = wilcoxon(delta).pvalue

            lower, upper = bootstrap_ci(delta)

            results.append({
                "condition": condition,
                "name_type": name_type,
                "metric": metric,
                "n_occupations": len(delta),
                "baseline_mean": b.mean(),
                "primed_mean": p.mean(),
                "effect": delta.mean(),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": t_p,
                "wilcoxon_p_value": w_p
            })

    results = pd.DataFrame(results)

    # correct for multiple comparisons
    results["p_adjusted"] = multipletests(
        results["p_value"],
        method="holm"
    )[1]

    results["significant"] = (
        results["p_adjusted"] < 0.05
    )

    results.to_csv(
        RESULTS_DIR / f"{condition}_diversity_significance.csv",
        index=False
    )

    print("\nNAME DIVERSITY SIGNIFICANCE")
    print("-" * 60)
    print(results.to_string(index=False))

    print("\nresults saved!")


if __name__ == "__main__":
    main()
