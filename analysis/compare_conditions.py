
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        default="associative_priming"
    )

    args = parser.parse_args()
    condition = args.condition

    baseline = pd.read_csv(
        RESULTS_DIR / "occupation_proportions.csv"
    )

    primed = pd.read_csv(
        RESULTS_DIR / f"{condition}_occupation_proportions.csv"
    )

    if set(baseline["profession"]) != set(primed["profession"]):
        raise ValueError("Occupations do not match")

    # match occupations across conditions
    merged = baseline.merge(
        primed,
        on="profession",
        suffixes=("_baseline", "_primed"),
        validate="one_to_one"
    )

    for benchmark in ["gsr", "bls", "gsr_recalculated"]:
        if not (
            merged[f"{benchmark}_baseline"]
            == merged[f"{benchmark}_primed"]
        ).all():
            raise ValueError(f"{benchmark} benchmarks do not match")

    # calculate change in male representation
    merged["delta_male_proportion"] = (
        merged["male_proportion_primed"]
        - merged["male_proportion_baseline"]
    )

    # calculate change in absolute distance from benchmarks
    for benchmark in ["bls", "gsr"]:
        merged[f"baseline_distance_{benchmark}"] = (
            merged["male_proportion_baseline"]
            - merged[f"{benchmark}_baseline"]
        ).abs()

        merged[f"primed_distance_{benchmark}"] = (
            merged["male_proportion_primed"]
            - merged[f"{benchmark}_primed"]
        ).abs()

        merged[f"delta_distance_{benchmark}"] = (
            merged[f"primed_distance_{benchmark}"]
            - merged[f"baseline_distance_{benchmark}"]
        )

    merged.to_csv(
        RESULTS_DIR / f"{condition}_occupation_comparison.csv",
        index=False
    )

    baseline_metrics = pd.read_csv(
        RESULTS_DIR / "reproduction_metrics.csv"
    ).iloc[0]

    primed_metrics = pd.read_csv(
        RESULTS_DIR / f"{condition}_metrics.csv"
    ).iloc[0]

    metric_names = [
        "mean_male_proportion",
        "median_male_proportion",
        "ampd_bls",
        "ampd_gsr",
        "tau_bls",
        "tau_gsr",
        "tau_difference",
        "ampd_gsr_recalculated",
        "tau_gsr_recalculated"
    ]

    # compare overall metrics between conditions
    comparison = []
    for metric in metric_names:
        comparison.append({
            "metric": metric,
            "baseline": baseline_metrics[metric],
            "primed": primed_metrics[metric],
            "change": (
                primed_metrics[metric]
                - baseline_metrics[metric]
            )
        })

    comparison = pd.DataFrame(comparison)

    comparison.to_csv(
        RESULTS_DIR / f"{condition}_metric_comparison.csv",
        index=False
    )

    delta = merged["delta_male_proportion"]

    print("\nCONDITION COMPARISON")
    print("-" * 60)
    print(comparison.to_string(index=False))

    print("\nOCCUPATION LEVEL CHANGES")
    print("-" * 60)

    print(f"Number of occupations: {len(merged)}")
    print(f"Mean change: {delta.mean():.6f}")
    print(f"Median change: {delta.median():.6f}")
    print(f"Mean absolute change: {delta.abs().mean():.6f}")

    print(f"Increase in male proportion: {(delta > 0).sum()}")
    print(f"Decrease in male proportion: {(delta < 0).sum()}")
    print(f"No change: {(delta == 0).sum()}")

    print("\nMEAN CHANGE IN BENCHMARK DISTANCE")
    print("-" * 60)

    for benchmark in ["bls", "gsr"]:
        change = merged[f"delta_distance_{benchmark}"].mean()
        print(f"{benchmark}: {change:.6f}")

    columns = [
        "profession",
        "male_proportion_baseline",
        "male_proportion_primed",
        "delta_male_proportion"
    ]

    print("\nLARGEST INCREASES")
    print(merged.nlargest(
        10, "delta_male_proportion"
    )[columns].to_string(index=False))

    print("\nLARGEST DECREASES")
    print(merged.nsmallest(
        10, "delta_male_proportion"
    )[columns].to_string(index=False))

    print("\nresults saved!")


if __name__ == "__main__":
    main()
