
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from metrics import occupation_proportions
from reproduction_analysis import load_benchmarks, reproduce_metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        default="associative_priming"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "luna_primed_classified_names.csv"
    )

    args = parser.parse_args()

    # load classified names
    names = pd.read_csv(args.input)
    names["profession"] = (
        names["profession"].str.strip().str.lower()
    )

    # check if using correct model
    if set(names["story_model"]) != {"gpt-5.6-luna"}:
        raise ValueError("Unexpected story model")

    # check for duplicate samples
    if names.duplicated(["profession", "sample_id"]).any():
        raise ValueError("Duplicate samples found")

    # calculate occupation level gender proportions
    proportions = occupation_proportions(names)

    # use the same benchmarks as the baseline
    benchmarks = load_benchmarks()

    merged = proportions.merge(
        benchmarks,
        on="profession",
        how="left",
        validate="one_to_one"
    )

    # check that all occupations have benchmarks
    if merged[["gsr", "gsr_recalculated", "bls"]].isna().any().any():
        raise ValueError("Missing benchmark values")

    if len(merged) != 106:
        raise ValueError("Expected 106 occupations")

    columns = [
        "male_proportion",
        "gsr",
        "gsr_recalculated",
        "bls"
    ]

    for column in columns:
        if not merged[column].between(0, 1).all():
            raise ValueError(f"Invalid values in {column}")

    # calculate the same metrics as the baseline
    metrics = reproduce_metrics(merged)
    metrics["condition"] = args.condition

    # save results separately from the baseline
    merged.to_csv(
        RESULTS_DIR / f"{args.condition}_occupation_proportions.csv",
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / f"{args.condition}_metrics.csv",
        index=False
    )

    print(f"\nRESULTS: {args.condition}")
    print("-" * 50)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")

    print("\nresults saved!")


if __name__ == "__main__":
    main()
