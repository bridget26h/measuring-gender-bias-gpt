
from pathlib import Path
import pandas as pd
import numpy as np
from metrics import (
    occupation_proportions,
    calculate_gsr,
    calculate_ampd,
    calculate_kendall_tau,
    calculate_tau_difference
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# load GSR and BLS benchmarks
def load_benchmarks():
    df = pd.read_csv(ROOT / "profession_list.csv")
    df = df.rename(columns={
        "Profession in GSR": "profession",
        "Average": "gsr_rating"
    })

    df["profession"] = df["profession"].str.strip().str.lower()

    # use original provided GSR values
    df["gsr"] = df["Men rate"] / 100

    # recalculate GSR for sensitivity analysis
    # since five entries are inconsistent
    df["gsr_recalculated"] = calculate_gsr(df["gsr_rating"])

    # convert BLS percentages to proportions
    df["bls"] = df["Men Percent"] / 100

    return df[["profession", "gsr", "gsr_recalculated", "bls"]]


# reproduce the statistical metrics from the original paper
def reproduce_metrics(df):
    model = df["male_proportion"]
    gsr = df["gsr"]
    bls = df["bls"]

    # calculate rank correlations and corresponding p-values
    tau_bls, p_bls = calculate_kendall_tau(model, bls)
    tau_gsr, p_gsr = calculate_kendall_tau(model, gsr)

    return {
        "model": "gpt-5.6-luna",
        "condition": "baseline",
        "n_occupations": len(df),
        "n_samples": df["total"].sum(),

        # overall gender representation
        "n_male": df["male"].sum(),
        "n_female": df["female"].sum(),
        "n_unknown": df["unknown"].sum(),
        "n_neutral": df["neutral"].sum(),

        "mean_male_proportion": model.mean(),
        "median_male_proportion": model.median(),

        # average deviations from benchmarks
        "ampd_bls": calculate_ampd(model, bls),
        "ampd_gsr": calculate_ampd(model, gsr),

        # correlations with benchmarks
        "tau_bls": tau_bls,
        "tau_gsr": tau_gsr,
        "p_value_bls": p_bls,
        "p_value_gsr": p_gsr,

        # difference in correlation for extension
        "tau_difference": calculate_tau_difference(
            model, gsr, bls
        ),

        # sensitivity analysis using recalculated GSR
        "ampd_gsr_recalculated": calculate_ampd(
            model, df["gsr_recalculated"]
        ),
        "tau_gsr_recalculated": calculate_kendall_tau(
            model, df["gsr_recalculated"]
        )[0]
    }


def main():
    # load classified baseline data
    names = pd.read_csv(
        RESULTS_DIR / "classified_names.csv"
    )

    names["profession"] = (
        names["profession"].str.strip().str.lower()
    )

    # make sure analyse only Luna baseline model
    if set(names["story_model"]) != {"gpt-5.6-luna"}:
        raise ValueError("Unexpected story model")

    proportions = occupation_proportions(names)
    benchmarks = load_benchmarks()

    # match each occupation to its corresponding benchmarks
    merged = proportions.merge(
        benchmarks,
        on="profession",
        how="left",
        validate="one_to_one"
    )

    # check for missing benchmarks
    if merged[["gsr", "gsr_recalculated", "bls"]].isna().any().any():
        raise ValueError("Missing benchmark values")

    if len(merged) != 106:
        raise ValueError("Expected 106 occupations")

    columns = ["male_proportion", "gsr", "gsr_recalculated", "bls"]

    for column in columns:
        if not merged[column].between(0, 1).all():
            raise ValueError(f"Invalid values in {column}")

    print(f"Matched occupations: {len(merged)}")

    # save occupation level results with benchmarks
    merged.to_csv(
        RESULTS_DIR / "occupation_proportions.csv",
        index=False
    )

    # calculate and save reproduction metrics
    metrics = reproduce_metrics(merged)

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "reproduction_metrics.csv",
        index=False
    )

    print("\nREPRODUCTION RESULTS")
    print("-" * 50)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")
    print("\nresults saved!")


if __name__ == "__main__":
    main()
