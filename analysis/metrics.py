
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# calculate proportion of male associated names
# exclude neutral and unknown classifications
def male_proportion(male_count, female_count):
    total = male_count + female_count

    # edge case of if no names can be classified
    if total == 0:
        return np.nan

    return male_count / total


# calculate gender distributions for each occupation
def occupation_proportions(df):
    # count number of samples in
    # each gender category for each profession
    counts = pd.crosstab(
        df["profession"],
        df["gender"]
    )

    # make sure all four categories are there
    counts = counts.reindex(
        columns=["male", "female", "neutral", "unknown"],
        fill_value=0
    )

    # total num samples before excluding
    counts["total"] = counts.sum(axis=1)
    # only include male and female classifications
    counts["classified"] = counts["male"] + counts["female"]
    # replace zero denominators with NaN
    denominator = counts["classified"].replace(0, np.nan)

    # calculate male and female proportions per occupation
    counts["male_proportion"] = counts["male"] / denominator
    counts["female_proportion"] = counts["female"] / denominator

    # Fraction of samples classified as either male or female
    counts["coverage"] = counts["classified"] / counts["total"]

    return counts.reset_index()


# Normalize Gender Stereotype Ratings (GSR)
def calculate_gsr(ratings):
    return (np.asarray(ratings) - 1) / 6


# calculate avg male proportion deviation (AMPD)
def calculate_ampd(model_values, benchmark_values):
    model_values = np.asarray(model_values)
    benchmark_values = np.asarray(benchmark_values)

    # negative AMPD means lower male representation relative to the benchmark
    return np.mean(model_values - benchmark_values)


# calculate Kendall's Tau to measure how closely
# model's occupational rankings align with a benchmark
def calculate_kendall_tau(model_values, benchmark_values):

    # use Tau-b to account for ties in the rankings
    tau, p_value = kendalltau(
        model_values,
        benchmark_values,
        variant="b"
    )

    # p-value tests the null hypothesis of
    # no rank association between the two variables
    return tau, p_value


# additional metric for extension:
# compare model's alignment with human stereotypes
# (GSR) against actual workforce statistics (BLS)
def calculate_tau_difference(model_values, gsr, bls):
    tau_gsr, _ = calculate_kendall_tau(model_values, gsr)
    tau_bls, _ = calculate_kendall_tau(model_values, bls)

    # positive difference means stronger alignment
    # with gender stereotypes than workforce statistics
    return tau_gsr - tau_bls
