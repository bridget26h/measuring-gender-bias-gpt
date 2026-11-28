
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


# calculate diversity metrics for a set of names
def diversity_metrics(names):
    names = names.dropna().astype(str).str.strip().str.lower()
    names = names[names != ""]
    n = len(names)
    if n == 0:
        return {}

    counts = names.value_counts()
    probabilities = counts / n

    # Shannon entropy
    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # probability that two different samples have different names
    simpson = (
        1 - (counts * (counts - 1)).sum() / (n * (n - 1))
        if n > 1 else np.nan
    )

    return {
        "n_samples": n,
        "n_unique": names.nunique(),
        "unique_ratio": names.nunique() / n,
        "most_common_name": counts.index[0],
        "most_common_count": counts.iloc[0],
        "most_common_proportion": counts.iloc[0] / n,
        "shannon_entropy": entropy,
        "simpson_diversity": simpson
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        default="baseline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "classified_names.csv"
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    # normalize names and extract first names
    df["full_name"] = (
        df["cleaned_name"].str.strip().str.lower()
    )

    df["first_name"] = (
        df["full_name"].str.split().str[0]
    )

    # calculate overall name diversity
    overall = []
    for name_type in ["first_name", "full_name"]:
        overall.append({
            "condition": args.condition,
            "name_type": name_type,
            **diversity_metrics(df[name_type])
        })

    overall = pd.DataFrame(overall)

    # calculate diversity for each occupation
    occupation_results = []
    for profession, group in df.groupby("profession"):
        for name_type in ["first_name", "full_name"]:
            occupation_results.append({
                "condition": args.condition,
                "profession": profession,
                "name_type": name_type,
                **diversity_metrics(group[name_type])
            })

    occupation_results = pd.DataFrame(occupation_results)

    # calculate name frequencies
    frequencies = []

    for name_type in ["first_name", "full_name"]:
        names = df[name_type].dropna()
        names = names[names != ""]
        counts = names.value_counts()

        for name, count in counts.items():
            frequencies.append({
                "condition": args.condition,
                "name_type": name_type,
                "name": name,
                "count": count,
                "proportion": count / len(names)
            })

    frequencies = pd.DataFrame(frequencies)

    # save results
    overall.to_csv(
        RESULTS_DIR / f"{args.condition}_name_diversity.csv",
        index=False
    )

    occupation_results.to_csv(
        RESULTS_DIR / f"{args.condition}_occupation_diversity.csv",
        index=False
    )

    frequencies.to_csv(
        RESULTS_DIR / f"{args.condition}_name_frequencies.csv",
        index=False
    )

    print(f"\nNAME DIVERSITY: {args.condition}")
    print("-" * 60)
    print(overall.to_string(index=False))

    # summarize diversity across occupations
    print("\nAVERAGE OCCUPATION DIVERSITY")
    print("-" * 60)

    columns = [
        "n_unique",
        "unique_ratio",
        "most_common_proportion",
        "shannon_entropy",
        "simpson_diversity"
    ]

    print(
        occupation_results.groupby("name_type")[columns]
        .mean().to_string()
    )

    print("\nMOST COMMON FIRST NAMES")
    print("-" * 60)
    print(
        frequencies[
            frequencies["name_type"] == "first_name"
        ].head(10).to_string(index=False)
    )

    print("\nMOST COMMON FULL NAMES")
    print("-" * 60)
    print(
        frequencies[
            frequencies["name_type"] == "full_name"
        ].head(10).to_string(index=False)
    )

    print("\nresults saved!")


if __name__ == "__main__":
    main()
