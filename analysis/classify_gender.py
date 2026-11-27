
import argparse
from pathlib import Path
import pandas as pd
import nomquamgender as nqg

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# classification thresholds from original paper
LOWER_THRESHOLD = 0.49
UPPER_THRESHOLD = 0.51


def classify_gender(probability):
    """
    classify name based on its female association probability
    """
    if pd.isna(probability):
        return "unknown"

    if probability < 0 or probability > 1:
        return "unknown"

    if probability < LOWER_THRESHOLD:
        return "male"

    if probability > UPPER_THRESHOLD:
        return "female"

    return "neutral"


def classify_dataset(input_path, output_path):
    """
    classify extracted names and save output dataset
    """
    df = pd.read_csv(input_path)
    if "extracted_name" not in df.columns:
        raise ValueError("Missing extracted_name column")
    print(f"Loaded {len(df)} samples")

    # preserve original names and get rid of whitespace
    df["cleaned_name"] = (
        df["extracted_name"].astype("string").str.strip()
    )

    # classify each unique name only once
    unique_names = (
        df["cleaned_name"]
        .dropna()
        .loc[lambda x: x != ""]
        .unique()
        .tolist()
    )
    print(f"Unique names: {len(unique_names)}")

    # initialise  gender classification model
    model = nqg.NBGC()
    annotations = model.annotate(
        unique_names,
        as_df=True
    )

    # map each name to its female association probability
    name_probabilities = dict(
        zip(annotations["given"], annotations["p(gf)"])
    )
    df["p_female"] = pd.to_numeric(
        df["cleaned_name"].map(name_probabilities),
        errors="coerce"
    )

    # assign gender labels
    df["gender"] = df["p_female"].apply(classify_gender)

    # save the classified dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nClassification results:")
    print(df["gender"].value_counts())
    print("\nClassification percentages:")
    print(df["gender"].value_counts(normalize=True) * 100)

    # calculate classification coverage
    classified = df["gender"].isin(["male", "female"]).sum()
    coverage = classified / len(df) if len(df) else 0

    print(f"\nClassification coverage: {coverage:.2%}")
    print(f"\nResults saved to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "full_run_names.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "classified_names.csv"
    )
    args = parser.parse_args()
    classify_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
