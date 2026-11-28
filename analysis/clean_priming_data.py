
import csv
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "priming"
INPUT_FILE = DATA_DIR / "luna_primed_raw.csv"
OUTPUT_FILE = DATA_DIR / "luna_primed_stories.csv"


def clean_priming_data():
    rows = []

    # use csv reader since the raw csv has inconsistent row lengths
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # add the missing column header
        header.append("primer_response")
        for row in reader:
            # some stories were primed but the primer
            # response was not recorded
            if len(row) == 5:
                row.append("")
            # check for unexpected formatting
            if len(row) != 6:
                raise ValueError(f"Unexpected row length: {len(row)}")

            rows.append(row)

    df = pd.DataFrame(rows, columns=header)

    df["sample_id"] = pd.to_numeric(df["sample_id"])

    # add condition label for comparative analysis later
    df["condition"] = "associative_priming"

    # check for duplicate samples
    duplicates = df.duplicated(
        subset=["profession", "sample_id", "model"]
    ).sum()

    if duplicates > 0:
        raise ValueError("Duplicate samples found")

    # save cleaned data without modifying the raw file
    df.to_csv(OUTPUT_FILE, index=False)

    print("PRIMING DATA QUALITY CHECK")
    print("-" * 50)

    print("Total stories:", len(df))
    print("Unique occupations:", df["profession"].nunique())
    print("Duplicate samples:", duplicates)

    print("\nStory models:")
    print(df["model"].value_counts())

    print("\nRecorded primer responses:")
    print(df["primer_response"].str.strip().ne("").sum())

    print("\nMissing primer responses:")
    print(df["primer_response"].str.strip().eq("").sum())

    print("\nSamples per occupation:")
    print(df.groupby("profession").size().describe())

    # identify occupations with incomplete samples
    counts = df.groupby("profession").size()

    print("\nOccupations with fewer than 75 samples:")
    print(counts[counts < 75])

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    clean_priming_data()
