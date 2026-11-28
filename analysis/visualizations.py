from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_heatmap_data():
    baseline = pd.read_csv(RESULTS_DIR / "occupation_proportions.csv")
    primed = pd.read_csv(RESULTS_DIR / "associative_priming_occupation_proportions.csv")

    baseline = baseline[["profession", "male_proportion", "gsr", "bls"]].copy()
    baseline = baseline.rename(columns={"male_proportion": "baseline"})

    primed = primed[["profession", "male_proportion"]].copy()
    primed = primed.rename(columns={"male_proportion": "primed"})

    df = baseline.merge(primed, on="profession", how="inner", validate="one_to_one")

    # sort occupations by GSR
    # goes from more female associated to more male associated
    df = df.sort_values("gsr").reset_index(drop=True)

    return df



def make_figure1_heatmap():
    df = load_heatmap_data()

    if len(df) != 106:
        raise ValueError("Expected 106 occupations")

    heatmap = df[
        ["baseline", "primed", "bls", "gsr"]
    ].to_numpy().T

    if not np.isfinite(heatmap).all():
        raise ValueError("Missing heatmap values")

    if not ((heatmap >= 0) & (heatmap <= 1)).all():
        raise ValueError("Invalid proportions")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42
    })

    fig, ax = plt.subplots(figsize=(6.75, 2.5))

    # plot occupations using the same scale as the original paper
    im = ax.pcolormesh(
        heatmap,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        shading="flat"
    )

    ax.set_xlim(0, len(df))
    ax.set_ylim(4, 0)

    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels([
        "Luna baseline",
        "Luna primed",
        "U.S. BLS",
        "GSR"
    ])

    # remove individual occupation labels
    ax.set_xticks([])
    ax.set_xlabel("Occupations (ordered by GSR)", labelpad=8)

    ax.tick_params(axis="y", length=0, pad=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.035,
        pad=0.04,
        aspect=20
    )

    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_label("Male Proportion", labelpad=6)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=2)

    fig.tight_layout(pad=0.8)

    fig.savefig(
        FIGURES_DIR / "figure1_heatmap.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        FIGURES_DIR / "figure1_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Figure 1 saved!")


if __name__ == "__main__":
    make_figure1_heatmap()
