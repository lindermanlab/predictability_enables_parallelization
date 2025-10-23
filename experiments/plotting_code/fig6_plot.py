"""
Plotting code for Figure 6.
Logic very specific to the sweeps we ran in wandb_sweep_configs
i.e. fig6_gd.yaml and fig6_no_gd.yaml
Would need to adjust logic for different sweeps.
Timing done on H100 with 80GB VRAM.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import argparse

colors = {
    "gd": "tab:brown",
    "seq": "tab:green",
    "deer": "tab:gray",
    "quasi": "tab:orange",
}

# Configure the line styles, colors etc.
cols = {
    "Sequential": colors["seq"],
    "DEER": colors["deer"],
    "Quasi-DEER": colors["quasi"],
    "Gradient Descent": colors["gd"],
}
tgs = {
    "seq": "Sequential",
    "deer": "DEER (Gauss-Newton)",
    "quasi": "Quasi-DEER",
    "gd": "Gradient Descent",
}
labs = [
    "Sequential",
    "DEER",
    "Quasi-DEER",
    "Gradient Descent",
]
opacity = 0.9

# Use TeX.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Configure font sizes.
INCREASE = 2
SMALL_SIZE = 6 + INCREASE
MEDIUM_SIZE = 8 + INCREASE
BIGGER_SIZE = 10 + INCREASE
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

def main(filepath_gd, filepath_no_gd):
    with open(filepath_no_gd, "rb") as f:
        df = pickle.load(f)
    with open(filepath_gd, "rb") as f:
        df_gd = pickle.load(f)
    # tuning the step size for gradient descent
    df_og_drop = df[~((df["alg"] == "gd") & (df["g"].isin([0.5, 0.6])))]
    df_surgery = pd.concat(
        [
            df_og_drop,
            df_gd[(df_gd["gd_step_size"] == 0.6) & (df_gd["g"] == 0.5)],
            df_gd[(df_gd["gd_step_size"] == 0.5) & (df_gd["g"] == 0.6)],
            df_gd[(df_gd["gd_step_size"] == 0.25) & (df_gd["g"] > 0.6)],
        ],
        ignore_index=True,
    )

    # full plot

    # set up
    lle_ticks = np.array([-0.6, -0.4, -0.2, 0])
    S_VAL = 10
    Z_VAL = 3
    Y_MIN = -25

    # make the fig
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    for alg in df_surgery["alg"].unique():
        sub = df_surgery[df_surgery["alg"] == alg]
        if alg != "seq":
            axes[0].scatter(
                sub["lle"],
                sub["n_iters"],
                label=tgs[alg],
                alpha=opacity,
                s=S_VAL,
                zorder=Z_VAL,
                color=colors[alg],
            )
        axes[1].scatter(
            sub["lle"],
            1000 * sub["time"],
            label=tgs[alg],
            alpha=opacity,
            s=S_VAL,
            zorder=Z_VAL,
            color=colors[alg],
        )

    # Labels & legend
    axes[0].set_ylabel("steps to convergence")
    axes[1].set_ylabel("wallclock time (ms)")
    axes[1].set_xlabel(r"LLE ($\lambda$)")
    axes[0].set_xticks(lle_ticks)
    # axes[1].legend(loc="upper left")
    axes[0].axvline(0, color="k", linestyle="--", label="lle=0")
    axes[1].axvline(0, color="k", linestyle="--", label="lle=0")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.axvline(x=0, color="k", linestyle=":", lw=1)
        ax.grid(True, which="major", color="gray", alpha=0.5)
        ax.minorticks_off()

    axes[0].set_ylim(Y_MIN, 1050)
    axes[1].set_ylim(0, 350)

    handles, labels = axes[1].get_legend_handles_labels()

    # ordering
    desired = ["DEER (Gauss-Newton)", "Quasi-DEER", "Gradient Descent", "Sequential"]

    # map to indices (and skip anything not present)
    idx = [labels.index(name) for name in desired if name in labels]

    axes[0].legend(
        [handles[i] for i in idx], [labels[i] for i in idx], loc="upper left", frameon=True
    )


    fig.suptitle("Convergence rates of optimization algorithms")

    plt.tight_layout()
    plt.savefig("./fig6.pdf", bbox_inches="tight", pad_inches=0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath_gd", type=str, required=True, help="filepath to where gd sweep stored"
    )
    parser.add_argument(
        "--filepath_no_gd",
        type=str,
        required=True,
        help="filepath to where no gd sweep stored",
    )
    args = parser.parse_args()
    main(args.filepath_gd, args.filepath_no_gd)
