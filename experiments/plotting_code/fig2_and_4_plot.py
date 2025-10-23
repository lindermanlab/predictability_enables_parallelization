"""
Code to plot Figures 2 and 4.
"""
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from layout import create_axis_at_location
from lle import log_mu_numerically_stable

# Use TeX.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Configure font sizes.
INCREASE = 3
SMALL_SIZE = 6 + INCREASE
MEDIUM_SIZE = 8 + INCREASE
BIGGER_SIZE = 10 + INCREASE
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

def main(filepath):
    D = 100
    Zs = []
    all_lles = []
    NUM_SEEDS = 20
    for seed in range(NUM_SEEDS):
        with open(filepath + f"threshold_D_100_when_before_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
        Ts = data["Ts"]
        lles = data["results"][0, :, 0]  # shape (n_g,)
        all_lles.append(lles)
        bounded = np.minimum(data["results"][1], Ts)
        Zs.append(bounded)
    gs = data["gs"]
    Zs_mean = np.mean(Zs, axis=0)
    all_lles = np.array(all_lles)

    median_lles = np.median(all_lles, axis=0)
    min_lles = np.min(all_lles, axis=0)
    max_lles = np.max(all_lles, axis=0)

    #  this is Figure 4
    plt.plot(gs, median_lles, label="Median LLE")
    plt.fill_between(gs, min_lles, max_lles, color="gray", alpha=0.3, label="Min-Max range")
    plt.xlabel("g")
    plt.ylabel("LLE")
    plt.legend()
    plt.axhline(y=0, color="k", linestyle="--")
    plt.title(rf"Median LLE over {NUM_SEEDS} seeds with Min-Max range, $D$={D}")
    plt.savefig("./fig4.pdf", bbox_inches="tight", pad_inches=0.02)

    ###############################################
    # this is Figure 2  
    T_fine = np.linspace(Ts.min(), Ts.max(), 400)
    lambda_fine = np.linspace(median_lles.min(), median_lles.max(), 400)
    Lg, Tg = np.meshgrid(lambda_fine, T_fine)  # swap order
    log_mu_fine = -np.vectorize(log_mu_numerically_stable)(Lg, Tg)

    lle_perm = np.argsort(median_lles)
    Zs_mean_perm = Zs_mean[lle_perm, :]
    median_lles_perm = median_lles[lle_perm]
    interp = RegularGridInterpolator((median_lles_perm, Ts), Zs_mean_perm)

    xg, yg = np.meshgrid(lambda_fine, T_fine)
    Zs_interp = interp(np.column_stack((xg.ravel(), yg.ravel()))).reshape(xg.shape)
    Zs_interp.shape

    T_idx = 33  # 1000
    T = Ts[T_idx]
    lle_ticks = np.array([-0.6, -0.4, -0.2, 0])
    line_color = "#AAAAAA"
    seeds = np.arange(NUM_SEEDS)

    fig = plt.figure(figsize=(5.75, 2))

    # FIRST SUBPLOT - log(mu) heatmap
    ax0 = create_axis_at_location(fig, 0.5, 0.5, 1.0, 1.0)
    pcm1 = ax0.imshow(
        log_mu_fine,
        extent=(lambda_fine.min(), lambda_fine.max(), T_fine.min(), T_fine.max()),
        cmap="coolwarm",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=2800,
    )
    ax0.set_title("Theory\n($-\\log(\\tilde{\\mu})$)")
    ax0.set_xlabel(r"LLE ($\lambda$)")
    ax0.set_ylabel(r"$T$")
    ax0.axvline(x=0, color="k", linestyle=":", lw=1)
    ax0.axhline(y=T, color=line_color, linestyle="-", lw=2)
    ax0.set_xlim(median_lles.min(), median_lles.max())
    ax0.set_ylim(Ts.min(), 10_000)
    ax0.set_xticks(lle_ticks)

    # Create a divider for the axes
    cax0 = create_axis_at_location(fig, 1.55, 0.5, 0.05, 1.0)
    cbar0 = fig.colorbar(pcm1, cax=cax0)
    cbar0.set_ticks(np.linspace(0, 2800, 5))

    # SECOND SUBPLOT - DEER steps heatmap
    ax1 = create_axis_at_location(fig, 2.2, 0.5, 1.0, 1.0)
    pcm2 = ax1.imshow(
        Zs_interp,
        extent=(median_lles.min(), median_lles.max(), Ts.min(), Ts.max()),
        cmap="coolwarm",
        origin="lower",
        aspect="auto",
    )
    ax1.set_title("Experiment\n(steps to convergence)")
    ax1.set_xlabel(r"LLE ($\lambda$)")
    ax1.set_yticklabels([])
    ax1.axvline(x=0, color="k", linestyle=":", lw=1)
    ax1.axhline(y=T, color=line_color, linestyle="-", lw=2)
    ax1.set_xlim(median_lles.min(), median_lles.max())
    ax1.set_ylim(Ts.min(), 10_000)
    ax1.set_xticks(lle_ticks)

    # Create a divider for the axes
    cax1 = create_axis_at_location(fig, 3.25, 0.5, 0.05, 1.0)
    cbar1 = fig.colorbar(pcm2, cax=cax1)
    cbar1.set_ticks(np.linspace(0, 10_000, 5))

    # THIRD SUBPLOT - Scatter plot
    ax2 = create_axis_at_location(fig, 4.3, 0.5, 1.0, 1.0)
    # Extract and plot scatter data
    for seed in seeds:
        with open(filepath + f"threshold_D_100_when_before_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
        # Plot points below or equal to threshold as dots
        ax2.scatter(
            data["results"][0, :, T_idx],
            np.minimum(data["results"][1, :, T_idx], T),
            color=line_color,
            alpha=0.5,
            s=1,
            zorder=3,
        )

    ax2.axvline(x=0, color="k", linestyle=":", lw=1)
    ax2.grid(True, which="major", color="gray", alpha=0.5)
    ax2.minorticks_off()
    ax2.set_xlabel(r"LLE ($\lambda$)")
    ax2.set_xticks(lle_ticks)
    ax2.set_xlim(lambda_fine.min(), lambda_fine.max())
    ax2.set_ylabel("steps to convergence")
    ax2.set_ylim(-25, 1050)
    ax2.set_title(rf"Experiment: $T={Ts[T_idx]}$")
    plt.savefig("./fig2.pdf", bbox_inches="tight", pad_inches=0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True, help="filepath to where data are stored")
    args = parser.parse_args()
    main(args.filepath)
