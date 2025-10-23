"""
Code to plots Figures 3 and 7
"""
import argparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pickle

from lle import get_spectral_norm, wrapper_estimate_lle_from_jacobians
from examples.two_well import TwoWell

# Use TeX.
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Configure font sizes.
SMALL_SIZE = 7 + 1
MEDIUM_SIZE = 8 + 1
BIGGER_SIZE = 10 + 1
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

def pad_to_length_jax(arr, target_len):
    pad_width = target_len - arr.shape[0]
    return jnp.pad(arr, (0, pad_width), mode="edge")

def main(filepath):
    all_steps, all_lle, all_merit = [], [], []

    # First pass to determine max length
    max_len = 0
    for seed in range(20):
        with open(filepath + f"fig3_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
            max_len = max(max_len, len(data["lle"]), len(data["merit"]))

    # Second pass: collect and pad everything
    for seed in range(20):
        with open(filepath + f"fig3_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
            all_steps.append(data["num_deer_steps"])  # Already uniform length
            all_lle.append(pad_to_length_jax(data["lle"], max_len))
            all_merit.append(pad_to_length_jax(data["merit"], max_len))

    all_lle = np.array(all_lle)
    all_merit = np.array(all_merit)

    # Compute LLE stats
    lle_median = np.median(all_lle, axis=0)
    lle_min = np.min(all_lle, axis=0)
    lle_max = np.max(all_lle, axis=0)

    # Compute merit stats
    mf_median = np.median(all_merit, axis=0)
    mf_min = np.min(all_merit, axis=0)
    mf_max = np.max(all_merit, axis=0)

    # Compute steps vs T stats
    all_steps = np.array(all_steps)  # Should be rectangular already
    Ts = data["Ts"]  # Same across all seeds
    step_median = np.median(all_steps, axis=0)
    step_min = np.min(all_steps, axis=0)
    step_max = np.max(all_steps, axis=0)

    experiment = TwoWell()
    seed = 7
    epsilon = 0.01
    D = experiment.D
    T = 10_000
    initial_state = jnp.zeros((D,))
    k1, k2, k3 = jr.split(jr.PRNGKey(seed), 3)
    noise_scale = 1.0
    inputs = noise_scale * jr.normal(k1, (T, D))
    _, true_states = jax.lax.scan(
        lambda c, a: experiment.scan_fxn(c, a), initial_state, inputs
    )
    Js = jax.vmap(jax.jacobian(experiment.deer_fxn), in_axes=(0, 0))(
        true_states[:-1], inputs[1:]
    )
    lle = (
        wrapper_estimate_lle_from_jacobians(Js, k3) / epsilon
    )  # go from discrete to continuous
    print(f"LLE: {lle:.3g}")

    # plot for Figure 3
    # settings constants for the colorbars
    VMIN = 0.93
    VMAX = 1.07
    fig = plt.figure(figsize=(7, 2), layout="constrained")

    gs = fig.add_gridspec(1, 3)

    # Create axes with specific sharing
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)  # Share y-axis with first plot
    ax2 = fig.add_subplot(gs[0, 2], sharex=ax1)  # Share x-axis with second plot
    axs = [ax0, ax1, ax2]

    # axs[0]: Jacobian and spectral norm
    ## Let's make a heatmap of the spectral radius of the jacobian at different points
    # Create a grid of x1 and x2 values
    x_vals = jnp.linspace(-3, 3, 100)  # Adjusted range for better visualization
    y_vals = jnp.linspace(-2, 2, 100)
    X, Y = jnp.meshgrid(x_vals, y_vals)

    # Stack and reshape for vectorized evaluation
    grid_points = jnp.column_stack([X.ravel(), Y.ravel()])  # Shape: (10000, 2)

    # Compute logp values efficiently
    Z = jax.vmap(experiment.logp)(grid_points).reshape(
        100, 100
    )  # Reshape back to (100, 100)
    Z2 = jax.vmap(get_spectral_norm, in_axes=(0, None, None))(
        grid_points, jnp.zeros((D,)), experiment.deer_fxn
    ).reshape(100, 100)

    # Plot the contour
    axs[0].imshow(
        Z2.T, extent=(-2, 2, -3, 3), origin="lower", cmap="RdBu_r", vmin=VMIN, vmax=VMAX
    )  # spectral norm of Jacobian
    axs[0].contour(
        Y, X, Z, levels=15, colors="black", alpha=0.5
    )  # contour of the potential

    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-3, 3)

    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(r"Two-well state space")

    # Calculate spectral norm values
    xmin = np.min(true_states[:, 0])
    xmax = np.max(true_states[:, 0])
    xs = np.linspace(xmin, xmax, 100)
    color_values = jax.vmap(get_spectral_norm, in_axes=(0, None, None))(
        np.vstack([xs, np.zeros_like(xs)]).T, jnp.zeros((D,)), experiment.deer_fxn
    )

    # Create a color mesh that aligns with plot coordinates
    T = len(true_states[:, 0])
    times = np.arange(T)
    X, Y = np.meshgrid(times, xs)
    axs[1].pcolormesh(
        X,
        Y,
        np.tile(color_values[:, None], T),
        shading="auto",
        cmap="RdBu_r",
        vmin=VMIN,
        vmax=VMAX,
    )

    # Add the line plot on top
    axs[1].plot(times, true_states[:, 0], color="black")

    # Add labels and colorbar
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("y")
    axs[1].set_title("Y coordinate over time")
    axs[1].set_xlim(0, T)
    # Create a color mesh that aligns with plot coordinates
    T = len(true_states[:, 0])
    times = np.arange(T)
    X, Y = np.meshgrid(times, xs)
    pc = axs[1].pcolormesh(
        X,
        Y,
        np.tile(color_values[:, None], T),
        shading="auto",
        cmap="RdBu_r",
        vmin=VMIN,
        vmax=VMAX,
        rasterized=True,
    )
    cbar = fig.colorbar(pc, ax=axs[1], location="right", label=r"Spectral Norm of $J(s)$")

    # axs[2]: scaling of convergence
    axs[2].plot(Ts, step_median, label="Median", color="tab:blue")
    axs[2].set_xlabel(r"Sequence Length $T$")
    axs[2].set_ylabel("Number of DEER steps")
    axs[2].set_title("Steps needed for convergence")

    # Remove redundant y-axis labels for second plot
    axs[1].set_ylabel(
        ""
    )  # Remove y label for the middle plot since it's shared with the first
    plt.setp(axs[1].get_yticklabels(), visible=False)  # Optionally hide tick labels too
    plt.savefig("./fig3.pdf", bbox_inches="tight")

    ###################################################
    # this is Figure 7
    all_steps, all_lle, all_merit = [], [], []

    # First pass to determine max length
    max_len = 0
    for seed in range(20):
        with open(filepath + f"fig3_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
            max_len = max(max_len, len(data["lle"]), len(data["merit"]))

    # Second pass: collect and pad everything
    for seed in range(20):
        with open(filepath + f"fig3_seed_{seed}.pkl", "rb") as f:
            data = pickle.load(f)
            all_steps.append(data["num_deer_steps"])  # Already uniform length
            all_lle.append(pad_to_length_jax(data["lle"], max_len))
            all_merit.append(pad_to_length_jax(data["merit"], max_len))

    all_lle = np.array(all_lle)
    all_merit = np.array(all_merit)

    # Compute LLE stats
    lle_median = np.median(all_lle, axis=0)
    lle_min = np.min(all_lle, axis=0)
    lle_max = np.max(all_lle, axis=0)

    # Compute merit stats
    mf_median = np.median(all_merit, axis=0)
    mf_min = np.min(all_merit, axis=0)
    mf_max = np.max(all_merit, axis=0)

    # Compute steps vs T stats
    all_steps = np.array(all_steps)  # Should be rectangular already
    Ts = data["Ts"]  # Same across all seeds
    step_median = np.median(all_steps, axis=0)
    step_min = np.min(all_steps, axis=0)
    step_max = np.max(all_steps, axis=0)

    # Plot all 3
    ts = np.arange(max_len)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1), sharex=False, constrained_layout=True)

    # LLE vs iteration
    axs[0].plot(ts, lle_median, color="black", label="Median")
    axs[0].fill_between(ts, lle_min, lle_max, alpha=0.3, color="black", label="Min-Max")
    axs[0].axhline(0, color="k", linestyle="--")
    axs[0].set_xlabel("DEER Iteration")
    axs[0].set_ylabel("LLE")
    axs[0].set_title("LLE over Iterations")

    # Merit vs iteration
    axs[1].plot(ts, mf_median, color="black", label="Median")
    axs[1].fill_between(ts, mf_min, mf_max, alpha=0.3, color="black", label="Min-Max")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("DEER Iteration")
    axs[1].set_ylabel("Merit function")
    axs[1].set_title("Merit Function")

    # T vs steps
    axs[2].plot(Ts, step_median, label="Median", color="black")
    axs[2].fill_between(Ts, step_min, step_max, alpha=0.3, color="black", label="Min-Max")
    axs[2].set_xlabel(r"Sequence Length $T$")
    axs[2].set_ylabel("Number of DEER steps")
    axs[2].set_title("Steps to convergence")

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="black", label="Median"),
            plt.Line2D(
                [0], [0], color="black", alpha=0.3, linewidth=10, label="Min-Max (20 seeds)"
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.1),
    )
    plt.tight_layout()
    plt.savefig("fig7.pdf", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath", type=str, required=True, help="filepath to where data are stored"
    )
    args = parser.parse_args()
    main(args.filepath)
