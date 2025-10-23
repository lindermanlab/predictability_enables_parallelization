"""
fig2.py

Gives the data behind Figure 2 in our paper
Experiment to look at a discretized MF RNN and track both LLE and number of DEER steps to convergence as a function of g
(mean field RNN based on: https://arxiv.org/abs/2006.02427)

We run in float64 to focus (as much as possible) on the interplay between dynamics and optimization.
We leave interesing questions about numerical precision for future work.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import wandb
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import lightning as L

import matplotlib.pyplot as plt

from examples.dmf_rnn import DiscretizedMeanFieldRNN
from deer import deer_alg
from lle import  wrapper_estimate_lle_from_jacobians

@hydra.main(version_base=None, config_path="configs", config_name="fig2")
def main(cfg: DictConfig):
    # import from the cfg
    seed = cfg.seed
    D = cfg.D
    nonlin_str = cfg.nonlin_str
    num_gs = cfg.num_gs
    mode = cfg.mode
    g_min = cfg.g_min
    g_max = cfg.g_max
    # sweep over Ts
    T_require = 1000
    T_min = jnp.minimum(cfg.T_min, T_require)
    T_max = jnp.maximum(cfg.T_max, T_require)
    num_Ts = cfg.num_Ts
    when = cfg.when
    tol = cfg.tol

    if nonlin_str == "tanh":
        nonlin = jnp.tanh
        self_couple = False
    elif nonlin_str == "relu":
        nonlin = jax.nn.relu
        self_couple = True

    logger = WandbLogger(project="predictability", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    Ts = jnp.logspace(jnp.log10(T_min), jnp.log10(T_max), num_Ts).astype(int)
    Ts = jnp.append(Ts, T_require)
    Ts = jnp.unique(Ts)  # ensures no duplicates
    Ts = jnp.sort(Ts)    # ensures correct ordering
    gs = jnp.linspace(g_min, g_max, num_gs)
    k1, k2, k3, k4, k5, k6, k7 = jr.split(jr.PRNGKey(seed), 7)
    A = jr.uniform(k1, shape=(D,), minval=0, maxval=0.2)
    omega = jr.uniform(k2, shape=(D,), minval=0.5, maxval=3.0)
    phi = jr.uniform(k3, shape=(D,), minval=0, maxval=2 * jnp.pi)
    x0 = jr.uniform(k4, shape=(D,), minval=-1., maxval=1.)
    def make_input(t):
        u_t = A * jnp.sin(omega * t + phi)
        return u_t
    inputs = jax.vmap(make_input)(jnp.arange(T_max)) # inputs to the system

    # build tensor to hold results
    results = np.zeros(
        (4, len(gs), len(Ts))
    )  # results[0] is LLE, results[1] is num_deer_steps, results[2] is max spectral norm, results[3] is average log spectral norm

    for g_idx, g in enumerate(gs):
        dmf_rnn = DiscretizedMeanFieldRNN(
            g=g,
            D=D,
            phi=nonlin,
            self_couple=self_couple,
            when=when,
            key=k5
        )
        states_guess = jr.uniform(k6, shape=(T_max, D), minval=-1., maxval=1.) # we've decided to initialize randomly
        all_states, final_state, newton_steps, _, is_nan, _, all_newtons, _ = (
            deer_alg(
                dmf_rnn.deer_fxn,
                x0, # initial state
                states_guess, # states guess
                inputs,
                num_iters=int(T_max),
                full_trace=False,
                Ts=Ts,
                tol=tol,
            )
        )

        _, seq_states = jax.lax.scan(
            lambda c, a: dmf_rnn.scan_fxn(c,a), x0, inputs)

        Js = jax.vmap(jax.jacobian(dmf_rnn.deer_fxn), in_axes=(0, 0))(
            seq_states[:-1], inputs[1:] 
        ) # compute rho_eff based on the sequential scan
        Js_norms = jax.vmap(lambda J: jnp.linalg.norm(J, ord=2))(Js) # get spectral norms of the Jacobians
        Js_log_norms = jnp.log(Js_norms)
        Js_maxes = jax.lax.cummax(Js_norms)[Ts] # now length 
        Js_log_norm_means = Js_log_norms[Ts] / (Ts + 1) # now length Ts

        lle_avgd = wrapper_estimate_lle_from_jacobians(Js, k7, Ts=Ts)
        print(f"when g ={g}, are nans encountered? {is_nan}, lle={lle_avgd}, num_deer_steps={newton_steps}")
        print()
        results[0, g_idx, :] = lle_avgd
        results[1, g_idx, :] = all_newtons
        results[2, g_idx, :] = Js_maxes
        results[3, g_idx, :] = Js_log_norm_means

    fig1, axs = plt.subplots(4, 1, figsize=(10, 8))
    # num deer steps
    axs[0].plot(gs, results[1, :, -1])
    axs[0].set_xlabel(r"$g$")
    axs[0].set_ylabel(f"T={T_max}")
    axs[0].set_xscale("log")

    axs[1].plot(gs, results[1, :, -1])
    axs[1].set_xlabel(r"$g$")
    axs[1].set_ylabel(f"T={T_max}")

    axs[2].plot(Ts, results[1, 0])
    axs[2].set_xlabel("T")
    axs[2].set_ylabel(f"g={gs[0]}")

    axs[3].plot(Ts, results[1, -1])
    axs[3].set_xlabel("T")
    axs[3].set_ylabel(f"g={gs[-1]}")
    plt.suptitle(f"num deer steps, D={D}, seed={seed}, when={when}")
    plt.tight_layout()
    plt.savefig(f"num_deer_steps_D_{D}_seed_{seed}_when_{when}.pdf")
    wandb.log({"num_deer_steps_plot": wandb.Image(fig1)})

    # LLE
    fig2, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(gs, results[0, :, -1])
    axs[0].axhline(0, color="k", linestyle="--")
    axs[0].set_xlabel(r"$g$")
    axs[0].set_ylabel(f" T={T_max}")
    axs[0].set_yscale("symlog")
    axs[0].set_xscale("log")

    axs[1].plot(gs, results[0, :, -1])
    axs[1].axhline(0, color="k", linestyle="--")
    axs[1].set_xlabel(r"$g$")
    axs[1].set_ylabel(f" T={T_max}")
    axs[1].set_yscale("symlog")

    axs[2].plot(Ts, results[0, 0])
    axs[2].axhline(0, color="k", linestyle="--")
    axs[2].set_xlabel("T")
    axs[2].set_ylabel(f"g={gs[0]}")
    axs[2].set_yscale("symlog")

    axs[3].plot(Ts, results[0, -1])
    axs[3].axhline(0, color="k", linestyle="--")
    axs[3].set_xlabel("T")
    axs[3].set_ylabel(f"g={round(gs[-1])}")
    axs[3].set_yscale("symlog")

    plt.suptitle(f"LLE, D={D}, seed={seed}, when={when}")
    plt.tight_layout()
    plt.savefig(f"LLE_D_{D}_seed_{seed}_when_{when}.pdf")
    wandb.log({"LLE_plot": wandb.Image(fig2)})

    # second figure to plot num deer steps vs lle
    fig3 = plt.figure()
    plt.plot(results[0, :, -1].flatten(), results[1, :, -1].flatten(), "o")
    plt.xlabel("LLE")
    plt.ylabel("Number of Deer Steps")
    plt.axvline(x=0.0, color="k", linestyle="--")

    plt.title(f"NumDEER_vs_LLE_seed={seed}_D={D}_when={when}")
    plt.savefig(f"NumDEER_vs_LLE_seed={seed}_D={D}_when={when}.pdf")
    wandb.log({"num_DEER_steps_vs_lle": wandb.Image(fig3)})

    fig4 = plt.figure()
    plt.plot(results[2, :, -1].flatten(), results[1, :, -1].flatten(), "o")
    plt.xlabel("Max Spectral Norm")
    plt.ylabel("Number of Deer Steps")
    plt.axvline(x=1., color="k", linestyle="--")

    plt.title(f"NumDEER_vs_MaxSpecNorm_seed={seed}_D={D}_when={when}")
    plt.savefig(f"NumDEER_vs_MaxSpecNorm_seed={seed}_D={D}_when={when}.pdf")
    wandb.log({"num_DEER_steps_vs_MaxSpecNorm": wandb.Image(fig4)})

    fig5 = plt.figure()
    plt.plot(results[3, :, -1].flatten(), results[1, :, -1].flatten(), "o")
    plt.xlabel("Average Log Spectral Norm")
    plt.ylabel("Number of Deer Steps")
    plt.axvline(x=0.0, color="k", linestyle="--")

    plt.title(f"NumDEER_vs_AvgLogSpecNorm_seed={seed}_D={D}_when={when}")
    plt.savefig(f"NumDEER_vs_AvgLogSpecNorm_seed={seed}_D={D}_when={when}.pdf")
    wandb.log({"num_DEER_steps_vs_AvgLogSpecNorm": wandb.Image(fig5)})
    # pickle and log the results
    results_dict = {
        "results": results,
        "gs": np.array(gs),
        "Ts": np.array(Ts),
    }
    pkl_file_name = f"threshold_D_{D}_when_{when}_seed_{seed}.pkl"
    with open(pkl_file_name, "wb") as f:
        pickle.dump(results_dict, f)
    wandb.save(pkl_file_name)
    wandb.finish()

if __name__=="__main__":
    main()
