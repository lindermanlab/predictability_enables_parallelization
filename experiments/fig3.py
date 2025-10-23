"""
fig3.py
Run the 2-well experiment

Information that I want:
- for many values of T, how many DEER steps to converge?
For the largest T
- what was the LLE at every intermediate step?
- what was the merit function at every intermediate step?
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import lightning as L

import matplotlib.pyplot as plt
import pickle
import pdb

from examples.two_well import TwoWell
from deer import deer_alg, merit_fxn

@hydra.main(version_base=None, config_path="configs", config_name="fig3")
def main(cfg: DictConfig):
    seed = cfg.seed
    T_min = cfg.T_min  # minimum time steps
    T_max = cfg.T_max  # maximum time steps
    num_Ts = cfg.num_Ts
    tol = cfg.tol  # DEER tolerance
    Ts = jnp.linspace(T_min, T_max, num_Ts).astype(int)
    mode = cfg.mode
    k1, k2 = jr.split(jr.PRNGKey(seed), 2)

    logger = WandbLogger(project="predictability", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    # set up the experiment
    experiment = TwoWell()
    D = experiment.D
    initial_state = jnp.zeros((D,))
    states_guess = jr.normal(k1, (T_max, D))
    inputs = jr.normal(k2, (T_max, D))
    dt = experiment.epsilon  # discretization of system (step size) 

    _, final_state, *_, all_newtons, _ = deer_alg(
        experiment.deer_fxn,
        initial_state,
        states_guess,
        inputs,
        num_iters=T_max,
        full_trace=False,
        Ts=Ts,
        tol=tol,
    )
    # final_state is an array with shape (T_max, D) [the converged DEER trajectory]
    # all_newtons is an array with shape (num_Ts,) [number of DEER steps for each T in Ts]
    # check sequential against final state
    _, seq_states = jax.lax.scan(
        lambda c, a: experiment.scan_fxn(c, a), initial_state, inputs
    )
    print("Number of DEER steps for each T:", all_newtons)
    assert jnp.allclose(
        final_state, seq_states, atol=1e-5
    ), f"Final state does not match sequential state! Max diff is {jnp.max(jnp.abs(final_state - seq_states)):.3g}. Merit fxn is {merit_fxn(experiment.deer_fxn, initial_state, final_state, inputs):.3g}"

    fig1 = plt.figure()
    plt.plot(Ts, all_newtons)
    plt.xlabel("T")
    plt.ylabel("Number of DEER steps")
    plt.title(f"seed {seed}")
    wandb.log({"num_deer_steps": wandb.Image(fig1)})
    plt.show()

    # now for the longest T, get the intermediate LLEs and merit function values
    _, _, _, lles, _, _, _, mf_vals = deer_alg(
        experiment.deer_fxn,
        initial_state,
        states_guess,
        inputs,
        all_newtons[-1],
        get_lles=True,
        full_trace=True,
        tol=tol,
    )
    lles = lles / dt  # convert to continuous time LLEs

    fig2, axs2 = plt.subplots(1, 2)
    axs2[0].plot(jnp.arange(len(lles)), lles)
    axs2[0].set_xlabel("DEER Iteration")
    axs2[0].set_ylabel("LLE")
    axs2[0].axhline(0, color="k", linestyle="--")

    axs2[1].plot(jnp.arange(len(mf_vals)), mf_vals)
    axs2[1].set_yscale("log")
    axs2[1].set_xlabel("DEER Iteration")
    axs2[1].set_ylabel("Merit function")
    plt.suptitle(f"seed {seed}")
    plt.tight_layout()
    wandb.log({"lle and merit": wandb.Image(fig2)})
    plt.show()

    # pickle and log the results
    results_dict = {
        "num_deer_steps": all_newtons,
        "lle": lles,
        "merit": mf_vals,
        "Ts": Ts,
    }

    pkl_file_name = f"fig3_seed_{seed}.pkl"
    with open(pkl_file_name, "wb") as f:
        pickle.dump(results_dict, f)
    wandb.save(pkl_file_name)
    wandb.finish()


if __name__ == "__main__":
    main()
