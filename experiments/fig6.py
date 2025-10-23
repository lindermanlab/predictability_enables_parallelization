"""
fig6.py

Single T for the the mean field RNN experiment (Figures 2 and 6 in our paper)

This script takes in different algorithms (sequential, DEER, quasi-Newton, gradient descent)
And so is used in particular for generating the data behind Figure 6 in our paper.

Timing carried out on an H100 with 80GB VRAM.

Notes:
- there is no batch size (i.e. batch size is 1)
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import wandb
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import time
from functools import partial

from examples.dmf_rnn import DiscretizedMeanFieldRNN
from deer import deer_alg, explicit_GD
from lle import (
    estimate_lle_and_overshoot
)


@hydra.main(version_base=None, config_path="configs", config_name="fig6")
def main(cfg: DictConfig):
    seed = cfg.seed
    T = cfg.T
    T_min = cfg.T_min
    num_Ts = cfg.num_Ts
    D = cfg.D
    mode = cfg.mode
    g = cfg.g
    alg = cfg.alg
    gd_step_size = cfg.gd_step_size
    tol = cfg.tol

    nonlin = jnp.tanh
    self_couple = False
    Ts = jnp.logspace(jnp.log10(T_min), jnp.log10(T), num_Ts).astype(int)

    logger = WandbLogger(project="predictability", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    k1, k2, k3, k4, k5, k6, k7 = jr.split(jr.PRNGKey(seed), 7)
    A = jr.uniform(k1, shape=(D,), minval=0, maxval=0.2)
    omega = jr.uniform(k2, shape=(D,), minval=0.5, maxval=3.0)
    phi = jr.uniform(k3, shape=(D,), minval=0, maxval=2 * jnp.pi)
    x0 = jr.uniform(k4, shape=(D,), minval=-1.0, maxval=1.0)

    def make_input(t):
        u_t = A * jnp.sin(omega * t + phi)
        return u_t

    inputs = jax.vmap(make_input)(jnp.arange(T))

    dmf_rnn = DiscretizedMeanFieldRNN(
        g=g, D=D, phi=nonlin, self_couple=self_couple, key=k5
    )
    states_guess = jr.uniform(k6, shape=(T, D), minval=-1.0, maxval=1.0)

    # set the algs
    def seq_eval(inputs):
        _, xs = jax.lax.scan(
            lambda c, a: dmf_rnn.scan_fxn(c, a), x0, inputs[: cfg.T]
        )
        return xs[-1]

    def deer_eval(inputs):
        _, final_state_deer, newton_steps, *_ = deer_alg(
            dmf_rnn.deer_fxn,
            x0,
            states_guess,
            inputs,
            num_iters=cfg.T,
            full_trace=False,
            tol=tol,
        )
        return newton_steps

    def quasi_eval(inputs):
        _, final_state_quasi, newton_steps, *_ = deer_alg(
            dmf_rnn.deer_fxn,
            x0,
            states_guess,
            inputs,
            num_iters=cfg.T,
            quasi=True,
            full_trace=False,
            tol=tol,
        )
        return newton_steps

    def gd_eval(inputs):
        *_, num_gd_steps = explicit_GD(
            dmf_rnn.deer_fxn,
            x0,
            states_guess,
            inputs,
            num_iters=cfg.T,
            alpha=gd_step_size,
            full_trace=False,
            tol=tol,
        )
        return num_gd_steps

    # compute lle
    _, seq_states = jax.lax.scan(lambda c, a: dmf_rnn.scan_fxn(c, a), x0, inputs)
    Js = jax.vmap(jax.jacobian(dmf_rnn.deer_fxn), in_axes=(0, 0))(
        seq_states[:-1], inputs[1:]
    )
    lle, _ = estimate_lle_and_overshoot(Js, k7, 1)

    # jit
    if alg == "seq":
        fxn = jax.jit(seq_eval)
        fxn.lower(inputs).compile()
    elif alg == "deer":
        fxn = jax.jit(deer_eval)
        fxn.lower(inputs).compile()
    elif alg == "quasi":
        fxn = jax.jit(quasi_eval)
        fxn.lower(inputs).compile()
    elif alg == "gd":
        fxn = jax.jit(gd_eval)
        fxn.lower(inputs).compile()
    else:
        raise ValueError(f"Invalid algorithm: {alg}")

    # warmup
    for _ in range(cfg.nwarmups):
        x1 = fxn(inputs)
        jax.block_until_ready(x1)
    t0 = time.time()
    # timing
    for _ in range(cfg.nreps):
        x1 = fxn(inputs)
        jax.block_until_ready(x1)
    t1 = time.time()
    elapsed_time = (t1 - t0) / cfg.nreps
    print(f"Average time for {alg} algorithm: {elapsed_time:.3e} seconds")

    if alg == "seq":
        n_iters = T
    else:
        n_iters = jnp.mean(x1)
    print(f"{alg} n_iters: {n_iters}")

    results = {
        "time": elapsed_time,
        "n_iters": n_iters,
        "lle": lle,
    }

    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
