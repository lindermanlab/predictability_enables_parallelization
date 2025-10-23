"""
tab1.py

Set up a hydra script to run the observer experiments that are depicted in Table 1.
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
import time

from deer import deer_alg
from lle import estimate_lle_and_overshoot

from jaxtyping import Array
import examples.chaotic_flows as cf
from dynamics import FeedbackObserver

@hydra.main(version_base=None, config_path="configs", config_name="tab1")
def main(cfg: DictConfig):
    print("starting")
    seed = cfg.seed  
    system_name = cfg.system_name
    T_max = cfg.T_max  
    dt = cfg.dt
    alg = cfg.alg  # deer, sequential
    observer_flag = cfg.observer_flag  # true or false
    mode = cfg.mode  # online or offline
    nwarmups = cfg.nwarmups
    nreps = cfg.nreps
    tol = cfg.tol

    logger = WandbLogger(project="predictability", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    k1, k2, k3, k4, k5, k6 = jr.split(jr.PRNGKey(seed), 6)

    C = jr.normal(shape=(1, 3), key=k1)
    K = C.T

    # measurement function
    def h(state: Array) -> Array:
        return C @ state

    # chaotic flows and corresponding observer
    rossler = cf.Rossler(a=0.2, b=0.2, c=5.7, dt=dt)
    rossler_observer = FeedbackObserver(
        system=rossler, K=K, h=h, dt=dt, inds_to_replace=[]
    )

    lorenz = cf.Lorenz(beta=8 / 3, rho=28, sigma=10, dt=dt)
    lorenz_observer = FeedbackObserver(
        system=lorenz, K=K, h=h, dt=dt, inds_to_replace=[1]
    )

    thomas = cf.Thomas(a=1.85, b=10, dt=dt)
    thomas_observer = FeedbackObserver(
        system=thomas, K=K, h=h, dt=dt, inds_to_replace=[1]
    )

    abc = cf.ABC(A=1.73205, B=1.41421, C=1, dt=dt)
    abc_observer = FeedbackObserver(system=abc, K=K, h=h, dt=dt, inds_to_replace=[0, 2])

    sprottb = cf.SprottB(dt=dt)
    sprottb_observer = FeedbackObserver(
        system=sprottb, K=K, h=h, dt=dt, inds_to_replace=[1]
    )

    ks = cf.KawczynskiStrizhak(beta=-0.4, gamma=0.49, kappa=0.2, mu=2.1, dt=dt)
    ks_observer = FeedbackObserver(system=ks, K=K, h=h, dt=dt, inds_to_replace=[0, 1])

    elnino = cf.VallisElNino(b=102, c=3, p=0, dt=dt)
    elnino_observer = FeedbackObserver(
        system=elnino, K=K, h=h, dt=dt, inds_to_replace=[1]
    )

    chua = cf.Chua(alpha=15.6, beta=28.0, m0=-1.142857, m1=-0.71429, dt=dt)
    chua_observer = FeedbackObserver(system=chua, K=K, h=h, dt=dt, inds_to_replace=[1])

    nosehoover = cf.NoseHoover(a=1.5, dt=dt)
    nosehoover_observer = FeedbackObserver(
        system=nosehoover, K=K, h=h, dt=dt, inds_to_replace=[0, 2]
    )

    # thesse keys used for estimated the LLE
    master_key = jr.PRNGKey(0)
    keys = jax.random.split(master_key, num=9)

    systems_and_observers = {
        "Rossler": {
            "system": rossler,
            "observer": rossler_observer,
            "C": C,
            "K": K,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e5),
            "ic": [6.5134412, 0.4772013, 0.34164294],
            "key": keys[0],
        },
        "Lorenz": {
            "system": lorenz,
            "observer": lorenz_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(5e4),
            "ic": [-9.7869288, -15.03852, 20.533978],
            "key": keys[1],
        },
        "Thomas": {
            "system": thomas,
            "observer": thomas_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e5),
            "ic": [2.199442, 2.3634225, 3.220197],
            "key": keys[2],
        },
        "ABC": {
            "system": abc,
            "observer": abc_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e6),
            "ic": [-0.78450179, -0.62887672, -0.17620268],
            "key": keys[3],
        },
        "SprottB": {
            "system": sprottb,
            "observer": sprottb_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e5),
            "ic": [1.6878282, 1.4848132, -0.53706628],
            "key": keys[4],
        },
        "Kawczynski-Strizhak": {
            "system": ks,
            "observer": ks_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e5),
            "ic": [-1.160501, 5.8636902, -1.2031947],
            "key": keys[5],
        },
        "Vallis El Nino": {
            "system": elnino,
            "observer": elnino_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(3e4),
            "ic": [-6.99286331487343, 0.023254276622294345, -0.1658177563067799],
            "key": keys[6],
        },
        "Chua": {
            "system": chua,
            "observer": chua_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(5e4),
            "ic": [1.1262351, -0.11386474, -0.78324421],
            "key": keys[7],
        },
        # NH is Nose-Hoover Thermostat
        "NH": {
            "system": nosehoover,
            "observer": nosehoover_observer,
            "C": None,
            "K": None,
            "h": h,
            "T_max": T_max,
            "time2plot": int(1e5),
            "ic": [-1.9758582, 0.24647527, 0.79024299],
            "key": keys[8],
        },
    }

    print(f"\nProcessing {system_name} system...")
    entry = systems_and_observers[system_name]  # still a dictionary
    true_system = entry["system"]
    observer = entry["observer"]
    key = entry["key"]

    x0 = jnp.array(entry["ic"])

    inputs = jnp.zeros(shape=(T_max, 3))  # zero inputs for API consistency
    # DEER initial guess
    states_guess = jax.random.normal(k3, shape=(T_max, 3))

    print(f"\nPerforming sequential rollout of system...")
    _, xs = jax.lax.scan(lambda c, a: true_system.scan_fxn(c, a), x0, inputs)
    xs_with_x0 = jnp.concatenate((jnp.expand_dims(x0, axis=0), xs), axis=0)
    perturbation = jr.normal(shape=(3), key=key)
    x0_observer = x0 + perturbation  # perturbed initial state
    _, xhats = jax.lax.scan(
        lambda c, a: observer.scan_fxn(c, a), x0_observer, xs_with_x0[:-1]
    )
    xhats_with_x0 = jnp.concatenate(
        (jnp.expand_dims(x0_observer, axis=0), xhats), axis=0
    )

    if observer_flag:
        print("Using observer")
        system = observer
        x0 = x0_observer  # perturb the initial condition for the observer
        deer_inputs = xs_with_x0[:-1]
        Jhats = jax.vmap(jax.jacobian(observer.deer_fxn), in_axes=(0, 0))(
            xhats_with_x0[:-1], xs_with_x0[:-1]
        )
        lle, a = estimate_lle_and_overshoot(Jhats, key, dt)
        true_xs = xhats
    else:
        print("Using true system")
        system = true_system
        deer_inputs = inputs
        Js = jax.vmap(jax.jacobian(system.deer_fxn), in_axes=(0, 0))(
            xs[:-1], inputs[1:]
        )
        lle, _ = estimate_lle_and_overshoot(Js, key, dt)
        true_xs = xs

    # set the algorithm
    def seq_eval(inputs):
        _, xs = jax.lax.scan(lambda c, a: system.scan_fxn(c, a), x0, inputs)
        return xs[-1]

    def deer_eval(inputs):
        _, final_state_deer, newton_steps, *_ = deer_alg(
            system.deer_fxn,
            x0,
            states_guess,
            deer_inputs,
            num_iters=T_max,
            full_trace=False,
            Ts=None,
            tol=tol,
        )
        return newton_steps

    def quasi_eval(inputs):
        _, final_state_quasi, newton_steps, *_ = deer_alg(
            system.deer_fxn,
            x0,
            states_guess,
            deer_inputs,
            num_iters=T_max,
            quasi=True,
            full_trace=False,
            Ts=None,
            tol=tol,
        )
        return newton_steps

    # jit and compute and record error
    if alg == "seq":
        fxn = jax.jit(seq_eval)
        fxn.lower(inputs).compile()
        _, xs = jax.lax.scan(lambda c, a: system.scan_fxn(c, a), x0, inputs)
        alg_xs = xs
    elif alg == "deer":
        fxn = jax.jit(deer_eval)
        fxn.lower(inputs).compile()
        _, final_state_deer, newton_steps, *_ = deer_alg(
            system.deer_fxn,
            x0,
            states_guess,
            deer_inputs,
            num_iters=T_max,
            full_trace=False,
            tol=tol,
            Ts=None,
        )
        alg_xs = final_state_deer
    elif alg == "quasi":
        fxn = jax.jit(quasi_eval)
        fxn.lower(inputs).compile()
        _, final_state_quasi, newton_steps, *_ = deer_alg(
            system.deer_fxn,
            x0,
            states_guess,
            deer_inputs,
            num_iters=T_max,
            quasi=True,
            full_trace=False,
            tol=tol,
            Ts=None,
        )
        alg_xs = final_state_quasi
    else:
        raise ValueError(f"Invalid algorithm: {alg}")

    # compute error
    err = jnp.linalg.norm(true_xs - alg_xs, axis=-1)

    # warmup
    for _ in range(nwarmups):
        x1 = fxn(inputs)
        jax.block_until_ready(x1)
    t0 = time.time()
    # timing
    for _ in range(nreps):
        x1 = fxn(inputs)
        jax.block_until_ready(x1)
    t1 = time.time()
    elapsed_time = (t1 - t0) / nreps
    print(f"Average time for {alg} algorithm: {elapsed_time:.3e} seconds")

    if alg == "seq":
        n_iters = T_max
    else:
        n_iters = jnp.mean(x1)
    print(f"{alg} n_iters: {n_iters}")

    results = {
        "time": elapsed_time,
        "n_iters": n_iters,
        "lle": lle,
        "err": jnp.mean(jnp.abs(err)),
    }

    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
