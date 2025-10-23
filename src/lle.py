"""
lle.py
This file contains helper functions to estimate the largest Lyapunov exponent (LLE) of a dynamical system.
We hope they are broadly useful for assessing predictability of dynamical systems.

Note that the functions:
- estimate_lle_from_vector
- estimate_lle_from_jacobians
- wrapper_estimate_lle_from_jacobians
All work for discrete-time systems (i.e. dt=1)

Note that the LLE can be defined for both discrete-time and continuous-time systems.
If we take a continuous-time system and discretize it with time step dt, then the LLE of the continuous-time system is given by:
LLE_continuous = LLE_discrete / dt
where LLE_discrete is the LLE computed from the discretized system using the three functions above.

Note that the function:
- estimate_lle_and_overshoot
Works for both discrete-time and continuous-time systems, as it takes in a time scale dt as an argument.
"""

import jax
import jax.random as jr
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp


def estimate_lle_from_vector(jacobians, v, Ts=None):
    """
    Estimate the (max) LLE given a sequence of Jacobians and an initial vector v.
    If ‖J v‖==0 at any step the running sum gets −∞, but the code stays NaN-free.

    Called by estimate_lle_from_jacobians and wrapper_estimate_lle_from_jacobians.
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = v / jnp.linalg.norm(v)  # normalise once

    def body_fn(carry, J):
        v, logsum = carry
        v = J[0, 0] * v if D == 1 else J @ v  # advance tangent vec
        n = jnp.linalg.norm(v)

        # safe update: if n==0 ⇒ add -inf, keep v unchanged
        logsum += lax.cond(
            n == 0.0,
            lambda _: -jnp.inf,  # branch taken when norm is zero
            lambda n_: jnp.log(n_),  # otherwise log(n)
            n,
        )
        v = lax.cond(
            n == 0.0,
            lambda _: v,  # leave v as-is (or set to zeros_like(v))
            lambda n_: v / n_,  # renormalise
            n,
        )
        return (v, logsum), logsum

    (v_final, lyap_sum), step_logs = lax.scan(body_fn, (v, 0.0), jacobians)

    if Ts is not None:
        # vectorised query at the requested time indices Ts
        return step_logs[Ts] / (Ts + 1)
    return lyap_sum / jacobians.shape[0]


def estimate_lle_from_jacobians(jacobians, key, Ts=None):
    """
    Estimate the largest Lyapunov exponent (lambda) from a collection of Jacobian matrices.
    **for a specific random key**. Does so by calling estimate_lle_from_vector with a random initial vector.

    Args:
        jacobians: jnp.ndarray of shape (..., D, D) or (...,) if D = 1
        key: jax.random.PRNGKey
        Ts: optional
                      if None, then only returns LLE for the whole sequence
                      ow is an array of Ts (ints), and returns the LLE for each T in Ts
                        has shape (1,len(Ts))

    Returns:
        jnp.ndarray: Estimated lambda
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = jr.uniform(key, (D,), minval=-1.0, maxval=1.0)
    return estimate_lle_from_vector(jacobians, v, Ts=Ts)


def wrapper_estimate_lle_from_jacobians(jacobians, key, Ts=None, numkeys=3):
    """
    Wrapper function to estimate the largest Lyapunov exponent (lambda) from a collection of Jacobian matrices.
    **averaged over numkeys random keys**

    Calls estimate_lle_from_jacobians

    Args:
        jacobians: jnp.ndarray of shape (..., D, D) or (...,) if D = 1
        key: jax.random.PRNGKey
        Ts: optional
                      if None, then only returns LLE for the whole sequence
                      ow is an array of Ts (ints), and returns the LLE for each T in Ts

    Returns:
        jnp.ndarray: Estimated lambda
            Returns correct shape
                scalar if Ts is None
                (1,len(Ts)) if Ts is not None (this should be the shape of Ts as well)
    """
    keys = jr.split(key, numkeys)
    lle_fn = lambda k: estimate_lle_from_jacobians(jacobians, k, Ts)
    lles = jax.vmap(lle_fn)(keys)
    return jnp.mean(lles, axis=0)


def estimate_rho_from_jacobians(jacobians, key):
    """
    Estimate rho_eff from a collection of Jacobian matrices.

    Args:
        jacobians: jnp.ndarray of shape (..., D, D) or (...,) if D = 1
        key: jax.random.PRNGKey

    Returns:
        jnp.ndarray: Estimated rho value
    """
    lle = estimate_lle_from_jacobians(jacobians, key)
    return jnp.exp(lle)


def log_mu_numerically_stable(LLE, T, a=1.0):
    x = LLE * jnp.arange(T)
    return -2 * (jnp.log(1 / a) + logsumexp(x))  # uses the geometric series


def log_mu_from_jacobians(jacobians, key):
    lypaunov_sum = estimate_lle_from_jacobians(jacobians, key)
    return log_mu_numerically_stable(lypaunov_sum, jacobians.shape[0])


def estimate_lle_and_overshoot(jacobians, key, dt, Ts=None):
    """
    Estimate the (max) LLE and overshoot constant 'a' given a sequence of Jacobians.
    Uses a random unit-norm initial vector v.

    Args:
        jacobians: array of shape (T, D, D) or (T,) for scalar case
        key: PRNG key for sampling the initial direction
        dt: time scale of discretization. The LLE of a continuous time system is LLE_discrete / dt, where LLE_discrete is the discrete-time LLE of the discretized system
        Ts: optional array of time indices to return LLE estimates at

    Returns:
        If Ts is None:
            lle: scalar (mean LLE over full trajectory)
            a: scalar (overshoot constant)
        If Ts is not None:
            lles: array of shape Ts.shape
            a: scalar
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = jr.normal(key, shape=(D,))
    v = v / jnp.linalg.norm(v)  # normalize to unit length

    def body_fn(carry, J):
        v, logsum = carry
        v = J[0, 0] * v if D == 1 else J @ v  # advance tangent vec
        n = jnp.linalg.norm(v)

        log_n = lax.cond(n == 0.0, lambda _: -jnp.inf, lambda n_: jnp.log(n_), n)
        logsum += log_n

        v = lax.cond(n == 0.0, lambda _: v, lambda n_: v / n_, n)

        return (v, logsum), logsum

    (v_final, lyap_sum), step_logs = lax.scan(body_fn, (v, 0.0), jacobians)

    steps = jnp.arange(len(step_logs)) + 1
    lles = step_logs / steps
    lle_final = lyap_sum / jacobians.shape[0]

    # Overshoot computation
    log_scaled = step_logs - lle_final * steps
    log_a = jnp.max(log_scaled)
    a = jnp.exp(log_a)

    if Ts is not None:
        return lles[Ts], a
    return lle_final / dt, a


def get_spectral_norm(x, u, f):
    """
    Returns spectral norm (largest singular value) of Jacobian of dynamics fxn at x
    Args:
        x: jnp.array, shape (D,)
        u: jnp.array, shape (d_in), the inputs
    """
    J = jax.jacobian(f)(x, u)
    return jnp.linalg.norm(J, ord=2)
