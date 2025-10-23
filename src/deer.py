"""
deer.py
DEER implementation (no backwards pass for simplicity)
"""

import jax
jax.config.update("jax_enable_x64", True)
from jax import vmap
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl

from lle import wrapper_estimate_lle_from_jacobians, estimate_lle_from_jacobians

@jax.vmap
def full_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j


@jax.vmap
def diag_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def get_residual(f, initial_state, states, drivers):
    fs = vmap(f)(states[:-1], drivers[1:])  # length T-1
    fs = jnp.concatenate([jnp.array([f(initial_state, drivers[0])]), fs])
    r = states - fs
    return r


def merit_fxn(f, initial_state, states, drivers, Ts=None):
    """
    Helper function to compute the merit function
    Note that this assumes that the initial state (say s0) is combined with the initial noise (drivers[0]) to make s1 (the first state)
    Args:
        f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
        initial_state: packed_state, jax.Array (DIM,)
        states, jax.Array, (T, DIM)
        drivers, jax.Array, (T,N_noise)
        Ts: optional
            if None, then just return the merit function
            ow an array of ints, and returns the merit function for each T in Ts
    """
    r = get_residual(f, initial_state, states, drivers)  # (T,D)
    T, D = r.shape
    Ls = 0.5 * jnp.sum(
        jnp.cumsum(r**2, axis=0), axis=1
    )  # make sure to sum over time!! (T,)
    Ls = jnp.where(jnp.isnan(Ls), jnp.inf, Ls)
    if Ts is not None:
        return Ls[Ts - 1] / Ts
    else:
        return Ls[-1] / T  # average per time step

def deer_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    quasi=False,
    diagonal_func=None,
    k=0,
    clip=False,
    get_lles=False,
    full_trace=False,
    Ts=None,
    reset=False,
    tol=1e-10,
):
    """
    Lightweight implementation of DEER
    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (T, DIM)
      drivers, jax.Array, (T,N_noise)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
      k: amount of damping, should be between 0 and 1. 0 is no damping, 1 is max damping.
      clip: bool, whether or not the Jacobian should be clipped to have eigenvalues in range [-1,1]
      get_lles: bool, whether or not to return the discrete-time LLE estimates.
      full_trace: bool, whether or not to return the full trace of the DEER iterations
        if True, uses scan
        if False, uses while loop
      Ts: optional
        if None, then just return the number of deer iterations
        ow return the number of deer iterations for each T in Ts
      reset: bool, whether or not to reset the states to the initial guess if they are NaN
      tol: float, the tolerance for convergence (merit function). We normalize the merit function by its sequence length T.
    Notes:
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:T] be the states, and e[0:T-1] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{T-1} ----> h_{T}
              |       |                   |          |
              e1      e2       ..... e_{T-1}      e_{T}

    NOTES:
    * DO NOT INIT FROM ZERO!!!!!
    * drivers struggles if it is passed a tuple...
    """
    DIM = len(initial_state)
    L = len(drivers)

    @jax.jit
    def _step(carry, args):
        """
        Args:
            carry: tuple of (states, is_nan, iter_num)
            args: None
        """
        states, is_nan = carry
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = vmap(f)(
            states[:-1], drivers[1:]
        )  # get the next. Note that states[0] is h1, and drivers[1] is e2. h2=f(h1, e2)
        nan_jac_idx = False
        # Compute the As and bs from fs and Jfs
        if quasi:
            if diagonal_func:
                As = vmap(diagonal_func)(states[:-1], drivers[1:])
            else:
                # Jfs are the Jacobians 
                Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])
                As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
            As = (1 - k) * As  # damping
            if clip:
                As = jnp.clip(As, -1, 1)
            bs = fs - As * states[:-1]
            lle = None
        else:
            # Jfs are the Jacobians 
            Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])  # (T, D, D)
            # clean up nans in Jfs
            if reset:
                nan_mask_jacobians = jnp.isnan(Jfs).any(
                    axis=(1, 2), keepdims=True
                )  # Shape (T, 1, 1)
                nan_jac_idx = jnp.isnan(Jfs).any()
                I = jnp.eye(Jfs.shape[1])
                Jfs = jnp.where(nan_mask_jacobians, I, Jfs)
            # clean up nans in Jfs
            if get_lles:
                lle = estimate_lle_from_jacobians(Jfs, jr.PRNGKey(42))
            else:
                lle = None
            As = Jfs
            As = (1 - k) * As  # damping
            bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

        # initial_state is h0
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        A0 = jnp.zeros_like(As[0])
        A = jnp.concatenate(
            [A0[jnp.newaxis, :], As]
        )  # (T, D, D) [or (T, D) for quasi]
        b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (T, D)
        if quasi:
            binary_op = diag_mat_operator
        else:
            binary_op = full_mat_operator

        # run appropriate parallel alg
        _, new_states = jax.lax.associative_scan(
            binary_op, (A, b)
        )  # a forward pass, but uses linearized dynamics
        is_nan = jnp.logical_or(
            jnp.logical_or(jnp.isnan(new_states).any(), is_nan), nan_jac_idx
        )
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, lle, nan_mask, mf_val)

    def cond_func(iter_inp):
        """
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        """
        Body func when we only use one T
        iter_inp: tuple of (iter_idx, states, err)
        iter_idx: int, the current iteration
        states: (T,D), the current states
        err: current value of merit function
        is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        """
        Body func when we use multiple Ts
        iter_inp: tuple of (iter_idx, states, err)
        iter_idx: int, the current iteration
        states: (T,D), the current states
        err: current value of merit function
        is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        iters_below: array of ints, for each T, the number of iterations that have been below tol for the merit function
        """
        iter_idx, states, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, new_errs[-1], is_nan, iters_below

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, lle, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                jnp.zeros_like(Ts),
            ),
        )
        Jfs = vmap(jax.jacrev(f, argnums=0))(final_state[:-1], drivers[1:])  # (T, D, D)
        lle = wrapper_estimate_lle_from_jacobians(Jfs, jr.PRNGKey(42))
        all_states, nan_mask, mf_val = None, None, None
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        Jfs = vmap(jax.jacrev(f, argnums=0))(final_state[:-1], drivers[1:])  # (T, D, D)
        lle = wrapper_estimate_lle_from_jacobians(Jfs, jr.PRNGKey(42))
        all_states, nan_mask, mf_val = None, None, None
    if iters_below is not None:
        all_newtons = newton_steps - iters_below + 1  # the last input is below the tol
    else:
        all_newtons = None
    return (
        all_states,
        final_state,
        newton_steps,
        lle,
        is_nan,
        nan_mask,
        all_newtons,
        mf_val,
    )


### GD


def explicit_GD(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,  # controls number of iteration
    alpha=1,  # alpha is the step size
    full_trace=False,  # whether to return the full trace of the states
    tol=1e-7,  # tolerance for convergence
):
    """
    Explicitly compute the gradient of the merit function and do gradient descent on it

    bool, whether or not to return the full trace of the DEER iterations
        if True, uses scan
        if False, uses while loop
    """
    DIM = len(initial_state)
    L = len(drivers)

    def residual(states):
        fs = vmap(f)(states[:-1], drivers[1:])  # length T-1
        fs = jnp.concatenate([jnp.array([f(initial_state, drivers[0])]), fs])
        return states - fs

    def merit_function(states):
        r = residual(states)
        return 0.5 * jnp.sum(r**2)

    def _step(states, args):
        g = jax.grad(merit_function)(states)
        new_states = states - alpha * g
        return new_states, new_states

    def cond_func(iter_inp):
        """
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
        """
        iter_idx, states, err = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        """
        Body func when we only use one T
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
        """
        iter_idx, states, err = iter_inp
        new_states, _ = _step(states, None)
        new_err = merit_function(new_states)
        return iter_idx + 1, new_states, new_err

    if full_trace:
        final_state, all_states = scan(_step, states_guess, None, length=num_iters)
        num_gd_steps = None
    else:
        num_gd_steps, final_state, _ = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, merit_function(states_guess)),
        )
        all_states = None
    return all_states, final_state, num_gd_steps


