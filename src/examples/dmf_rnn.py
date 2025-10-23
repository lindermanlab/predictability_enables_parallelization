"""
dmf_rnn.py
Code to implement a discretized mean field RNN
Based on: https://arxiv.org/abs/2006.02427
"""

import jax
import jax.nn as nn
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array
from typing import Callable

from dynamics import DynamicalSystem


class DiscretizedMeanFieldRNN(DynamicalSystem):
    """
    Code to implement a discretized mean field RNN
    Based on: https://arxiv.org/abs/2006.02427
    Recovers a fairly standard discrete time vanilla RNN if dt=1
    """

    W: Array
    phi: Callable = eqx.field(static=True)
    when: str = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        g=1.0,
        D=2,
        phi=jnp.tanh,
        self_couple=False,
        when="before",
        scale=1.0,
        key=jr.PRNGKey(0),
    ):
        """
        Initialize the discretized mean field RNN
        Args:
            g, float, desired maximum eigenvalue
            D: int, dimensionality of the state
            dt: float, step size for time discretization
            phi: callable, nonlinearity
            self_couple: bool, whether or not to self_couple
            when: string, when to apply the nonlinearity
                "before": apply the nonlinearity before the linear transformation
                "after": apply the nonlinearity after the linear transformation
            key: jax random key
        """
        # initialize W according to mean field
        self.W = jr.normal(key, shape=(D, D)) * g / jnp.sqrt(D)
        if not self_couple:
            self.W = jnp.fill_diagonal(self.W, 0, inplace=False)
        self.phi = phi
        self.when = when
        self.scale = scale

    def deer_fxn(self, state, input):
        """
        For now, hardcode dt=1
        """
        if self.when == "before":
            new_state = (
                self.W @ self.phi(state) + input
            )  # notice that state is the total synaptic unit, i.e. not bounded between -1 and 1.
        elif self.when == "after":
            new_state = self.W @ state + input
            new_state = self.phi(new_state)
        new_state = self.scale * new_state
        return new_state

    def hardcoded_derivative(self, state):
        """
        Hardcode derivative for when=before, phi=jnp.tanh
        """
        return self.W @ jnp.diag(1 / ((jnp.cosh(state)) ** 2))

    def scan_fxn(self, state, input):
        new_state = self.deer_fxn(state, input)
        return new_state, new_state
