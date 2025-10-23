"""
dynamics.py
helper functions for the dynamics experiments
"""

import equinox as eqx
import jax


class DynamicalSystem(eqx.Module):
    """
    Remember that when you use methods of an equinox module in a scan, you have to use an anonymous function
    """

    def deer_fxn(self, state, input):
        raise NotImplementedError("Subclasses must implement deer_fxn.")

    def scan_fxn(self, state, input):
        state = self.deer_fxn(state, input)
        return state, state


class Observer(DynamicalSystem):
    def system_nonlinearity(self, x, t):
        raise NotImplementedError("Subclasses must implement system nonlinearity.")

    def deer_fxn_observer(self, state, input):
        raise NotImplementedError("Subclasses must implement system nonlinearity.")

    def scan_fxn_observer(self, state, input):
        state = self.deer_fxn_observer(state, input)
        return state, state

    def simulate_with_observer(self, x0, xhat0, N_steps):
        """
        Simulate the system with an observer
        Args:
            x0: initial state
            xhat0: initial state of the observer
            N_steps: number of steps to simulate
        Returns:
            xs: states
            xhats: states of the observer
            ys: measurements
            y_hats: measurements of the observer
            times: time steps
        """
        raise NotImplementedError("Subclasses must implement simulate_with_observer.")


from jaxtyping import Array
from typing import Callable, List


class FeedbackObserver(eqx.Module):
    system: eqx.Module
    K: jax.Array
    h: Callable = eqx.field(static=True)
    dt: float
    inds_to_replace: List[int] = eqx.field(static=True)

    def dynamics(self, state: jax.Array, input: jax.Array = None) -> jax.Array:
        fz = self.system.dynamics(state)

        if self.K is not None:
            hz = self.h(state)
            hx = self.h(input)
            correction = self.K @ (hx - hz)
        else:
            correction = 0.0

        return fz + correction

    def deer_fxn(self, state: jax.Array, input: jax.Array = None) -> jax.Array:
        for ind in self.inds_to_replace:
            state = state.at[ind].set(input[ind])
        dz = self.dynamics(state, input)
        return state + dz * self.dt

    def scan_fxn(
        self, state: jax.Array, input: jax.Array = None
    ) -> tuple[jax.Array, jax.Array]:
        new_z = self.deer_fxn(state, input)
        return new_z, new_z
