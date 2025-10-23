# a translation of https://github.com/williamgilpin/dysts/blob/master/dysts/flows.py into JAX and the DEER framework.


import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array

from dynamics import DynamicalSystem


class Lorenz(DynamicalSystem):
    beta: float
    rho: float
    sigma: float
    dt: float

    def __init__(self, beta: float, rho: float, sigma: float, dt: float = 0.01):
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.sigma * (y - x)
        ydot = self.rho * x - x * z - y
        zdot = x * y - self.beta * z
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        dx = self.dynamics(state)
        return state + dx * self.dt  # You can define `dt` as a class variable if needed

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Rossler(DynamicalSystem):
    a: float
    b: float
    c: float
    dt: float

    def __init__(self, a: float, b: float, c: float, dt: float = 0.01):
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = -y - z
        ydot = x + self.a * y
        zdot = self.b + z * (x - self.c)
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        dx = self.dynamics(state)
        return state + dx * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Thomas(DynamicalSystem):
    a: float
    b: float
    dt: float

    def __init__(self, a: float, b: float, dt: float = 0.01):
        self.a = a
        self.b = b
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = -self.a * x + self.b * jnp.sin(y)
        ydot = -self.a * y + self.b * jnp.sin(z)
        zdot = -self.a * z + self.b * jnp.sin(x)
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        dx = self.dynamics(state)
        return state + dx * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class GlycolyticOscillation(DynamicalSystem):
    d: float
    l1: float
    l2: float
    nu: float
    q1: float
    q2: float
    s1: float
    s2: float
    k: float

    dt: float

    def __init__(
        self,
        k: float,
        d: float,
        l1: float,
        l2: float,
        nu: float,
        q1: float,
        q2: float,
        s1: float,
        s2: float,
        dt: float = 0.01,
    ):
        self.k = k
        self.d = d
        self.l1 = l1
        self.l2 = l2
        self.nu = nu
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        a, b, c = state
        phi = (a * (1 + a) * (1 + b) ** 2) / (self.l1 + (1 + a) ** 2 * (1 + b) ** 2)
        eta = (
            b
            * (1 + self.d * b)
            * (1 + c) ** 2
            / (self.l2 + (1 + self.d * b) ** 2 * (1 + c) ** 2)
        )
        adot = self.nu - self.s1 * phi
        bdot = self.q1 * self.s1 * phi - self.s2 * eta
        cdot = self.q2 * self.s2 * eta - self.k * c
        return jnp.array([adot, bdot, cdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        dx = self.dynamics(state)
        return state + dx * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class GuckenheimerHolmes(DynamicalSystem):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    dt: float

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        dt: float = 0.01,
    ):
        self.a, self.b, self.c, self.d, self.e, self.f, self.dt = a, b, c, d, e, f, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.a * x - self.b * y + self.c * z * x + self.d * z * (x**2 + y**2)
        ydot = self.a * y + self.b * x + self.c * z * y
        zdot = self.e - z**2 - self.f * (x**2 + y**2) - self.a * z**3
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Halvorsen(DynamicalSystem):
    a: float
    b: float
    dt: float

    def __init__(self, a: float, b: float, dt: float = 0.01):
        self.a, self.b, self.dt = a, b, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = -self.a * x - self.b * (y + z) - y**2
        ydot = -self.a * y - self.b * (z + x) - z**2
        zdot = -self.a * z - self.b * (x + y) - x**2
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Chua(DynamicalSystem):
    alpha: float
    beta: float
    m0: float
    m1: float
    dt: float

    def __init__(
        self, alpha: float, beta: float, m0: float, m1: float, dt: float = 0.01
    ):
        self.alpha, self.beta, self.m0, self.m1, self.dt = alpha, beta, m0, m1, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        ramp_x = self.m1 * x + 0.5 * (self.m0 - self.m1) * (
            jnp.abs(x + 1) - jnp.abs(x - 1)
        )
        xdot = self.alpha * (y - x - ramp_x)
        ydot = x - y + z
        zdot = -self.beta * y
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Duffing(DynamicalSystem):
    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float,
        omega: float,
        dt: float = 0.01,
    ):
        self.alpha, self.beta, self.delta, self.gamma, self.omega, self.dt = (
            alpha,
            beta,
            delta,
            gamma,
            omega,
            dt,
        )

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = y
        ydot = (
            -self.delta * y
            - self.beta * x
            - self.alpha * x**3
            + self.gamma * jnp.cos(z)
        )
        zdot = self.omega
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class KawczynskiStrizhak(DynamicalSystem):
    beta: float
    gamma: float
    kappa: float
    mu: float
    dt: float

    def __init__(
        self, beta: float, gamma: float, kappa: float, mu: float, dt: float = 0.01
    ):
        self.beta, self.gamma, self.kappa, self.mu, self.dt = beta, gamma, kappa, mu, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.gamma * y - self.gamma * x**3 + 3 * self.mu * self.gamma * x
        ydot = -2 * self.mu * x - y - z + self.beta
        zdot = self.kappa * (x - z)
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class IsothermalChemical(DynamicalSystem):
    def __init__(
        self, delta: float, kappa: float, mu: float, sigma: float, dt: float = 0.01
    ):
        self.delta, self.kappa, self.mu, self.sigma, self.dt = (
            delta,
            kappa,
            mu,
            sigma,
            dt,
        )

    def dynamics(self, state: Array, input: Array = None) -> Array:
        alpha, beta, gamma = state
        alphadot = self.mu * (self.kappa + gamma) - alpha * beta**2 - alpha
        betadot = (alpha * beta**2 + alpha - beta) / self.sigma
        gammadot = (beta - gamma) / self.delta
        return jnp.array([alphadot, betadot, gammadot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class VallisElNino(DynamicalSystem):
    b: float
    c: float
    p: float
    dt: float

    def __init__(self, b: float, c: float, p: float, dt: float = 0.01):
        self.b, self.c, self.p, self.dt = b, c, p, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.b * y - self.c * (x + self.p)
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class RabinovichFabrikant(DynamicalSystem):
    a: float
    g: float
    dt: float

    def __init__(self, a: float, g: float, dt: float = 0.01):
        self.a, self.g, self.dt = a, g, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = y * z - y + y * x**2 + self.g * x
        ydot = 3 * x * z + x - x**3 + self.g * y
        zdot = -2 * self.a * z - 2 * x * y * z
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class NoseHoover(DynamicalSystem):
    a: float
    dt: float

    def __init__(self, a: float, dt: float = 0.01):
        self.a = a
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = y
        ydot = -x + y * z
        zdot = self.a - y**2
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Dadras(DynamicalSystem):
    def __init__(
        self, c: float, e: float, o: float, p: float, r: float, dt: float = 0.01
    ):
        self.c, self.e, self.o, self.p, self.r, self.dt = c, e, o, p, r, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = y - self.p * x + self.o * y * z
        ydot = self.r * y - x * z + z
        zdot = self.c * x * y - self.e * z
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class RikitakeDynamo(DynamicalSystem):
    def __init__(self, a: float, mu: float, dt: float = 0.01):
        self.a, self.mu, self.dt = a, mu, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = -self.mu * x + y * z
        ydot = -self.mu * y - self.a * x + x * z
        zdot = 1 - x * y
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class Aizawa(DynamicalSystem):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    dt: float

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        dt: float = 0.01,
    ):
        self.a, self.b, self.c, self.d, self.e, self.f, self.dt = a, b, c, d, e, f, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = x * z - self.b * x - self.d * y
        ydot = self.d * x + y * z - self.b * y
        zdot = (
            self.c
            + self.a * z
            - (z**3) / 3
            - x**2
            - y**2
            - self.e * z * x**2
            - self.e * z * y**2
            + self.f * z * x**3
        )
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class ABC(DynamicalSystem):
    A: float
    B: float
    C: float
    dt: float

    def __init__(self, A: float, B: float, C: float, dt: float = 0.01):
        self.A, self.B, self.C, self.dt = A, B, C, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.A * jnp.sin(z) + self.C * jnp.cos(y)
        ydot = self.B * jnp.sin(x) + self.A * jnp.cos(z)
        zdot = self.C * jnp.sin(y) + self.B * jnp.cos(x)
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class ChenLee(DynamicalSystem):
    a: float
    b: float
    c: float
    dt: float

    def __init__(self, a: float, b: float, c: float, dt: float = 0.01):
        self.a, self.b, self.c, self.dt = a, b, c, dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = self.a * x - y * z
        ydot = self.b * y + x * z
        zdot = self.c * z + (1 / 3) * x * y
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state


class SprottB(DynamicalSystem):
    dt: float

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def dynamics(self, state: Array, input: Array = None) -> Array:
        x, y, z = state
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return jnp.array([xdot, ydot, zdot])

    def deer_fxn(self, state: Array, input: Array = None) -> Array:
        return state + self.dynamics(state) * self.dt

    def scan_fxn(self, state: Array, input: Array = None) -> tuple[Array, Array]:
        new_state = self.deer_fxn(state)
        return new_state, new_state
