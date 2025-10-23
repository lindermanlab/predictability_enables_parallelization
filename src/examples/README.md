# Examples

Each file in this directory contains code for a specific example (a particular state space model) used in the paper.

The idea is that we use similar objects for different state space models, allowing us to drop different SSMs into different experiments.

For example, the `dmf_rnn.py` file contains code for the discretized mean field RNN model used in Figures 2 and 6, while the `two_well.py` file contains code for the two-well potential model used in Figure 3.

Each of these dynamics objects has two important methods:
* `deer_fxn(state, input)` return the new state, i.e. gives $s_t = f(s_{t-1}, u_t)$. i.e. `deer_fxn` is just the dynamics function $f$. We name it such because `deer_fxn` is the argument `f` you want to give to `deer.deer_alg` in order to run DEER.
* `scan_fxn(state, input)` is basically the same thing as `deer_fxn`, but returns $s_t$ twice so that we can use `jax.lax.scan` to run the dynamics forward in time (standard sequential rollout)