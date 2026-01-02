# solvers/tdse.py
import jax
import jax.numpy as jnp
from .rk4 import rk4_step

def schrodinger_rhs(psi, t, system, params):
    H = system.hamiltonian(t)
    return -1j / params.hbar * (H @ psi)

def evolve_tdse(psi0, t_array, system, params):
    dt = t_array[1] - t_array[0]

    def step(psi, t):
        psi_next = rk4_step(
            lambda p, tt, pr: schrodinger_rhs(p, tt, system, pr),
            psi, t, dt, params
        )
        return psi_next, psi_next

    _, psi_t = jax.lax.scan(step, psi0, t_array[:-1])
    return psi_t


evolve_tdse = jax.jit(
    evolve_tdse,
    static_argnames=("system", "params")
)
