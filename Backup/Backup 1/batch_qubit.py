import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from quantum.states import random_qubit_state
from quantum.systems import DrivenQubit, DrivenQubitParams
from solvers import evolve_tdse
from simulation import TDSESimulation
from quantum import sx, sy, sz


def main():

    N = 100  
    t = jnp.linspace(0.0, 20.0, 1500)

    params = DrivenQubitParams(
        omega=1.0,
        Omega=0.5,
        omega_d=1.0
    )

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N)

    psi0_batch = jax.vmap(random_qubit_state)(keys)

    def evolve_one(psi0):
        system = DrivenQubit(params)
        return evolve_tdse(psi0, t, system, params)

    evolve_batch = jax.jit(
        jax.vmap(evolve_one, in_axes=0, out_axes=0)
    )

    psi_t_batch = evolve_batch(psi0_batch)

    sim = TDSESimulation(t=t[1:], psi=psi_t_batch)

    ex = sim.expectation(sx)
    ey = sim.expectation(sy)
    ez = sim.expectation(sz)

    ex_mean = np.asarray(jnp.mean(ex, axis=0))
    ey_mean = np.asarray(jnp.mean(ey, axis=0))
    ez_mean = np.asarray(jnp.mean(ez, axis=0))

    plt.plot(sim.t, ex_mean, label=r"$\langle\sigma_x\rangle$")
    plt.plot(sim.t, ey_mean, label=r"$\langle\sigma_y\rangle$")
    plt.plot(sim.t, ez_mean, label=r"$\langle\sigma_z\rangle$")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Expectation value")
    plt.title("Ensemble-averaged Bloch dynamics")
    plt.show()


if __name__ == "__main__":
    main()
