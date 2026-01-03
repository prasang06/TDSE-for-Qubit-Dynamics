import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantum import ket0, QubitParams, StaticQubit
from solvers import evolve_tdse
from simulation import TDSESimulation
from visualization import plot_qubit, plot_bloch, animate_qubit, animate_bloch

def main():
    params = QubitParams(omega=1.0)   # ω = 1
    system = StaticQubit(params)

    psi0 = (ket0() + jnp.array([0.0, 1.0])) / jnp.sqrt(2)

    t = jnp.linspace(0.0, 10.0, 1000)

    psi_t = evolve_tdse(psi0, t, system, params)

    sim = TDSESimulation(t=t[1:], psi=psi_t)

    print("Max norm deviation:", jnp.max(jnp.abs(sim.norm - 1.0)))

    plot_qubit(sim)

    ani = animate_bloch(sim, interval=5)
    plt.show()


if __name__ == "__main__":
    main()
