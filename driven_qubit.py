import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantum import ket0
from quantum.systems import DrivenQubit, DrivenQubitParams
from solvers import evolve_tdse
from simulation import TDSESimulation
from visualization import animate_qubit, plot_qubit, plot_bloch, animate_bloch


def main():
    params = DrivenQubitParams(
        omega=1.0,
        Omega=0.5,
        omega_d=1.0
    )

    system = DrivenQubit(params)

    psi0 = ket0()

    t = jnp.linspace(0.0, 20.0, 800)

    psi_t = evolve_tdse(psi0, t, system, params)

    sim = TDSESimulation(t=t[1:], psi=psi_t)

    print("Max norm deviation:", jnp.max(jnp.abs(sim.norm - 1.0)))

    plot_qubit(sim)

    ani = animate_bloch(sim, interval=10)
    plt.show()



if __name__ == "__main__":
    main()
