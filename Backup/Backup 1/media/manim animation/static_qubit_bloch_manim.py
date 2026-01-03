import numpy as np
import jax.numpy as jnp

from quantum import ket0, sx, sy, sz
from quantum.systems import StaticQubit, QubitParams
from solvers import evolve_tdse
from simulation import TDSESimulation


def main():
    # --- system ---
    params = QubitParams(omega=1.0)
    system = StaticQubit(params)

    # --- initial state ---
    psi0 = (ket0() + jnp.array([0.0, 1.0])) / jnp.sqrt(2)

    # --- time grid ---
    t = jnp.linspace(0.0, 20.0, 2000)

    # --- TDSE solve ---
    psi_t = evolve_tdse(psi0, t, system, params)
    sim = TDSESimulation(t=t[1:], psi=psi_t)

    # --- Bloch coordinates ---
    ex = np.asarray(sim.expectation(sx))
    ey = np.asarray(sim.expectation(sy))
    ez = np.asarray(sim.expectation(sz))

    bloch_xyz = np.stack([ex, ey, ez], axis=1)

    # --- SAVE FILE ---
    np.save("bloch_static.npy", bloch_xyz)
    print("Saved bloch_static.npy")


if __name__ == "__main__":
    main()
