# simulation/result.py
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class TDSESimulation:
    t: jnp.ndarray
    psi: jnp.ndarray

    @property
    def probability(self):
        return jnp.abs(self.psi) ** 2

    @property
    def norm(self):
        return jnp.sum(self.probability, axis=1)

    def expectation(self, operator):
        return jnp.array([
            jnp.vdot(psi_t, operator @ psi_t).real
            for psi_t in self.psi
        ])
