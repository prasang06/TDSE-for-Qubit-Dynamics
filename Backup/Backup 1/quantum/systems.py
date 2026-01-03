from dataclasses import dataclass
from .pauli import sx, sz
import jax.numpy as jnp

@dataclass(frozen=True)
class QubitParams:
    omega: float
    hbar: float = 1.0

@dataclass(frozen=True)
class DrivenQubitParams:
    omega: float
    Omega: float
    omega_d: float
    hbar: float = 1.0


class StaticQubit:
    def __init__(self, params: QubitParams):
        self.params = params

    def hamiltonian(self, t):
        return 0.5 * self.params.omega * sz

class DrivenQubit:
    def __init__(self, params: DrivenQubitParams):
        self.params = params

    def hamiltonian(self, t):
        return (
            0.5 * self.params.omega * sz
            + self.params.Omega * jnp.cos(self.params.omega_d * t) * sx
        )
