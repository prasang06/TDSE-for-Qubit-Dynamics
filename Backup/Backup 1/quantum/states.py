import jax
import jax.numpy as jnp

def random_qubit_state(key):
    a = jax.random.normal(key, (2,))
    b = jax.random.normal(key, (2,))
    psi = a + 1j * b
    return psi / jnp.linalg.norm(psi)


def ket0():
    return jnp.array([1.0, 0.0], dtype=jnp.complex64)

def ket1():
    return jnp.array([0.0, 1.0], dtype=jnp.complex64)

def superposition(alpha, beta):
    psi = jnp.array([alpha, beta], dtype=jnp.complex64)
    return normalize(psi)

def normalize(psi):
    return psi / jnp.sqrt(jnp.vdot(psi, psi))
