import jax.numpy as jnp

def infinite_square_well(x, L):
    return jnp.where((x > 0) & (x < L), 0.0, 1e12)

def harmonic_oscillator(x, m, omega):
    return 0.5 * m * omega**2 * x**2
