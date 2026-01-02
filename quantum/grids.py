import jax.numpy as jnp

def uniform_grid(xmin, xmax, N):
    x = jnp.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx
