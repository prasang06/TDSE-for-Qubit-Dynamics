# visualization/bloch.py

import numpy as np
import matplotlib.pyplot as plt
from quantum import sx, sy, sz


def plot_bloch(sim):

    ex = np.asarray(sim.expectation(sx))
    ey = np.asarray(sim.expectation(sy))
    ez = np.asarray(sim.expectation(sz))

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(
        x, y, z,
        color="lightgray",
        alpha=0.15,
        linewidth=0
    )

    ax.quiver(0, 0, 0, 1, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, 1, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1, color="b", linewidth=2)

    ax.plot(ex, ey, ez, color="k", lw=2, label="Bloch trajectory")
    ax.scatter(ex[0], ey[0], ez[0], color="green", s=50, label="start")
    ax.scatter(ex[-1], ey[-1], ez[-1], color="red", s=50, label="end")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel(r"$\langle\sigma_x\rangle$")
    ax.set_ylabel(r"$\langle\sigma_y\rangle$")
    ax.set_zlabel(r"$\langle\sigma_z\rangle$")

    ax.set_title("Bloch Sphere Trajectory")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()
