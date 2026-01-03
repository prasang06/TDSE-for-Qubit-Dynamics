import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_bloch_ensemble(ex, ey, ez, indices, ex_mean=None, ey_mean=None, ez_mean=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Bloch sphere wireframe
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.25)

    # Plot trajectories
    for k in indices:
        ax.plot(
            ex[k],
            ey[k],
            ez[k],
            linewidth=1.2,
            alpha=0.5
        )

        # start & end
        ax.scatter(ex[k, 0], ey[k, 0], ez[k, 0], color="green", s=20)
        ax.scatter(ex[k, -1], ey[k, -1], ez[k, -1], color="red", s=20)

    # Optional ensemble mean
    if ex_mean is not None:
        ax.plot(
            ex_mean,
            ey_mean,
            ez_mean,
            color="black",
            linewidth=3,
            label="Ensemble mean"
        )
        ax.legend()

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel(r"$\langle \sigma_x \rangle$")
    ax.set_ylabel(r"$\langle \sigma_y \rangle$")
    ax.set_zlabel(r"$\langle \sigma_z \rangle$")
    ax.set_title("Ensemble Bloch-sphere trajectories")

    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.show()
