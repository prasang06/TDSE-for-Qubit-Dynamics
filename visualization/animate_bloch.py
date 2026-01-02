import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from quantum import sx, sy, sz


def animate_bloch(sim, interval=30, trail=True):
    # --- expectation values ---
    ex = np.asarray(sim.expectation(sx))
    ey = np.asarray(sim.expectation(sy))
    ez = np.asarray(sim.expectation(sz))


    #--- probabilities
    psi = np.asarray(sim.psi)
    p0 = np.abs(psi[:, 0])**2
    p1 = np.abs(psi[:, 1])**2


    # --- Bloch sphere mesh ---
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Bloch Sphere Evolution")


    # --- draw sphere ---
    ax.plot_surface(
        x, y, z,
        color="lightgray",
        alpha=0.15,
        linewidth=0
    )

    # --- axes ---
    ax.quiver(0, 0, 0, 1, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, 1, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1, color="b", linewidth=2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel(r"$\langle\sigma_x\rangle$")
    ax.set_ylabel(r"$\langle\sigma_y\rangle$")
    ax.set_zlabel(r"$\langle\sigma_z\rangle$")

    # --- trajectory & point ---
    traj, = ax.plot([], [], [], lw=2, color="k")
    point, = ax.plot([], [], [], "o", color="red", markersize=6)

    # --- start marker ---
    ax.scatter(ex[0], ey[0], ez[0], color="green", s=40, label="start")
    ax.legend()

    stats_text = ax.text2D(
        0.05, 0.95, "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )


    def init():
        traj.set_data([], [])
        traj.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        stats_text.set_text("")
        return traj, point, stats_text


    def update(frame):
        if trail:
            traj.set_data(ex[:frame], ey[:frame])
            traj.set_3d_properties(ez[:frame])

        point.set_data([ex[frame]], [ey[frame]])
        point.set_3d_properties([ez[frame]])

        stats_text.set_text(
            f"P(|0⟩) = {p0[frame]:.3f}\n"
            f"P(|1⟩) = {p1[frame]:.3f}"
        )

        return traj, point, stats_text


    ani = FuncAnimation(
        fig,
        update,
        frames=len(ex),
        init_func=init,
        interval=interval,
        blit=False   # 3D plots require blit=False
    )

    return ani
