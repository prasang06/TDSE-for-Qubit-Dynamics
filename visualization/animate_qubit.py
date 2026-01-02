# visualization/animate_qubit.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_qubit(sim, interval=30):
    t = np.asarray(sim.t)
    psi = np.asarray(sim.psi)

    p0 = np.abs(psi[:, 0]) ** 2
    p1 = np.abs(psi[:, 1]) ** 2

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, sharex=True, figsize=(9, 8)
    )

    fig.suptitle("Qubit TDSE Evolution", fontsize=14)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

    ax2.set_xlabel("Time")

    ax0.set_title(r"$|\langle 0 | \psi(t) \rangle|^2$")
    ax1.set_title(r"$|\langle 1 | \psi(t) \rangle|^2$")
    ax2.set_title("Overlapping probabilities")


    line0, = ax0.plot(t, p0, lw=2, color="C0")
    line1, = ax1.plot(t, p1, lw=2, color="C1")

    line0_ov, = ax2.plot([], [], lw=2, color="C0", label=r"$|0\rangle$")
    line1_ov, = ax2.plot([], [], lw=2, color="C1", label=r"$|1\rangle$")
    ax2.legend()

    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        line0_ov.set_data([], [])
        line1_ov.set_data([], [])
        return line0, line1, line0_ov, line1_ov

    def update(frame):
        line0.set_data(t[:frame], p0[:frame])
        line1.set_data(t[:frame], p1[:frame])
        line0_ov.set_data(t[:frame], p0[:frame])
        line1_ov.set_data(t[:frame], p1[:frame])
        return line0, line1, line0_ov, line1_ov

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval,
        blit=True
    )

    return ani