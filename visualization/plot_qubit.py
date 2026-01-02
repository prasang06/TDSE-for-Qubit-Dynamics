import numpy as np
import matplotlib.pyplot as plt


def plot_qubit(sim):
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

    ax0.plot(t, p0, lw=2, color="C0")
    ax1.plot(t, p1, lw=2, color="C1")

    ax2.plot(t, p0, lw=2, color="C0", label=r"$|0\rangle$")
    ax2.plot(t, p1, lw=2, color="C1", label=r"$|1\rangle$")
    ax2.legend()

    plt.tight_layout()
    plt.show()
