import matplotlib.pyplot as plt
import numpy as np


def plot_sigma_time_traces(t, ex, ey, ez, indices):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    sigmas = {
        r"$\langle \sigma_x \rangle$": ex,
        r"$\langle \sigma_y \rangle$": ey,
        r"$\langle \sigma_z \rangle$": ez,
    }

    for ax, (label, data) in zip(axes, sigmas.items()):
        for k in indices:
            ax.plot(
                t,
                data[k],
                linewidth=1.2,
                alpha=0.7,
                label=f"Qubit {k}"
            )

        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=9, title="Sampled qubits")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Sample individual qubit dynamics")

    plt.tight_layout()
    plt.show()
