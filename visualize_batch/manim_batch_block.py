from manim import *
import numpy as np


class BlochBatch(ThreeDScene):
    def construct(self):
        # -----------------------------
        # Load precomputed data
        # -----------------------------
        ex = np.load("data/ex.npy")
        ey = np.load("data/ey.npy")
        ez = np.load("data/ez.npy")

        N, Nt = ex.shape

        # choose 5–6 qubits to visualize
        indices = np.linspace(0, N - 1, 6, dtype=int)

        # -----------------------------
        # Bloch sphere + axes
        # -----------------------------
        axes = ThreeDAxes(
            x_range=[-1, 1, 1],
            y_range=[-1, 1, 1],
            z_range=[-1, 1, 1],
            x_length=4,
            y_length=4,
            z_length=4,
        )

        sphere = Sphere(
            radius=2,
            resolution=(24, 24),
            fill_opacity=0.05,
            stroke_opacity=0.4,
            color=BLUE
        )

        self.set_camera_orientation(phi=65 * DEGREES, theta=35 * DEGREES)
        self.add(axes, sphere)

        # -----------------------------
        # Qubit points (all on same sphere)
        # -----------------------------
        colors = [RED, GREEN, YELLOW, ORANGE, PURPLE, TEAL]

        dots = []
        for c in colors[:len(indices)]:
            dot = Dot3D(radius=0.06, color=c)
            dots.append(dot)
            self.add(dot)

        # -----------------------------
        # Optional: trails (very nice)
        # -----------------------------
        trails = [
            TracedPath(dot.get_center, stroke_color=dot.color, stroke_width=2)
            for dot in dots
        ]
        for trail in trails:
            self.add(trail)

        # -----------------------------
        # Animate time evolution
        # -----------------------------
        for t in range(0, Nt, 4):
            for dot, k in zip(dots, indices):
                dot.move_to([
                    2 * ex[k, t],
                    2 * ey[k, t],
                    2 * ez[k, t],
                ])
            self.wait(0.02)
