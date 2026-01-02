import numpy as np
from manim import *


# =========================
# CONFIGURATION
# =========================

DATA_FILE = "bloch_driven.npy"   # change to bloch_static.npy if needed
SPHERE_RADIUS = 2.0


# =========================
# LOAD DATA (ON IMPORT)
# =========================

bloch_xyz = np.load(DATA_FILE)        # shape (T, 3)
bloch_xyz = np.clip(bloch_xyz, -1, 1) # safety


# =========================
# MANIM SCENE
# =========================

class BlochSphereScene(ThreeDScene):
    def construct(self):

        # -------------------------
        # Camera
        # -------------------------
        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=45 * DEGREES,
            zoom=0.9
        )

        # -------------------------
        # Bloch Sphere
        # -------------------------
        sphere = Sphere(
            radius=SPHERE_RADIUS,
            resolution=(24, 48),
            fill_opacity=0.12,
            stroke_width=1
        )

        axes = ThreeDAxes(
            x_range=[-SPHERE_RADIUS, SPHERE_RADIUS, 1],
            y_range=[-SPHERE_RADIUS, SPHERE_RADIUS, 1],
            z_range=[-SPHERE_RADIUS, SPHERE_RADIUS, 1],
        )

        self.add(axes, sphere)

        # -------------------------
        # Axis Labels
        # -------------------------
        labels = VGroup(
            MathTex(r"\langle\sigma_x\rangle").next_to(axes.x_axis.get_end(), RIGHT),
            MathTex(r"\langle\sigma_y\rangle").next_to(axes.y_axis.get_end(), UP),
            MathTex(r"\langle\sigma_z\rangle").next_to(axes.z_axis.get_end(), OUT),
        )
        self.add_fixed_orientation_mobjects(*labels)
        self.add(*labels)

        # -------------------------
        # Initial Bloch Point
        # -------------------------
        x0, y0, z0 = bloch_xyz[0]
        point = Dot3D(
            point=SPHERE_RADIUS * np.array([x0, y0, z0]),
            color=RED,
            radius=0.06
        )
        self.add(point)

        # -------------------------
        # Trajectory Path
        # -------------------------
        path = VMobject(color=YELLOW)
        path.set_points_as_corners(
            [SPHERE_RADIUS * np.array(p) for p in bloch_xyz]
        )

        # -------------------------
        # Probability Stats
        # -------------------------
        p0_vals = (1 + bloch_xyz[:, 2]) / 2
        p1_vals = (1 - bloch_xyz[:, 2]) / 2

        frame = ValueTracker(0)

        stats = always_redraw(
            lambda: VGroup(
                Text(f"P(|0⟩) = {p0_vals[int(frame.get_value())]:.3f}", font_size=30),
                Text(f"P(|1⟩) = {p1_vals[int(frame.get_value())]:.3f}", font_size=30),
            ).arrange(DOWN, aligned_edge=LEFT)
             .to_corner(UL)
        )

        self.add_fixed_in_frame_mobjects(stats)
        self.add(stats)

        # -------------------------
        # Animations
        # -------------------------
        self.play(Create(path), run_time=4)

        self.play(
            MoveAlongPath(point, path),
            frame.animate.set_value(len(bloch_xyz) - 1),
            run_time=6,
            rate_func=linear
        )

        self.wait(1)
