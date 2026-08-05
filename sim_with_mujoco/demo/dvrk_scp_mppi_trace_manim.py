"""Render the trace written by the dVRK constrained SCP-MPPI demos.

This is a dedicated entry point for the constrained SCP-MPPI and SCP-SMPPI
demos.  It reuses the shared trace-scene implementation while preserving its
own timestamped output directory.
"""

from datetime import datetime
from pathlib import Path
import sys

import manim as mn
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim_with_mujoco.demo import dvrk_smppi_trace_manim as _shared_trace


TRACE_PATH = ROOT_DIR / "temp" / "dvrk_scp_mppi_trace.pt"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = ROOT_DIR / "media" / "dvrk_scp_mppi_trace_runs" / RUN_ID

# The shared scene reads this module-level path at render time.  Make the
# baseline trace source explicit here rather than relying on an implicit match.
_shared_trace.TRACE_PATH = TRACE_PATH
mn.config.media_dir = str(RUN_DIR)
mn.config.video_dir = str(RUN_DIR)
mn.config.output_file = "dvrk_scp_mppi_trace.mp4"


class DvrkSCPMPPITraceScene(_shared_trace.DvrkSMPPITraceScene):
    """Constrained SCP-MPPI trace scene with tip kinematics plots."""

    def construct(self):
        trace = torch.load(TRACE_PATH, map_location="cpu", weights_only=True)
        self._save_tip_kinematics(trace)
        super().construct()

    @staticmethod
    def _save_tip_kinematics(trace):
        try:
            time = trace["physics_tip_time"].numpy().astype(np.float64)
            position = trace["physics_tip_position"].numpy().astype(np.float64)
        except KeyError as error:
            raise ValueError(
                "This trace has no physics-rate tip samples. "
                "Run dvrk_scp_smppi.py again before rendering this scene."
            ) from error

        if time.ndim != 1 or position.shape != (len(time), 3) or len(time) < 3:
            raise ValueError("The physics-rate tip trace must contain at least three (time, xyz) samples.")
        if not np.all(np.diff(time) > 0.0):
            raise ValueError("The physics-rate tip timestamps must be strictly increasing.")

        # 1) full physics-rate view
        DvrkSCPMPPITraceScene._save_tip_velocity_acceleration(
            time=time,
            position=position,
            samples_per_control=0,
            output_path=RUN_DIR / "tip_velocity_acceleration.png",
        )

        # 2) control-step sampled view (default 10 physics updates per control action)
        samples_per_control = int(trace.get("physics_samples_per_control", 10))
        if samples_per_control < 1:
            samples_per_control = 1
        if samples_per_control > 1:
            control_tag = f"tip_velocity_acceleration_controlstep{samples_per_control}.png"
            DvrkSCPMPPITraceScene._save_tip_velocity_acceleration(
                time=time,
                position=position,
                samples_per_control=samples_per_control,
                output_path=RUN_DIR / control_tag,
            )

    @staticmethod
    def _save_tip_velocity_acceleration(time, position, samples_per_control, output_path):
        time = time - time[0]

        if samples_per_control > 1:
            idx = np.arange(0, len(time), samples_per_control)
            if idx[-1] != len(time) - 1:
                idx = np.r_[idx, len(time) - 1]
            time = time[idx]
            position = position[idx]
            boundary_time = np.array([])
            title_prefix = f"(control-step={samples_per_control})"
            suffix = " / control-step sampled"
        else:
            boundary_time = time[np.arange(int(1), len(time), int(1))]
            title_prefix = "(physics rate)"
            suffix = ""

        velocity = np.gradient(position, time, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, time, axis=0, edge_order=2)

        figure, (velocity_axis, acceleration_axis) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            constrained_layout=True,
        )
        DvrkSCPMPPITraceScene._plot_vector_signal(
            velocity_axis,
            time,
            velocity * 1000.0,
            boundary_time,
            f"Tip velocity (world frame) {title_prefix}",
            "Velocity [mm/s]",
        )
        DvrkSCPMPPITraceScene._plot_vector_signal(
            acceleration_axis,
            time,
            acceleration * 1000.0,
            boundary_time,
            f"Tip acceleration (world frame) {title_prefix}{suffix}",
            "Acceleration [mm/s²]",
        )
        acceleration_axis.set_xlabel("Simulation time [s]")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        print(f"Saved tip kinematics plot: {output_path}")

    @staticmethod
    def _plot_vector_signal(axis, time, signal, boundary_time, title, ylabel):
        colors = ("#4C78A8", "#F58518", "#54A24B")
        for component, color, label in zip(range(3), colors, ("x", "y", "z")):
            axis.plot(time, signal[:, component], color=color, linewidth=1.2, alpha=0.85, label=label)
        axis.plot(
            time,
            np.linalg.vector_norm(signal, axis=1),
            color="#E45756",
            linewidth=2.0,
            label="norm",
        )
        for boundary in boundary_time:
            axis.axvline(boundary, color="#777777", linewidth=0.5, alpha=0.18)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=4, fontsize=9)
