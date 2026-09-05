"""
Sensor Explainability Module: Physical Telemetry Deviation & Envelope Visualizer.
Plots observed sensor values against documented warning and critical threshold boundaries.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_sensor_threshold_plot(
    sensor_measurements: list[dict[str, Any]],
    output_path: Path | None = None,
) -> str | None:
    """
    Plot bar chart of normalized sensor measurements relative to warning and critical limits.
    """
    if not sensor_measurements:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [m.get("parameter", "P") for m in sensor_measurements]
        values = [float(m.get("value", 0.0)) for m in sensor_measurements]
        warnings = [
            float(m.get("warning_threshold", 0.0)) if m.get("warning_threshold") is not None else 0.0
            for m in sensor_measurements
        ]
        criticals = [
            float(m.get("critical_threshold", 0.0)) if m.get("critical_threshold") is not None else 0.0
            for m in sensor_measurements
        ]

        x = np.arange(len(labels))
        width = 0.28

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width, values, width, label="Observed Value", color="dodgerblue")
        ax.bar(x, warnings, width, label="Warning Limit", color="orange", alpha=0.85)
        ax.bar(x + width, criticals, width, label="Critical Limit", color="crimson", alpha=0.85)

        ax.set_ylabel("Measurement Magnitude")
        ax.set_title("Telemetry Sensor Measurements vs. Operational Envelope")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=120)
            plt.close(fig)
            return str(output_path.resolve())

        plt.close(fig)
        return None

    except Exception as e:
        logger.warning(f"Sensor visualization failed: {e}")
        return None
