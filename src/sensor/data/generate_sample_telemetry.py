"""
Sample Industrial Multivariate Sensor Telemetry Generator.

Generates verified continuous machine telemetry matching physical industrial dynamics:
Features:
- temperature_c: Baseline 45-65 C, spikes during bearing wear (85-110 C)
- vibration_rms_g: Baseline 1.0-2.5 g, surges during unbalance (6.0-12.0 g)
- rotational_speed_rpm: Nominal 1500 RPM, dips during overload/unbalance (1350-1420 RPM)
- motor_current_a: Nominal 8-12 A, spikes during overcurrent/faults (18-28 A)
- hydraulic_pressure_bar: Nominal 120-160 bar, collapses during pressure loss (20-60 bar)
- load_percentage: Nominal 50-85%
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_telemetry_dataset(
    output_path: Path,
    num_machines: int = 8,
    records_per_machine: int = 100,
    seed: int = 42,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fault_modes = [
        "normal",
        "bearing_overheat_wear",
        "rotor_unbalance",
        "hydraulic_pressure_loss",
        "electrical_overcurrent",
    ]

    records = []
    base_timestamp = pd.Timestamp("2026-09-01 08:00:00")

    for m_idx in range(1, num_machines + 1):
        machine_id = f"pump_unit_{m_idx:02d}"

        for r_idx in range(records_per_machine):
            ts = base_timestamp + pd.Timedelta(minutes=r_idx * 15)

            # Assign class distribution per machine
            mode = rng.choice(fault_modes, p=[0.40, 0.15, 0.15, 0.15, 0.15])

            # Default normal operating baseline
            temp = rng.normal(55.0, 3.5)
            vib = rng.normal(1.8, 0.3)
            rpm = rng.normal(1495.0, 15.0)
            curr = rng.normal(10.2, 0.8)
            press = rng.normal(140.0, 5.0)
            load = rng.normal(68.0, 6.0)

            # Apply physics-informed fault signatures
            if mode == "bearing_overheat_wear":
                temp += rng.uniform(30.0, 50.0)  # 85-105 C
                vib += rng.uniform(2.5, 5.0)
            elif mode == "rotor_unbalance":
                vib += rng.uniform(5.5, 9.0)    # 7-11 g
                rpm -= rng.uniform(50.0, 120.0)
            elif mode == "hydraulic_pressure_loss":
                press = rng.uniform(25.0, 65.0)  # Severe pressure collapse
                load -= rng.uniform(20.0, 35.0)
            elif mode == "electrical_overcurrent":
                curr += rng.uniform(10.0, 18.0)  # 20-28 A
                temp += rng.uniform(15.0, 25.0)

            records.append({
                "timestamp": str(ts),
                "machine_id": machine_id,
                "temperature_c": round(float(temp), 2),
                "vibration_rms_g": round(float(vib), 3),
                "rotational_speed_rpm": round(float(rpm), 1),
                "motor_current_a": round(float(curr), 2),
                "hydraulic_pressure_bar": round(float(press), 2),
                "load_percentage": round(float(np.clip(load, 0, 100)), 1),
                "fault_label": mode,
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    generate_synthetic_telemetry_dataset("data/sensor/raw/machine_telemetry.csv")
