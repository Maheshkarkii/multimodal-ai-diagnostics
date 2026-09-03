"""
Sensor Telemetry Inference CLI with Anomaly and Embedding Extraction.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensor.inference.sensor_predictor import SensorPredictor
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sensor State & Anomaly Inference on Telemetry Readings")
    parser.add_argument(
        "--json-input",
        type=str,
        default='{"temperature_c": 92.5, "vibration_rms_g": 6.8, "rotational_speed_rpm": 1420.0, "motor_current_a": 11.2, "hydraulic_pressure_bar": 138.0, "load_percentage": 75.0}',
        help="JSON string containing sensor measurements",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sensor_state_and_anomaly_baseline_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K predicted candidate states",
    )
    parser.add_argument(
        "--extract-embedding",
        action="store_true",
        help="Whether to return the 256-dim sensor feature embedding for multimodal fusion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("SensorInference")

    sensor_dict = json.loads(args.json_input)
    predictor = SensorPredictor(checkpoint_path=args.checkpoint)

    result = predictor.predict(
        sensor_input=sensor_dict,
        top_k=args.top_k,
        return_embedding=args.extract_embedding,
    )

    display_result = dict(result)
    if "sensor_embedding" in display_result:
        emb = display_result["sensor_embedding"]
        display_result["sensor_embedding"] = f"[{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}] (length={len(emb)})"

    logger.info("Sensor Inference Result:\n%s", json.dumps(display_result, indent=2))


if __name__ == "__main__":
    main()
