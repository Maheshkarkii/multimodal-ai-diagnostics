"""
Multimodal Inference CLI with Flexible Input Evidence.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multimodal.inference.multimodal_predictor import MultimodalPredictor
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Multimodal Inference on Machine Evidence")
    parser.add_argument("--image", type=str, default=None, help="Path to component image (.png/.jpg)")
    parser.add_argument("--audio", type=str, default=None, help="Path to acoustic audio recording (.wav)")
    parser.add_argument("--sensor-json", type=str, default=None, help="JSON string with telemetry readings")
    parser.add_argument("--notes", type=str, default=None, help="Technician maintenance observation string")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_fusion_baseline_best.pt")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--extract-unified-embedding", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("MultimodalInference")

    sensor_dict = json.loads(args.sensor_json) if args.sensor_json else None

    # Fallback to sample evidence if none supplied
    if not args.image and not args.audio and not sensor_dict and not args.notes:
        logger.info("No inputs specified. Running demonstration query with Audio + Sensor + Notes...")
        sensor_dict = {
            "temperature_c": 94.0, "vibration_rms_g": 5.2, "rotational_speed_rpm": 1470.0,
            "motor_current_a": 10.5, "hydraulic_pressure_bar": 139.0, "load_percentage": 70.0
        }
        notes = "Audible high-pitched periodic chirping from drive-end bearing. Casing hot to touch."
        image_input = None
        audio_input = None
    else:
        notes = args.notes
        image_input = args.image
        audio_input = args.audio

    predictor = MultimodalPredictor(checkpoint_path=args.checkpoint)

    result = predictor.predict(
        image=image_input,
        audio=audio_input,
        sensor_data=sensor_dict,
        technician_notes=notes,
        top_k=args.top_k,
        return_unified_embedding=args.extract_unified_embedding,
    )

    display_result = dict(result)
    if "unified_machine_embedding" in display_result:
        emb = display_result["unified_machine_embedding"]
        display_result["unified_machine_embedding"] = f"[{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}] (length={len(emb)})"

    logger.info("Multimodal Diagnostic Result:\n%s", json.dumps(display_result, indent=2))


if __name__ == "__main__":
    main()
