"""
Single-image / batch inference CLI.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import VisionPredictor
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Vision Inference on an Image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K predicted candidate classes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("VisionInference")

    predictor = VisionPredictor(
        checkpoint_path=args.checkpoint,
    )

    result = predictor.predict(image_input=args.image, top_k=args.top_k)
    logger.info("Inference Result:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
