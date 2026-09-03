"""
Inference CLI with Multimodal Feature Vector Output.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import VisionPredictor
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Vision Inference on an Equipment Image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input equipment image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/exp_frozen_baseline_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K predicted candidate classes",
    )
    parser.add_argument(
        "--extract-embedding",
        action="store_true",
        help="Whether to return the 1280-dim feature embedding for multimodal fusion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("VisionInference")

    predictor = VisionPredictor(
        checkpoint_path=args.checkpoint,
    )

    result = predictor.predict(
        image_input=args.image,
        top_k=args.top_k,
        return_embedding=args.extract_embedding,
    )

    # Truncate embedding array for clean terminal output if extracted
    display_result = dict(result)
    if "feature_embedding" in display_result:
        emb = display_result["feature_embedding"]
        display_result["feature_embedding"] = f"[{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}] (length={len(emb)})"

    logger.info("Inference Result:\n%s", json.dumps(display_result, indent=2))


if __name__ == "__main__":
    main()
