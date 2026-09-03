"""
Audio Inference CLI with Acoustic Feature Vector Extraction.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.inference.audio_predictor import AudioPredictor
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Acoustic Diagnostic Inference on a WAV file")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input .wav audio recording",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/acoustic_fault_baseline_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K predicted candidate anomaly classes",
    )
    parser.add_argument(
        "--extract-embedding",
        action="store_true",
        help="Whether to return the 512-dim acoustic feature embedding for multimodal fusion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("AudioInference")

    predictor = AudioPredictor(checkpoint_path=args.checkpoint)

    result = predictor.predict(
        audio_input=args.audio,
        top_k=args.top_k,
        return_embedding=args.extract_embedding,
    )

    display_result = dict(result)
    if "acoustic_embedding" in display_result:
        emb = display_result["acoustic_embedding"]
        display_result["acoustic_embedding"] = f"[{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}] (length={len(emb)})"

    logger.info("Acoustic Inference Result:\n%s", json.dumps(display_result, indent=2))


if __name__ == "__main__":
    main()
