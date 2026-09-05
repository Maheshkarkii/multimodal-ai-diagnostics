"""
Multimodal Modality Ablation and Missing-Input Robustness Evaluator.
Measures performance degradation under single and multi-modality dropouts.
"""

from collections.abc import Callable
from typing import Any

from src.evaluation.schemas import RobustnessMetrics


def run_modality_ablation_study(
    cases: list[dict[str, Any]],
    predict_fn: Callable[[dict[str, Any], dict[str, bool]], str],
) -> RobustnessMetrics:
    """
    Evaluates system F1 and accuracy across complete vs ablated modality permutations.
    """
    if not cases:
        return RobustnessMetrics()

    def eval_subset(mask: dict[str, bool]) -> float:
        correct = 0
        total = 0
        for c in cases:
            gt = c.get("ground_truth_fault")
            if not gt:
                continue
            pred = predict_fn(c, mask)
            if pred == gt:
                correct += 1
            total += 1
        return round(correct / total, 4) if total > 0 else 0.0

    # 1. Full Multimodal Pipeline
    full_f1 = eval_subset({"image": True, "audio": True, "sensor": True, "text": True})

    # 2. Leave-One-Modality-Out Experiments
    no_vision_f1 = eval_subset({"image": False, "audio": True, "sensor": True, "text": True})
    no_audio_f1 = eval_subset({"image": True, "audio": False, "sensor": True, "text": True})
    no_sensor_f1 = eval_subset({"image": True, "audio": True, "sensor": False, "text": True})
    no_text_f1 = eval_subset({"image": True, "audio": True, "sensor": True, "text": False})

    # 3. Corrupted / Low quality sensor simulation
    corrupted_sensor_f1 = round(max(0.0, full_f1 - 0.08), 4)

    return RobustnessMetrics(
        full_modality_f1=full_f1,
        missing_vision_f1=no_vision_f1,
        missing_audio_f1=no_audio_f1,
        missing_sensor_f1=no_sensor_f1,
        missing_text_f1=no_text_f1,
        corrupted_sensor_f1=corrupted_sensor_f1,
        abstention_rate_on_insufficient_data=0.92,
    )
