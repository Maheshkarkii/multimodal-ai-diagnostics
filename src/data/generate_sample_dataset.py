"""
Sample Industrial Fault Dataset Generator.

Generates verified, synthetic structural inspection patterns (Corrosion, Cracks, Bearing Faults, Damaged Parts, Normal)
with equipment IDs for deterministic testing and reproducible experiment execution when external raw industrial data is not mounted.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_industrial_dataset(
    output_dir: Path,
    samples_per_class: int = 40,
    seed: int = 42,
) -> Path:
    """Generate structured synthetic industrial visual inspection dataset with machine metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    classes = ["normal", "bearing_fault", "corrosion", "surface_crack", "damaged_component"]

    for class_name in classes:
        c_dir = output_dir / class_name
        c_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            eq_id = f"machine_{(i % 8) + 1:02d}"
            img_path = c_dir / f"{eq_id}_{class_name}_{i:03d}.png"

            # Base metallic component pattern (224x224 RGB)
            base_color = rng.integers(120, 180, size=3)
            img_arr = np.ones((224, 224, 3), dtype=np.uint8) * base_color.astype(np.uint8)
            noise = rng.normal(0, 12, (224, 224, 3)).astype(np.int16)
            img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)

            # Draw class-specific visual signatures
            if class_name == "surface_crack":
                # Dark jagged line
                points = [(rng.integers(20, 80), rng.integers(20, 80))]
                for _ in range(4):
                    points.append((points[-1][0] + rng.integers(20, 40), points[-1][1] + rng.integers(15, 35)))
                draw.line(points, fill=(20, 20, 20), width=3)
            elif class_name == "corrosion":
                # Rust patch
                for _ in range(6):
                    cx, cy = rng.integers(40, 180), rng.integers(40, 180)
                    r = rng.integers(15, 35)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(160, 70, 30))
            elif class_name == "bearing_fault":
                # Concentric ring with spalling marks
                draw.ellipse([50, 50, 174, 174], outline=(50, 50, 50), width=6)
                draw.ellipse([80, 80, 144, 144], outline=(80, 80, 80), width=4)
                # Spall defects
                for _ in range(3):
                    sx, sy = rng.integers(60, 160), rng.integers(60, 160)
                    draw.rectangle([sx, sy, sx + 8, sy + 8], fill=(30, 30, 30))
            elif class_name == "damaged_component":
                # Missing chunk / fracture
                draw.polygon([(150, 150), (224, 130), (224, 224), (130, 224)], fill=(10, 10, 10))
            # "normal" remains clean metallic base

            img.save(img_path)

    return output_dir


if __name__ == "__main__":
    generate_synthetic_industrial_dataset("data/industrial_inspection", samples_per_class=50)
