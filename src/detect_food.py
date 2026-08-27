"""
Food-waste object detection prototype.

This module provides a simple inference pipeline using a pretrained
YOLO model. The prototype is designed as a computational component
of an AI-assisted smart-bin system.
"""

from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = "yolo11n.pt"


def load_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    """Load a pretrained YOLO object-detection model."""
    return YOLO(model_name)


def detect_food(
    image_path: str,
    model_name: str = DEFAULT_MODEL,
    confidence: float = 0.25,
    output_dir: str = "results/predictions",
):
    """
    Run object detection on a food-waste image.

    Parameters
    ----------
    image_path:
        Path to the input image.
    model_name:
        YOLO model used for inference.
    confidence:
        Minimum confidence threshold.
    output_dir:
        Directory where prediction images are saved.

    Returns
    -------
    results:
        YOLO detection results.
    """
    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)

    results = model.predict(
        source=str(image),
        conf=confidence,
        save=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run food-waste object detection."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )

    args = parser.parse_args()

    detect_food(
        image_path=args.image,
        confidence=args.confidence,
    )
