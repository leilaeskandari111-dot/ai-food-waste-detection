"""
AI-Based Food Waste Detection
Proof-of-concept computer-vision pipeline for detecting
food-related objects in images.
"""

from pathlib import Path
import argparse
import csv

from ultralytics import YOLO


# Food-related classes available in the pretrained model
FOOD_CLASSES = {
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
}


def detect_food(
    image_path: str,
    output_dir: str = "results",
    confidence: float = 0.35,
):
    """
    Detect food-related objects in an image.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Run object detection
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        verbose=False,
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            class_name = model.names[class_id]

            if class_name.lower() not in FOOD_CLASSES:
                continue

            x1, y1, x2, y2 = (
                float(value)
                for value in box.xyxy[0]
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(score, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )

        # Save annotated image
        annotated = result.plot()

        output_image = (
            output_dir /
            f"{image_path.stem}_detected.jpg"
        )

        import cv2

        cv2.imwrite(
            str(output_image),
            annotated
        )

    # Save CSV summary
    csv_path = (
        output_dir /
        f"{image_path.stem}_detections.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )

        writer.writeheader()
        writer.writerows(detections)

    # Print summary
    print("\nFood-waste detection summary")
    print("-----------------------------")

    if not detections:
        print("No food-related objects detected.")
    else:

        counts = {}

        for detection in detections:

            name = detection["class"]

            counts[name] = (
                counts.get(name, 0) + 1
            )

        for name, count in sorted(
            counts.items()
        ):
            print(f"{name}: {count}")

        print(
            f"\nTotal food-related objects: "
            f"{len(detections)}"
        )

    print(
        f"\nDetection results saved to: "
        f"{output_dir}"
    )

    return detections


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Detect food-related objects "
            "using a pretrained YOLO model."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image."
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Output directory."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold."
    )

    args = parser.parse_args()

    detect_food(
        image_path=args.image,
        output_dir=args.output,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
