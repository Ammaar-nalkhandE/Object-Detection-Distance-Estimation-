"""Bounding-box annotation utilities."""
from __future__ import annotations

import cv2
import numpy as np


def pretty_label(class_name: str) -> str:
    """Make underscore labels readable, e.g. stop_sign -> Stop Sign."""
    return " ".join(part.capitalize() for part in class_name.replace("-", "_").split("_"))


def annotate_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detections and distance labels onto a copy of a frame."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection["bbox"])
        class_name = pretty_label(str(detection["class_name"]))
        confidence = float(detection.get("confidence", 0.0))
        distance = detection.get("distance_m")
        distance_text = f"{distance:.1f}m" if isinstance(distance, (int, float)) else "N/A"
        label = f"{class_name}, {distance_text} ({confidence:.2f})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(annotated.shape[1] - 1, x1 + text_width + 8), y1),
            (60, 220, 60),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 2, y1 - baseline - 4)),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return annotated
