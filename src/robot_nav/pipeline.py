"""Glue code for detection, monocular distance estimation, and annotation."""
from __future__ import annotations

from typing import Any

import numpy as np

from .distance import DistanceEstimator
from .detector import ObjectDetector
from .visualization import annotate_frame


class NavigationPerceptionPipeline:
    """A single entry point for frame-by-frame robotics perception."""

    def __init__(self, detector: ObjectDetector, distance_estimator: DistanceEstimator) -> None:
        self.detector = detector
        self.distance_estimator = distance_estimator

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        detections = self.detector.detect(frame)
        image_height, image_width = frame.shape[:2]
        del image_height  # width is used by the pinhole distance formula

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            bbox_width = max(0.0, x2 - x1)
            distance_m = self.distance_estimator.estimate_distance_m(
                class_name=detection["class_name"],
                bbox_width_px=bbox_width,
                image_width_px=image_width,
            )
            detection["distance_m"] = distance_m

        return annotate_frame(frame, detections), detections
