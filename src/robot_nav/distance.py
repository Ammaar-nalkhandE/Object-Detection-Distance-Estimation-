"""Approximate monocular distance estimation using known object width.

For an object with known physical width W and observed bounding-box width w:
    distance = W * focal_length_px / w

This is a geometry-based estimate, not a substitute for calibrated stereo/depth sensing.
"""
from __future__ import annotations

import math
from typing import Mapping

from .config import normalize_class_name


class DistanceEstimator:
    """Estimate object distance in metres using known apparent object width."""

    def __init__(
        self,
        object_widths_m: Mapping[str, float],
        horizontal_fov_deg: float = 70.0,
        focal_length_px: float | None = None,
    ) -> None:
        if not 1.0 < float(horizontal_fov_deg) < 179.0:
            raise ValueError("horizontal_fov_deg must be between 1 and 179 degrees")
        self.object_widths_m = {
            normalize_class_name(str(key)): float(value)
            for key, value in object_widths_m.items()
        }
        if any(width <= 0 for width in self.object_widths_m.values()):
            raise ValueError("All object widths must be positive values in metres")
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.focal_length_px = float(focal_length_px) if focal_length_px is not None else None
        if self.focal_length_px is not None and self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive when supplied")

    def focal_length_for_image(self, image_width_px: int) -> float:
        """Return focal length in pixels, using calibration if available."""
        if image_width_px <= 0:
            raise ValueError("image_width_px must be positive")
        if self.focal_length_px is not None:
            return self.focal_length_px
        half_fov_rad = math.radians(self.horizontal_fov_deg / 2.0)
        return (image_width_px / 2.0) / math.tan(half_fov_rad)

    def estimate_distance_m(
        self,
        class_name: str,
        bbox_width_px: float,
        image_width_px: int,
    ) -> float | None:
        """Return approximate distance in metres, or None if class is unconfigured."""
        key = normalize_class_name(class_name)
        physical_width_m = self.object_widths_m.get(key)
        if physical_width_m is None or bbox_width_px <= 0 or image_width_px <= 0:
            return None
        focal_px = self.focal_length_for_image(image_width_px)
        distance_m = physical_width_m * focal_px / float(bbox_width_px)
        if not math.isfinite(distance_m) or distance_m <= 0:
            return None
        return distance_m
