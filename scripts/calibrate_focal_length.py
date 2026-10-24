#!/usr/bin/env python3
"""Compute focal length in pixels from one known object/distance measurement."""
from __future__ import annotations

import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-width-m", type=float, required=True, help="Actual width of the reference object in metres")
    parser.add_argument("--distance-m", type=float, required=True, help="Distance from camera sensor plane to object in metres")
    parser.add_argument("--bbox-width-px", type=float, required=True, help="Reference object's bounding-box width in the calibration image")
    parser.add_argument("--image-width-px", type=int, required=True, help="Calibration image width in pixels")
    args = parser.parse_args()
    if min(args.known_width_m, args.distance_m, args.bbox_width_px, args.image_width_px) <= 0:
        parser.error("all dimensions and distances must be positive")

    focal_length_px = args.bbox_width_px * args.distance_m / args.known_width_m
    hfov_deg = math.degrees(2.0 * math.atan((args.image_width_px / 2.0) / focal_length_px))
    print("Estimated camera intrinsics from the supplied reference measurement:")
    print(f"focal_length_px: {focal_length_px:.3f}")
    print(f"horizontal_fov_deg: {hfov_deg:.3f}")
    print("\nCopy the focal length into configs/default.yaml under camera.focal_length_px.")
    print("Use the same camera, zoom, and image resolution during inference as during calibration.")


if __name__ == "__main__":
    main()
