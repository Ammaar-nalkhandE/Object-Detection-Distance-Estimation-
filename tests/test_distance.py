import math

import pytest

from robot_nav.distance import DistanceEstimator


def test_distance_uses_pinhole_width_formula():
    estimator = DistanceEstimator(
        object_widths_m={"cone": 0.30},
        focal_length_px=800.0,
    )
    # D = W * f / w = 0.30 * 800 / 120 = 2 metres
    assert estimator.estimate_distance_m("cone", 120.0, 1280) == pytest.approx(2.0)


def test_class_name_normalization():
    estimator = DistanceEstimator({"stop_sign": 0.75}, focal_length_px=600.0)
    assert estimator.estimate_distance_m("Stop Sign", 150.0, 640) == pytest.approx(3.0)


def test_missing_width_returns_none():
    estimator = DistanceEstimator({"cone": 0.30}, focal_length_px=800.0)
    assert estimator.estimate_distance_m("person", 80.0, 1280) is None


def test_fov_fallback_calculates_focal_length():
    estimator = DistanceEstimator({"cone": 0.30}, horizontal_fov_deg=90.0)
    assert estimator.focal_length_for_image(640) == pytest.approx(320.0)
    assert estimator.estimate_distance_m("cone", 48.0, 640) == pytest.approx(2.0)


def test_invalid_fov_rejected():
    with pytest.raises(ValueError):
        DistanceEstimator({"cone": 0.30}, horizontal_fov_deg=180.0)


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        DistanceEstimator({"cone": 0.0}, focal_length_px=600.0)
