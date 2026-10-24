#!/usr/bin/env python3
"""Run detection + approximate distance estimation on an image/video/webcam."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_nav.config import load_config  # noqa: E402
from robot_nav.detector import ObjectDetector  # noqa: E402
from robot_nav.distance import DistanceEstimator  # noqa: E402
from robot_nav.pipeline import NavigationPerceptionPipeline  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image, image directory, video path, stream URL, or webcam index (0)")
    parser.add_argument("--model", default="runs/detect/robot-nav-yolo/weights/best.pt", help="Trained YOLO .pt model or supported exported model")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/predictions"))
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold from config")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size from config")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, or 0")
    parser.add_argument("--show", action="store_true", help="Display the output in a window")
    return parser.parse_args()


def make_pipeline(args: argparse.Namespace) -> NavigationPerceptionPipeline:
    config = load_config(args.config)
    inference = config.get("inference", {})
    detector = ObjectDetector(
        model_path=args.model,
        confidence=args.conf if args.conf is not None else float(inference.get("confidence", 0.25)),
        image_size=args.imgsz if args.imgsz is not None else int(inference.get("image_size", 640)),
        device=args.device if args.device is not None else inference.get("device", "auto"),
    )
    camera = config.get("camera", {})
    estimator = DistanceEstimator(
        object_widths_m=config.get("object_widths_m", {}),
        horizontal_fov_deg=float(camera.get("horizontal_fov_deg", 70.0)),
        focal_length_px=camera.get("focal_length_px"),
    )
    return NavigationPerceptionPipeline(detector, estimator)


def process_image(path: Path, pipeline: NavigationPerceptionPipeline, output_dir: Path, show: bool) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[WARN] Could not read image: {path}")
        return
    annotated, detections = pipeline.process(frame)
    output_path = output_dir / f"{path.stem}_annotated.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not write output image: {output_path}")
    print(f"Saved {output_path} ({len(detections)} detections)")
    if show:
        cv2.imshow("Object detection + distance", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(source: str, pipeline: NavigationPerceptionPipeline, output_dir: Path, show: bool) -> None:
    capture_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("The source did not report valid frame dimensions")

    stem = "webcam" if source.isdigit() else Path(source).stem or "stream"
    output_path = output_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, detections = pipeline.process(frame)
            writer.write(annotated)
            frame_count += 1
            if show:
                cv2.imshow("Object detection + distance", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    print(f"Saved {output_path} ({frame_count} frames)")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = make_pipeline(args)

    if args.source.isdigit():
        process_stream(args.source, pipeline, output_dir, args.show)
        return

    source_path = Path(args.source).expanduser()
    if source_path.is_dir():
        images = sorted(path for path in source_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No supported image files found in {source_path}")
        for image_path in images:
            process_image(image_path, pipeline, output_dir, args.show)
    elif source_path.exists() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        process_image(source_path, pipeline, output_dir, args.show)
    else:
        process_stream(args.source, pipeline, output_dir, args.show)


if __name__ == "__main__":
    main()
