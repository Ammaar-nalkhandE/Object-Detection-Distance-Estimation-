#!/usr/bin/env python3
"""Measure model inference latency/FPS on CPU and/or available GPU."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_nav.detector import resolve_device  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/detect/robot-nav-yolo/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image, folder of images, or video file")
    parser.add_argument("--device", choices=["cpu", "gpu", "both", "auto"], default="both")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=100, help="Number of measured inference iterations per device")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum frames loaded from a video/folder")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "outputs/benchmark.csv"))
    return parser.parse_args()


def load_frames(source: str, max_frames: int) -> list:
    path = Path(source).expanduser()
    frames = []
    if path.is_dir():
        for image_path in sorted(path.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            frame = cv2.imread(str(image_path))
            if frame is not None:
                frames.append(frame)
            if len(frames) >= max_frames:
                break
    elif path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open benchmark source: {source}")
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
    if not frames:
        raise ValueError(f"No readable frames found at source: {source}")
    return frames


def benchmark_device(args: argparse.Namespace, device_name: str, frames: list) -> dict:
    device = resolve_device(device_name)
    model = YOLO(args.model)
    # Warm-up is not included in the timing to reduce one-time setup effects.
    for index in range(args.warmup):
        model.predict(frames[index % len(frames)], imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
    if isinstance(device, int) and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    elapsed_start = time.perf_counter()
    for index in range(args.iterations):
        frame = frames[index % len(frames)]
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
    if isinstance(device, int) and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - elapsed_start

    result = {
        "device": device_name,
        "iterations": args.iterations,
        "elapsed_seconds": round(elapsed, 4),
        "average_latency_ms": round((elapsed / args.iterations) * 1000.0, 2),
        "fps": round(args.iterations / elapsed, 2) if elapsed > 0 else 0.0,
        "model": args.model,
        "imgsz": args.imgsz,
    }
    print(
        f"{device_name.upper():>4} | FPS: {result['fps']:.2f} | "
        f"avg inference: {result['average_latency_ms']:.2f} ms | "
        f"iterations: {args.iterations}"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.max_frames < 1:
        raise ValueError("iterations and max-frames must be positive; warmup cannot be negative")
    frames = load_frames(args.source, args.max_frames)
    devices: list[str]
    if args.device == "both":
        devices = ["cpu"] + (["gpu"] if torch.cuda.is_available() else [])
        if not torch.cuda.is_available():
            print("[INFO] CUDA unavailable; benchmarking CPU only.")
    elif args.device == "auto":
        devices = ["gpu" if torch.cuda.is_available() else "cpu"]
    elif args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but PyTorch reports no CUDA device")
    else:
        devices = [args.device]

    results = [benchmark_device(args, device, frames) for device in devices]
    csv_path = Path(args.csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved benchmark results: {csv_path}")
    print("FPS here measures repeated model prediction on in-memory frames, not full camera-to-actuator latency.")


if __name__ == "__main__":
    main()
