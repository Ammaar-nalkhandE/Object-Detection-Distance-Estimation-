#!/usr/bin/env python3
"""Fine-tune a pretrained YOLO model on the project's annotated dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_nav.detector import resolve_device  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data/dataset.yaml"), help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model checkpoint, e.g. yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or GPU index such as 0")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs/detect"))
    parser.add_argument("--name", default="robot-nav-yolo")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Resume from --model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, batch must be positive and imgsz must be at least 32")

    device = resolve_device(args.device)
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        pretrained=True,
        resume=args.resume,
        plots=True,
        exist_ok=True,
    )
    print(f"Training finished. Results: {results}")
    print(f"Best checkpoint is usually at: {Path(args.project) / args.name / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
