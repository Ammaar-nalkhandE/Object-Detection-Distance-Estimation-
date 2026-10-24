#!/usr/bin/env python3
"""Export a trained YOLO checkpoint to an edge-friendly deployment format."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/detect/robot-nav-yolo/weights/best.pt")
    parser.add_argument("--format", choices=["onnx", "openvino", "engine", "torchscript"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="Request INT8 quantization when supported by the selected exporter")
    parser.add_argument("--data", default=None, help="Representative YOLO dataset YAML required for INT8 calibration")
    parser.add_argument("--device", default=None, help="Optional export device, e.g. 0 for TensorRT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).expanduser().exists() and not args.model.endswith(".pt"):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.int8 and not args.data:
        raise ValueError("INT8 export generally needs representative calibration data; provide --data data/dataset.yaml")

    model = YOLO(args.model)
    export_args = {"format": args.format, "imgsz": args.imgsz}
    if args.int8:
        export_args["int8"] = True
        export_args["data"] = str(Path(args.data).expanduser().resolve())
    if args.device is not None:
        export_args["device"] = args.device

    exported_path = model.export(**export_args)
    print(f"Export complete: {exported_path}")
    if not args.int8:
        print("This was a non-INT8 export. Benchmark it against the original .pt checkpoint before selecting a deployment artifact.")
    else:
        print("INT8 calibration/export was requested. Re-run validation and FPS measurements on the exported artifact.")


if __name__ == "__main__":
    main()
