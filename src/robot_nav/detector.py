"""Ultralytics YOLO inference wrapper."""
from __future__ import annotations

from typing import Any

import torch
from ultralytics import YOLO


def resolve_device(device: str | int | None = "auto") -> str | int:
    """Convert the portable device selector to an Ultralytics device argument."""
    if device is None or str(device).lower() == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    text = str(device).lower()
    if text in {"cpu", "mps"}:
        if text == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available in this PyTorch installation")
        return text
    if text in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but torch.cuda.is_available() is False")
        return 0
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {text} was requested, but CUDA is unavailable")
        return int(text)
    raise ValueError("device must be 'auto', 'cpu', 'gpu', 'cuda', 'mps', or a CUDA index such as '0'")


class ObjectDetector:
    """Thin wrapper that returns consistent Python dictionaries per detection."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | int = "auto",
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = resolve_device(device)

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Run one image through YOLO and return pixel-space bounding boxes."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            coords = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            x1, y1, x2, y2 = (float(value) for value in coords)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                }
            )
        return detections
