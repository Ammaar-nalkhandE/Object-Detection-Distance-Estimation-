#!/usr/bin/env python3
"""Batch detection + relative depth for robot navigation test images.

Default input: test_imgs/
Default YOLO weights: runs/detect/robot-nav-baseline2/weights/best.pt
Outputs: annotated images and detections.csv under runs/detect/distance_estimation/

Relative depth is NOT distance in metres. Metric estimates require a camera calibration
NPZ (made with OpenCV calibrateCamera) or a known focal length plus measured object width.
No webcam or GUI window is used by this script.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/robot-nav-baseline2/weights/best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def args_parser():
    p = argparse.ArgumentParser(description="Detect cones/stop signs and estimate relative depth on images.")
    p.add_argument("--source", default="test_imgs", help="Image path or folder of images (default: test_imgs).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained YOLO .pt weights.")
    p.add_argument("--output-dir", default="runs/detect/distance_estimation", help="Folder for annotated images and CSV.")
    p.add_argument("--device", default="auto", help="auto, cpu, or CUDA index such as 0.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--distance-mode", choices=("auto", "relative", "metric", "none"), default="auto",
                   help="auto uses relative depth unless camera calibration/focal length is supplied.")
    p.add_argument("--relative-model", choices=("MiDaS_small", "DPT_Hybrid", "DPT_Large"), default="MiDaS_small")
    p.add_argument("--calibration", default=None, help="OpenCV calibration NPZ containing camera_matrix, dist_coeffs, image_width, image_height.")
    p.add_argument("--fx-px", type=float, default=None, help="Known focal length in pixels at the input image resolution.")
    p.add_argument("--cone-width-m", type=float, default=0.30, help="Measured real width of the cone in metres.")
    p.add_argument("--stop-sign-width-m", type=float, default=0.60, help="Measured real width of the stop sign in metres.")
    p.add_argument("--warmup-frames", type=int, default=3, help="Initial YOLO frames excluded from timing stats when possible.")
    return p.parse_args()


def select_device(value):
    if value.lower() == "auto":
        if torch.cuda.is_available():
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
            return 0
        print("[INFO] CUDA unavailable; using CPU.")
        return "cpu"
    if value.lower() == "cpu":
        return "cpu"
    return int(value) if value.isdigit() else value


def class_label(raw_name, class_id):
    name = str(raw_name).lower().replace("-", "_").replace(" ", "_")
    if "cone" in name:
        return "cone"
    if "stop" in name:
        return "stop_sign"
    # This project trained class 0=cone, class 1=stop_sign.
    return {0: "cone", 1: "stop_sign"}.get(class_id, name or f"class_{class_id}")


def load_depth_model(model_name, device):
    print(f"[INFO] Loading {model_name} for relative depth (first run may download weights).")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            "MiDaS could not be loaded. Check internet access for its first run, or use --distance-mode none for detection only.\n"
            f"Original error: {exc}"
        ) from exc
    td = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    model.to(td).eval()
    transform = transforms.dpt_transform if model_name in ("DPT_Large", "DPT_Hybrid") else transforms.small_transform
    return model, transform, td


@torch.inference_mode()
def estimate_relative_depth(frame, depth_pack):
    model, transform, device = depth_pack
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    pred = model(batch)
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze(1).squeeze(0)
    depth = pred.detach().float().cpu().numpy()
    good = np.isfinite(depth)
    if not good.any():
        return np.zeros(depth.shape, dtype=np.float32)
    lo, hi = np.percentile(depth[good], [2, 98])
    if hi - lo < 1e-8:
        return np.zeros(depth.shape, dtype=np.float32)
    # MiDaS output is usually inverse-depth-like: higher generally means closer.
    out = np.clip((depth - lo) / (hi - lo), 0, 1)
    out[~good] = 0
    return out.astype(np.float32)


def object_relative_score(depth_map, box):
    if depth_map is None:
        return None
    x1, y1, x2, y2 = box
    h, w = depth_map.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    # Use the central portion of the box to reduce background contamination.
    dx, dy = int((x2-x1)*0.2), int((y2-y1)*0.2)
    crop = depth_map[y1+dy:y2-dy or y2, x1+dx:x2-dx or x2]
    return float(np.median(crop)) if crop.size else float(np.median(depth_map[y1:y2, x1:x2]))


def read_camera_matrix(calibration_path, fx_override, frame_size):
    width, height = frame_size
    if calibration_path:
        path = Path(calibration_path)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with np.load(path) as data:
            matrix = data["camera_matrix"].astype(np.float64)
            dist = data["dist_coeffs"].astype(np.float64) if "dist_coeffs" in data else None
            old_w = int(data["image_width"]) if "image_width" in data else width
            old_h = int(data["image_height"]) if "image_height" in data else height
        sx, sy = width / old_w, height / old_h
        matrix[0, 0] *= sx; matrix[0, 2] *= sx
        matrix[1, 1] *= sy; matrix[1, 2] *= sy
        return matrix, dist
    if fx_override is not None:
        if fx_override <= 0:
            raise ValueError("--fx-px must be positive")
        return np.array([[fx_override, 0, width/2], [0, fx_override, height/2], [0, 0, 1]], dtype=np.float64), None
    return None, None


def estimate_metric_distance(class_name, bbox_width_px, fx_px, cli_args):
    if fx_px is None or bbox_width_px <= 1:
        return None
    real_width = cli_args.cone_width_m if class_name == "cone" else cli_args.stop_sign_width_m if class_name == "stop_sign" else None
    if real_width is None:
        return None
    return real_width * fx_px / bbox_width_px


def process_image(path, root, model, names, device, cli_args, depth_pack, out_dir, timing):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        return []
    h, w = frame.shape[:2]
    matrix = dist = None
    if cli_args.distance_mode == "metric":
        matrix, dist = read_camera_matrix(cli_args.calibration, cli_args.fx_px, (w, h))
        if dist is not None:
            frame = cv2.undistort(frame, matrix, dist, None, matrix)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = model.predict(frame, imgsz=cli_args.imgsz, conf=cli_args.conf, device=device, verbose=False)[0]
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()
    timing.append(time.perf_counter() - t0)
    detections = []
    for box in result.boxes or []:
        xyxy = tuple(int(v) for v in box.xyxy[0].detach().cpu().tolist())
        cid = int(box.cls[0].item())
        label = class_label(names.get(cid, f"class_{cid}"), cid)
        confidence = float(box.conf[0].item())
        bw = max(0, xyxy[2] - xyxy[0])
        detections.append({"image": str(path.relative_to(root)) if root else path.name, "class": label,
                           "confidence": confidence, "box": xyxy, "bbox_width_px": bw,
                           "distance_m": estimate_metric_distance(label, bw, float(matrix[0, 0]) if matrix is not None else cli_args.fx_px, cli_args) if cli_args.distance_mode == "metric" else None,
                           "relative_score": None})
    depth_map = estimate_relative_depth(frame, depth_pack) if depth_pack else None
    for det in detections:
        det["relative_score"] = object_relative_score(depth_map, det["box"])

    # Rank relative depths only within this image; never interpret them as metres.
    valid = [d for d in detections if d["relative_score"] is not None]
    valid.sort(key=lambda d: d["relative_score"], reverse=True)
    labels = {}
    for i, det in enumerate(valid):
        if len(valid) == 1:
            labels[id(det)] = "relative only"
        elif i < max(1, len(valid)//3):
            labels[id(det)] = "nearer (relative)"
        elif i >= max(1, (2*len(valid))//3):
            labels[id(det)] = "farther (relative)"
        else:
            labels[id(det)] = "middle (relative)"

    output = frame.copy()
    for det in detections:
        x1,y1,x2,y2 = det["box"]
        color = (0, 220, 0) if det["class"] == "cone" else (0, 165, 255)
        cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
        cv2.putText(output, f'{det["class"]} {det["confidence"]:.2f}', (x1, max(22,y1-30)), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 2, cv2.LINE_AA)
        if cli_args.distance_mode == "metric":
            dm = det["distance_m"]
            label = f"Distance: {dm:.2f} m (approx.)" if dm is not None else "Distance: N/A"
        elif cli_args.distance_mode == "relative":
            score = det["relative_score"]
            label = f'Relative: {score:.2f} ({labels.get(id(det), "relative only")})' if score is not None else "Relative depth: N/A"
        else:
            label = "Distance disabled"
        cv2.putText(output, label, (x1, min(h-8, max(44,y1-7))), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 2, cv2.LINE_AA)
    if cli_args.distance_mode == "relative":
        cv2.putText(output, "RELATIVE DEPTH ONLY - NOT METRES", (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,255), 1, cv2.LINE_AA)

    relative = path.relative_to(root) if root else Path(path.name)
    out_path = out_dir / relative.parent / f"annotated_{relative.stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), output):
        print(f"[WARN] Could not write {out_path}")
    print(f"[OK] {relative}: {len(detections)} detection(s) -> {out_path}")
    rows = []
    for det in detections:
        x1,y1,x2,y2 = det["box"]
        score = det["relative_score"]
        dm = det["distance_m"]
        rows.append({"image": det["image"], "class": det["class"], "confidence": f'{det["confidence"]:.6f}',
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2, "bbox_width_px": det["bbox_width_px"],
                     "estimated_distance_m": f"{dm:.4f}" if dm is not None else "",
                     "relative_depth_score": f"{score:.4f}" if score is not None else "",
                     "relative_depth_label": labels.get(id(det), "relative depth unavailable"),
                     "distance_status": "approximate_metric" if dm is not None else "relative_only_not_metres" if score is not None else "disabled_or_unavailable"})
    return rows


def main():
    cli_args = args_parser()
    source = Path(cli_args.source).expanduser()
    model_path = Path(cli_args.model).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}. Check --source.")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}. Run from project root or pass --model.")
    if not 0 < cli_args.conf < 1 or cli_args.imgsz <= 0:
        raise ValueError("--conf must be between 0 and 1 and --imgsz must be positive")
    if cli_args.cone_width_m <= 0 or cli_args.stop_sign_width_m <= 0:
        raise ValueError("Object widths must be positive")
    metric_available = cli_args.calibration is not None or cli_args.fx_px is not None
    if cli_args.distance_mode == "auto":
        cli_args.distance_mode = "metric" if metric_available else "relative"
    if cli_args.distance_mode == "metric" and not metric_available:
        raise ValueError("Metric mode requires --calibration or --fx-px. Otherwise use --distance-mode relative.")
    if cli_args.calibration and cli_args.fx_px is not None:
        print("[INFO] Both calibration and --fx-px were supplied; camera calibration takes precedence.")
    device = select_device(cli_args.device)
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"[INFO] YOLO weights: {model_path}")
    print(f"[INFO] Classes: {names}")
    print(f"[INFO] Device: {device}; distance mode: {cli_args.distance_mode}")
    depth_pack = load_depth_model(cli_args.relative_model, device) if cli_args.distance_mode == "relative" else None
    if cli_args.distance_mode == "metric":
        print(f"[WARN] Metric estimates assume cone width={cli_args.cone_width_m}m and stop-sign width={cli_args.stop_sign_width_m}m. Measure your actual objects.")
    if source.is_dir():
        root = source
        image_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        root = None
        image_paths = [source]
    else:
        raise ValueError("This single-file version accepts one image or a folder of images. For your project, use --source test_imgs.")
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {source}")
    out_dir = Path(cli_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing, all_rows = [], []
    start = time.perf_counter()
    for i, path in enumerate(image_paths, 1):
        all_rows.extend(process_image(path, root, model, names, device, cli_args, depth_pack, out_dir, timing))
        print(f"[{i}/{len(image_paths)}]")
    csv_path = out_dir / "detections.csv"
    columns = ["image", "class", "confidence", "x1", "y1", "x2", "y2", "bbox_width_px", "estimated_distance_m", "relative_depth_score", "relative_depth_label", "distance_status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(all_rows)
    usable = timing[cli_args.warmup_frames:] if len(timing) > cli_args.warmup_frames else timing
    print(f"\n[SAVED] Annotated images: {out_dir.resolve()}")
    print(f"[SAVED] CSV report: {csv_path.resolve()}")
    print(f"Images processed: {len(image_paths)}; detections: {len(all_rows)}")
    if usable:
        mean_ms = 1000 * float(np.mean(usable))
        median_ms = 1000 * float(np.median(usable))
        print(f"Mean YOLO inference: {mean_ms:.2f} ms/image")
        print(f"Median YOLO inference: {median_ms:.2f} ms/image")
        print(f"YOLO throughput: {1000/mean_ms:.2f} FPS (approx.; excludes depth-model time)")
    print(f"Total elapsed time: {time.perf_counter()-start:.2f} s")
    if cli_args.distance_mode == "relative":
        print("NOTE: relative-depth scores are not metres and are normalized independently for each image.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise