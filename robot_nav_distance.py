#!/usr/bin/env python3
"""Object detection and monocular distance estimation for robot navigation.

Project model: runs/detect/robot-nav-baseline2/weights/best.pt
Classes expected from data_2class: 0=cone, 1=stop_sign

Distance formula: distance_m = (known_object_width_m * focal_length_px) / bbox_width_px
A calibrated camera and measured object dimensions are required for meaningful estimates.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/robot-nav-baseline2/weights/best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect cones/stop signs, estimate approximate monocular distance, and benchmark inference."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained YOLO weights.")
    parser.add_argument("--source", default="0", help="Image/video path, webcam index (0), or camera stream URL.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0 (first CUDA GPU), or another Ultralytics device value.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--output-dir", default="runs/detect/distance_estimation", help="Directory for output images/videos.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video/webcam stream as MP4.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV display window (useful on headless systems).")
    parser.add_argument("--camera-width", type=int, default=None, help="Optional webcam capture width, e.g. 1280.")
    parser.add_argument("--camera-height", type=int, default=None, help="Optional webcam capture height, e.g. 720.")
    parser.add_argument("--calibration", default=None, help="Camera calibration .npz created by this script.")
    parser.add_argument("--fx-px", type=float, default=None, help="Optional calibrated focal length in pixels at the active image resolution; fallback if no .npz is supplied.")
    parser.add_argument("--cone-width-m", type=float, default=0.30, help="ASSUMED real-world cone width in metres (default 0.30; measure your cone).")
    parser.add_argument("--stop-sign-width-m", type=float, default=0.60, help="ASSUMED real-world stop-sign width in metres (default 0.60; measure your sign).")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Initial stream frames excluded from benchmark averages (default: 5).")

    # Camera-calibration workflows are included in this same file.
    parser.add_argument("--capture-calibration", action="store_true", help="Open the camera and save checkerboard frames when you press 's'; press 'q' to exit.")
    parser.add_argument("--calibration-dir", default="calibration_images", help="Directory to store/read checkerboard calibration images.")
    parser.add_argument("--calibrate-dir", default=None, help="Calibrate camera using checkerboard photos in this directory, then exit.")
    parser.add_argument("--board-cols", type=int, default=9, help="Number of INNER checkerboard corners horizontally (default 9).")
    parser.add_argument("--board-rows", type=int, default=6, help="Number of INNER checkerboard corners vertically (default 6).")
    parser.add_argument("--square-size-m", type=float, default=0.025, help="Actual checkerboard square side in metres (default 0.025 = 25 mm).")
    parser.add_argument("--calibration-out", default="camera_calibration.npz", help="Output calibration .npz path.")
    return parser.parse_args()


def resolve_source(source_text: str) -> Any:
    """Convert a numeric source such as '0' into webcam index 0; keep URLs/paths as strings."""
    if source_text.isdigit():
        return int(source_text)
    return source_text


def resolve_device(requested: str) -> Any:
    if requested.lower() == "auto":
        if torch.cuda.is_available():
            print(f"[INFO] Auto-selected CUDA GPU: {torch.cuda.get_device_name(0)}")
            return 0
        print("[INFO] CUDA is unavailable; auto-selected CPU.")
        return "cpu"
    if requested.lower() == "cpu":
        return "cpu"
    # Ultralytics accepts device indices like 0 and values such as '0,1'.
    return int(requested) if requested.isdigit() else requested


def load_calibration(path_text: str | None, fx_override: float | None, frame_size: tuple[int, int]):
    """Return a size-adjusted camera matrix and distortion coefficients, if available."""
    width, height = frame_size
    if path_text:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with np.load(path) as data:
            camera_matrix = data["camera_matrix"].astype(np.float64)
            dist_coeffs = data["dist_coeffs"].astype(np.float64)
            calibrated_width = int(data["image_width"])
            calibrated_height = int(data["image_height"])
        sx = width / calibrated_width
        sy = height / calibrated_height
        if abs((width / height) - (calibrated_width / calibrated_height)) > 0.02:
            warnings.warn(
                "Active frame aspect ratio differs from calibration. Distance estimates may be less reliable; "
                "calibrate at the same aspect ratio/resolution for best results."
            )
        camera_matrix[0, 0] *= sx  # fx
        camera_matrix[0, 2] *= sx  # cx
        camera_matrix[1, 1] *= sy  # fy
        camera_matrix[1, 2] *= sy  # cy
        print(f"[INFO] Loaded calibration: {path} (calibrated at {calibrated_width}x{calibrated_height})")
        print(f"[INFO] Effective fx={camera_matrix[0, 0]:.2f}px at {width}x{height}")
        return camera_matrix, dist_coeffs

    if fx_override is not None:
        if fx_override <= 0:
            raise ValueError("--fx-px must be greater than zero.")
        print(f"[INFO] Using manually supplied fx={fx_override:.2f}px; distortion correction is disabled.")
        return np.array([[fx_override, 0, width / 2], [0, fx_override, height / 2], [0, 0, 1]], dtype=np.float64), None

    print("[WARN] No camera calibration supplied. Detections will run, but distance will show N/A.")
    return None, None


def calibrate_camera(image_dir_text: str, output_text: str, cols: int, rows: int, square_size_m: float) -> None:
    image_dir = Path(image_dir_text)
    output_path = Path(output_text)
    if cols < 2 or rows < 2:
        raise ValueError("Checkerboard inner-corner rows/columns must each be at least 2.")
    if square_size_m <= 0:
        raise ValueError("--square-size-m must be greater than zero.")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Calibration image directory not found: {image_dir}")

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")

    # Object coordinates are expressed in metres because square_size_m is in metres.
    object_template = np.zeros((rows * cols, 3), dtype=np.float32)
    object_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32) * square_size_m
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    successful = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Cannot read {image_path}; skipping.")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size_now = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size_now
        elif size_now != image_size:
            print(f"[WARN] {image_path.name} has size {size_now}, unlike {image_size}; skipping.")
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"[WARN] Checkerboard not found in {image_path.name}; skipping.")
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        successful += 1
        print(f"[OK] Checkerboard detected: {image_path.name}")

    if successful < 8 or image_size is None:
        raise RuntimeError(
            f"Only {successful} usable checkerboard images. Use at least 8 (preferably 15-25) varied views, then retry."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=np.int32(image_size[0]),
        image_height=np.int32(image_size[1]),
        rms=np.float64(rms),
        board_cols=np.int32(cols),
        board_rows=np.int32(rows),
        square_size_m=np.float64(square_size_m),
    )
    print("\n[CALIBRATION COMPLETE]")
    print(f"Usable images : {successful}/{len(image_paths)}")
    print(f"Image size    : {image_size[0]}x{image_size[1]}")
    print(f"RMS reproj. error: {rms:.5f} (lower is generally better; inspect calibration quality too)")
    print(f"Focal length  : fx={camera_matrix[0, 0]:.2f}px, fy={camera_matrix[1, 1]:.2f}px")
    print(f"Saved to      : {output_path.resolve()}")


def capture_calibration_images(source: Any, directory_text: str, width: int | None, height: int | None) -> None:
    directory = Path(directory_text)
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/source: {source}")
    saved_count = 0
    print("[CALIBRATION CAPTURE] Show a printed checkerboard to the SAME camera you will use for detection.")
    print("Move it to different positions/tilts and distances. Press 's' to save a frame; press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read a frame; exiting capture.")
                break
            preview = frame.copy()
            cv2.putText(preview, f"Saved: {saved_count} | s=save | q=quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Camera Calibration Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                saved_count += 1
                out_path = directory / f"calib_{saved_count:03d}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"[SAVED] {out_path}")
            elif key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved_count} image(s) to {directory.resolve()}")
    if saved_count < 8:
        print("[NOTE] Save at least 8 usable images; 15-25 varied views are recommended.")


def canonical_class_name(raw_name: str, class_id: int) -> str:
    label = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    if "cone" in label:
        return "cone"
    if "stop" in label and ("sign" in label or label == "stop"):
        return "stop_sign"
    # The trained data_2class model uses IDs 0=cone and 1=stop_sign.
    if class_id == 0:
        return "cone"
    if class_id == 1:
        return "stop_sign"
    return label or f"class_{class_id}"


def known_width_m(class_name: str, args: argparse.Namespace) -> float | None:
    if class_name == "cone":
        return args.cone_width_m
    if class_name == "stop_sign":
        return args.stop_sign_width_m
    return None


def estimate_distance_m(known_width: float | None, focal_length_px: float | None, bbox_width_px: float) -> float | None:
    if known_width is None or focal_length_px is None or bbox_width_px <= 1:
        return None
    # Pinhole camera approximation: Z = (real width * focal length in pixels) / apparent bbox width.
    return (known_width * focal_length_px) / bbox_width_px


def annotate_frame(frame: np.ndarray, result: Any, names: dict[int, str], camera_matrix: np.ndarray | None, args: argparse.Namespace) -> np.ndarray:
    output = frame.copy()
    fx = float(camera_matrix[0, 0]) if camera_matrix is not None else None
    boxes = result.boxes
    detection_count = 0
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            raw_name = names.get(class_id, f"class_{class_id}")
            class_name = canonical_class_name(raw_name, class_id)
            bbox_width_px = max(0, x2 - x1)
            distance = estimate_distance_m(known_width_m(class_name, args), fx, bbox_width_px)
            if distance is None:
                distance_text = "Distance: N/A (calibrate camera)"
            else:
                distance_text = f"Distance: {distance:.2f} m (approx.)"

            color = (0, 220, 0) if class_name == "cone" else (0, 165, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(output, label, (x1, max(22, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(output, distance_text, (x1, max(44, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            detection_count += 1

    if camera_matrix is None:
        status = "Distance disabled: add --calibration camera_calibration.npz"
        cv2.putText(output, status, (12, output.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"Detections: {detection_count}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    return output


def run_detection(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\nRun this script from your project root or pass --model with the full path."
        )
    if args.cone_width_m <= 0 or args.stop_sign_width_m <= 0:
        raise ValueError("Object widths must be greater than zero.")
    if not 0 < args.conf < 1:
        raise ValueError("--conf must be between 0 and 1.")

    source = resolve_source(args.source)
    device = resolve_device(args.device)
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {i: name for i, name in enumerate(model.names)}
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Classes: {names}")
    print(f"[INFO] Device: {device}; imgsz={args.imgsz}; confidence={args.conf}")
    print(f"[INFO] Width assumptions: cone={args.cone_width_m:.3f}m, stop sign={args.stop_sign_width_m:.3f}m")
    print("[IMPORTANT] Replace these width assumptions with measured dimensions of the objects you are detecting.")

    source_path = Path(source) if isinstance(source, str) else None
    is_image_file = source_path is not None and source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS
    is_image_dir = source_path is not None and source_path.is_dir()
    is_video_file = source_path is not None and source_path.suffix.lower() in VIDEO_EXTENSIONS
    if source_path is not None and not source_path.exists() and not source_text_is_url(str(source)):
        raise FileNotFoundError(f"Source path not found: {source_path}")
    if source_path is not None and source_path.exists() and source_path.is_file() and not (is_image_file or is_video_file):
        raise ValueError(f"Unsupported input file type: {source_path.suffix}. Use an image, video, folder of images, webcam index, or stream URL.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = None
    writer = None
    inference_times: list[float] = []
    processed_frames = 0
    started = time.perf_counter()
    camera_matrix = None
    dist_coeffs = None
    calibration_checked = False
    use_cuda_timing = device != "cpu" and torch.cuda.is_available()

    if is_image_dir:
        # Batch-process all images in a directory, recursively, without requiring OpenCV GUI support.
        image_paths = sorted(
            p for p in source_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in directory: {source_path}")
        print(f"[INFO] Batch mode: found {len(image_paths)} image(s) in {source_path}")
        print("[INFO] Batch mode saves annotated files without opening display windows (works with headless OpenCV).")
        detection_rows: list[dict[str, Any]] = []
        calibration_cache: dict[tuple[int, int], tuple[np.ndarray | None, np.ndarray | None]] = {}
        for image_index, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[WARN] Could not read image; skipping: {image_path}")
                continue
            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size not in calibration_cache:
                calibration_cache[frame_size] = load_calibration(
                    args.calibration, args.fx_px, frame_size
                )
            camera_matrix, dist_coeffs = calibration_cache[frame_size]
            if camera_matrix is not None and dist_coeffs is not None:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
            if use_cuda_timing:
                torch.cuda.synchronize()
            tick = time.perf_counter()
            result = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False
            )[0]
            if use_cuda_timing:
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - tick)
            output = annotate_frame(frame, result, names, camera_matrix, args)
            relative_path = image_path.relative_to(source_path)
            output_path = output_dir / relative_path.parent / f"annotated_{relative_path.stem}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), output):
                print(f"[WARN] Could not save output image: {output_path}")
                continue
            processed_frames += 1
            box_list = result.boxes
            num_detections = len(box_list) if box_list is not None else 0
            print(f"[{image_index}/{len(image_paths)}] {relative_path} -> {num_detections} detection(s); saved {output_path}")
            # Record each detection for a machine-readable results CSV.
            if box_list is not None:
                fx = float(camera_matrix[0, 0]) if camera_matrix is not None else None
                for box in box_list:
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = xyxy
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = canonical_class_name(names.get(class_id, f"class_{class_id}"), class_id)
                    distance = estimate_distance_m(
                        known_width_m(class_name, args), fx, max(0, x2 - x1)
                    )
                    detection_rows.append({
                        "image": str(relative_path),
                        "class": class_name,
                        "confidence": round(confidence, 6),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "bbox_width_px": max(0, x2 - x1),
                        "estimated_distance_m": round(distance, 4) if distance is not None else "",
                        "distance_status": "approximate" if distance is not None else "not_calibrated",
                    })
        if processed_frames == 0:
            raise RuntimeError(f"No images could be processed from {source_path}")
        csv_path = output_dir / "detections.csv"
        import csv
        fieldnames = ["image", "class", "confidence", "x1", "y1", "x2", "y2", "bbox_width_px", "estimated_distance_m", "distance_status"]
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(detection_rows)
        print(f"[SAVED] Detection table: {csv_path.resolve()}")
    elif is_image_file:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source_path}")
        camera_matrix, dist_coeffs = load_calibration(args.calibration, args.fx_px, (frame.shape[1], frame.shape[0]))
        if camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
        if use_cuda_timing:
            torch.cuda.synchronize()
        tick = time.perf_counter()
        result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        if use_cuda_timing:
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - tick)
        output = annotate_frame(frame, result, names, camera_matrix, args)
        output_path = output_dir / f"annotated_{source_path.stem}.jpg"
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"Could not save output image: {output_path}")
        processed_frames = 1
        print(f"[SAVED] {output_path.resolve()}")
        if not args.no_display:
            cv2.imshow("Robot Navigation - Detection and Distance", output)
            print("Press any key in the image window to close it.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(source)
        if args.camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps_in) or fps_in < 1 or fps_in > 240:
            fps_in = 25.0
        writer_path = output_dir / (f"annotated_{source_path.stem}.mp4" if is_video_file else "annotated_webcam.mp4")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if not calibration_checked:
                    camera_matrix, dist_coeffs = load_calibration(args.calibration, args.fx_px, (frame.shape[1], frame.shape[0]))
                    calibration_checked = True
                if camera_matrix is not None and dist_coeffs is not None:
                    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

                if use_cuda_timing:
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
                if use_cuda_timing:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                inference_times.append(elapsed)
                output = annotate_frame(frame, result, names, camera_matrix, args)
                processed_frames += 1

                if args.save_video:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(writer_path), fourcc, fps_in, (output.shape[1], output.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create output video: {writer_path}")
                    writer.write(output)

                if not args.no_display:
                    cv2.imshow("Robot Navigation - Detection and Distance", output)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()
        if args.save_video and processed_frames:
            print(f"[SAVED] {writer_path.resolve()}")

    total_time = time.perf_counter() - started
    # For streams, exclude initial warm-up frames from the inference latency summary where possible.
    measured = inference_times[args.warmup_frames:] if len(inference_times) > args.warmup_frames else inference_times
    if measured:
        mean_ms = float(np.mean(measured) * 1000)
        median_ms = float(np.median(measured) * 1000)
        print("\n[PERFORMANCE SUMMARY]")
        print(f"Frames processed          : {processed_frames}")
        print(f"Mean inference latency    : {mean_ms:.2f} ms/frame")
        print(f"Median inference latency  : {median_ms:.2f} ms/frame")
        print(f"Inference-only throughput : {1000 / mean_ms:.2f} FPS (approx.)")
        print(f"End-to-end processing rate: {processed_frames / total_time:.2f} FPS (includes capture/display/output)")
        print(f"Warm-up frames excluded  : {max(0, len(inference_times) - len(measured))}")
    if torch.cuda.is_available() and device != "cpu":
        print(f"CUDA GPU                  : {torch.cuda.get_device_name(0)}")
    else:
        print("Compute device             : CPU (or CUDA not used)")


def source_text_is_url(source_text: str) -> bool:
    return source_text.startswith(("rtsp://", "http://", "https://", "rtmp://"))


def main() -> None:
    args = parse_args()
    if args.capture_calibration:
        capture_calibration_images(
            resolve_source(args.source), args.calibration_dir, args.camera_width, args.camera_height
        )
        return
    if args.calibrate_dir:
        calibrate_camera(
            args.calibrate_dir,
            args.calibration_out,
            args.board_cols,
            args.board_rows,
            args.square_size_m,
        )
        return
    run_detection(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
