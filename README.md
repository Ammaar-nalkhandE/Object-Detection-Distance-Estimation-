# Object Detection and Distance Estimation for Robotics Navigation

This project detects **cones** and **stop signs** with a fine-tuned YOLO11n model, saves annotated images and CSV detections, and supports CPU/GPU inference benchmarking. It can optionally estimate **relative depth** with MiDaS or approximate metric distance if valid camera parameters and real-world object dimensions are available.

> **Safety:** monocular distance estimates are approximate. Do not use them as the sole input for braking or collision avoidance. A real robot should use validated sensing such as calibrated stereo/depth cameras or LiDAR, plus controlled safety tests.

## 1. Project directory and paths

The project directory on your Ubuntu machine is:

```bash
cd "$HOME/Documents/zzz HACKS&PROJS/eric_assisment/robot-nav-distance"
```

The quotes are important because the directory name contains `&`.

The commands below assume these paths exist in that directory:

```text
robot-nav-distance/
├── robot_nav_distance_2.py
├── test_imgs/                         # Your own image samples
├── data_2class/
│   ├── dataset.yaml
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
├── runs/
│   └── detect/
│       └── robot-nav-baseline2/
│           └── weights/
│               └── best.pt             # Trained YOLO11n weights
└── runs/benchmark/                     # Created by benchmark commands
```

Your trained model is loaded from:

```text
runs/detect/robot-nav-baseline2/weights/best.pt
```

Run the commands from the project root so the relative model and dataset paths resolve correctly. If your script has a different filename, replace `robot_nav_distance_2.py` in the commands with that filename.

## 2. Environment check

Use the same Python environment you used to train the model. Check dependencies and CUDA:

```bash
python -c "import torch, cv2, numpy; from ultralytics import YOLO; print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If the imports show that Ultralytics or OpenCV is missing, install those packages in the active environment:

```bash
python -m pip install ultralytics opencv-python numpy
```

Avoid reinstalling PyTorch if your current CUDA-enabled installation is already working.

## 3. Run detection on your `test_imgs` folder

For detection-only mode, skip the MiDaS relative-depth model:

```bash
cd "$HOME/Documents/zzz HACKS&PROJS/eric_assisment/robot-nav-distance"

python robot_nav_distance_2.py \
  --source test_imgs \
  --device 0 \
  --distance-mode none \
  --output-dir runs/test_images/gpu
```

Use `--device cpu` instead of `--device 0` to run on the CPU. The output folder contains annotated images and a CSV, as configured by the script.

### Relative-depth mode (optional)

To use MiDaS for relative-depth scores, run:

```bash
python -m pip install timm
python robot_nav_distance_2.py \
  --source test_imgs \
  --device 0 \
  --distance-mode relative \
  --output-dir runs/test_images/relative_depth
```

The first successful run may download MiDaS model files and therefore needs internet access. Relative-depth scores are **not metres**; normalized scores from different photographs should not be compared as if they share a common scale.

### Metric-distance mode (only after calibration)

Approximate metric distance requires a camera calibration file or a known focal length in pixels, and measured object widths. Do not run metric mode with guessed camera parameters and treat the result as ground truth. Once you have a valid calibration file, use the options supported by your script, for example:

```bash
python robot_nav_distance_2.py \
  --source test_imgs \
  --device 0 \
  --distance-mode metric \
  --calibration camera_calibration.npz \
  --output-dir runs/test_images/metric
```

The basic pinhole-camera approximation is:

\[
D = \frac{W f_{px}}{w_{px}}
\]

where `D` is estimated distance in metres, `W` is measured real-world object width in metres, `f_px` is horizontal focal length in pixels, and `w_px` is the object's detected bounding-box width in pixels. Cone sizes, sign sizes, object orientation, occlusion, and bounding-box errors affect accuracy. The sample widths in a script are only defaults; measure representative objects in your environment.

## 4. Evaluate the trained model on the held-out test split

To reproduce the detector evaluation with Ultralytics:

```bash
yolo detect val \
  model=runs/detect/robot-nav-baseline2/weights/best.pt \
  data=data_2class/dataset.yaml \
  split=test \
  imgsz=640 \
  batch=8 \
  device=0
```

Previously recorded test results for this checkpoint were:

| Metric | Test result |
|---|---:|
| Images | 197 |
| Ground-truth objects | 324 |
| Precision | 0.946 |
| Recall | 0.930 |
| mAP@0.5 | 0.963 |
| mAP@0.5:0.95 | 0.766 |
| Mean inference time reported by validation | about 7.0 ms/image |

Per-class results from that run:

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| cone | 282 | 0.900 | 0.859 | 0.931 | 0.595 |
| stop_sign | 42 | 0.992 | 1.000 | 0.995 | 0.937 |

Results can change if the dataset, checkpoint, package versions, image size, or command-line arguments change. Test with real camera footage before drawing conclusions about deployment.

## 5. Benchmark CPU versus GPU

This benchmark uses the same held-out images, model, resolution, confidence threshold, and detection-only mode for both devices. This isolates YOLO inference and avoids measuring MiDaS or its download.

### GPU (NVIDIA RTX 2050)

```bash
cd "$HOME/Documents/zzz HACKS&PROJS/eric_assisment/robot-nav-distance"
mkdir -p runs/benchmark

python robot_nav_distance_2.py \
  --source data_2class/images/test \
  --device 0 \
  --distance-mode none \
  --imgsz 640 \
  --conf 0.25 \
  --warmup-frames 5 \
  --output-dir runs/benchmark/gpu_outputs \
  2>&1 | tee runs/benchmark/gpu.log
```

### CPU

```bash
python robot_nav_distance_2.py \
  --source data_2class/images/test \
  --device cpu \
  --distance-mode none \
  --imgsz 640 \
  --conf 0.25 \
  --warmup-frames 5 \
  --output-dir runs/benchmark/cpu_outputs \
  2>&1 | tee runs/benchmark/cpu.log
```

### Print the comparison

```bash
grep -E \
'Images processed|Mean YOLO inference|Median YOLO inference|YOLO throughput|Total elapsed time' \
runs/benchmark/gpu.log \
runs/benchmark/cpu.log
```

### Measured benchmark results

These are the values from the benchmark run you reported:

| Metric | GPU (device 0) | CPU |
|---|---:|---:|
| Images processed | 197 | 197 |
| Detections | 367 | 367 |
| Mean YOLO inference latency | 8.91 ms/image | 48.56 ms/image |
| Median YOLO inference latency | 8.78 ms/image | 46.92 ms/image |
| Approx. YOLO throughput | 112.18 FPS | 20.59 FPS |
| Total elapsed time | 2.79 s | 10.35 s |

From this run, the GPU's mean YOLO inference latency was approximately **5.45× lower** than the CPU's (about **81.7% lower latency**). Its reported YOLO throughput was approximately **5.45× higher**. Total elapsed-time speedup was about **3.71×**. These are results for this specific setup, not a guarantee for other machines or full robot deployments.

The FPS reported by the script is based on inference latency and should not be interpreted as end-to-end camera-to-actuator frame rate. Full deployment time can also include image capture, preprocessing, annotation, networking, planning, and robot control.

## 6. Current class coverage and limitations

The current model has two classes:

```text
0: cone
1: stop_sign
```

**Barrier detection is not currently supported by this checkpoint** because the training dataset used for this model did not contain labelled barrier examples. To add barriers, add representative, correctly labelled barrier images to the training dataset, update the YAML class mapping, and retrain/evaluate the model.

Relative depth from MiDaS is useful as a qualitative cue, but it is not a calibrated distance measurement. Monocular metric distance should be validated against measured reference distances across different sizes, viewpoints, lighting conditions, and ranges. For real robotics navigation, fuse vision with a suitable depth sensor or LiDAR and perform controlled safety testing.

## 7. Suggested report table

Use measured values only; leave entries blank until you run the relevant experiment.

| Variant | Runtime/device | Image size | Model size | FPS | Mean latency (ms) | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11n `.pt` | CPU | 640 | — | 20.59* | 48.56* | 0.766** |
| YOLO11n `.pt` | NVIDIA RTX 2050 | 640 | — | 112.18* | 8.91* | 0.766** |
| Exported model (if tested) | Target device | 640 | — | — | — | — |

`*` Inference benchmark values from the reported CPU/GPU run. `**` mAP is from the separate held-out test evaluation, not from the timing benchmark itself.

## 8. Future optimization

After establishing this baseline, you can evaluate ONNX or another supported deployment format, then compare model size, latency, FPS, and detection accuracy on the target device. Do not assume export or quantization improves speed or preserves accuracy—measure both. Any deployment optimization should be evaluated against the same held-out test set.
