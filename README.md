# DroneTastic RHUL

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-PC%20%2B%20Raspberry%20Pi-green)
![Vision](https://img.shields.io/badge/Vision-ArUco%20%2F%20AprilTag-orange)
![Detection](https://img.shields.io/badge/Detection-YOLOv8-yellow)
![Stream](https://img.shields.io/badge/Video-RTSP%20720p%4060-red)

A PC-side drone vision and telemetry project for the DroneTastic RHUL hackathon setup. The Raspberry Pi provides the camera stream, and the laptop runs ArUco detection or ArUco + YOLOv8 detection.

## Table of Contents

- [What This Repo Does](#what-this-repo-does)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Start ArUco Only](#start-aruco-only)
- [Start ArUco Localisation](#start-aruco-localisation)
- [Start ArUco--YOLOv8](#start-arucoyolov8)
- [Optional Telemetry](#optional-telemetry)
- [Generate Markers](#generate-markers)
- [Troubleshooting](#troubleshooting)

## What This Repo Does

- Receives the Raspberry Pi camera feed on a PC
- Runs ArUco or AprilTag detection
- Optionally runs YOLOv8 in parallel with ArUco
- Displays an annotated local preview
- Saves ArUco crops to `artifacts/aruco_crops/`
- Includes simple Pi-to-PC UDP telemetry scripts

## Project Layout

- `src/runtime/aruco_demo.py`: ArUco-only runtime
- `src/runtime/localization_demo.py`: ArUco world-frame localisation runtime
- `src/runtime/pipeline.py`: ArUco + YOLOv8 runtime
- `src/stages/aruco_detector.py`: ArUco / AprilTag detector
- `src/localization/aruco_localizer.py`: multi-marker world pose estimation
- `src/stages/yolo_detector.py`: YOLO detector
- `src/vision/frame_provider.py`: RTSP and UDP frame input
- `scripts/download_yolo_onnx.py`: download or export a YOLO ONNX model
- `scripts/generate_aruco_markers.py`: generate printable markers
- `scripts/pc_receive_telemetry.py`: PC telemetry receiver
- `scripts/pi_send_telemetry.py`: Raspberry Pi telemetry sender

## Prerequisites

Before you start, make sure:

- You are running commands from the `Drone` project folder
- Your PC and Raspberry Pi are on the same network
- The Pi camera feed is available at `rtsp://dronetastic.local:8554/cam1`
- Python `3.10+` is installed

For the lowest-latency video preview, this tested `ffplay` command should work from your PC:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -analyzeduration 0 -probesize 32 -vf setpts=0 rtsp://dronetastic.local:8554/cam1
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use YOLOv8, download the ONNX model once on your PC:

```bash
python scripts/download_yolo_onnx.py --model yolov8s --output yolov8s.onnx
```

## Quick Start

If you only want the main startup commands, use one of these:

ArUco only:

```bash
python -m src.runtime.aruco_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family 6x6_250 --show
```

ArUco + YOLOv8:

```bash
python -m src.runtime.pipeline --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family tag36h11 --yolo-model yolov8s.onnx --yolo-classes coco.names --yolo-input 640 --show
```

ArUco localisation:

```bash
python -m src.runtime.localization_demo --marker-length-m 0.10 --area-width-m 2.00 --area-height-m 1.50 --camera-fx 920 --camera-fy 920 --camera-cx 640 --camera-cy 360 --show
```

## Start ArUco Only

Use this when you want marker detection without YOLO.

### Recommended command

```bash
python -m src.runtime.aruco_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family 6x6_250 --show
```

### What it does

- Connects to the RTSP stream from the Pi
- Runs ArUco detection only
- Opens a preview window with marker overlays

### Notes

- Default ArUco family in `aruco_demo` is `6x6_250`
- Press `q` or `Esc` to close the preview
- If you are using a different marker family, change `--family` to match it

## Start ArUco Localisation

Use this when you want the camera or drone position in a shared floor coordinate system defined by the fixed ArUco markers.

### World frame

- Marker `0` is the world origin `(0, 0, 0)`
- `+X` points from marker `0` toward marker `1`
- `+Y` points from marker `0` toward marker `2`
- `+Z` points upward from the floor

### Before you run it

Measure these values from your real setup:

- `--marker-length-m`: physical edge length of one printed ArUco marker
- `--area-width-m`: center-to-center distance from marker `0` to marker `1`
- `--area-height-m`: center-to-center distance from marker `0` to marker `2`
- `--camera-fx`, `--camera-fy`, `--camera-cx`, `--camera-cy`: calibrated camera intrinsics

### Recommended RTSP command

```bash
python -m src.runtime.localization_demo --marker-length-m 0.10 --area-width-m 2.00 --area-height-m 1.50 --camera-fx 920 --camera-fy 920 --camera-cx 640 --camera-cy 360 --show
```

### UDP command

```bash
python -m src.runtime.localization_demo --rtsp-url "" --bind-ip 0.0.0.0 --video-port 5600 --marker-length-m 0.10 --area-width-m 2.00 --area-height-m 1.50 --camera-fx 920 --camera-fy 920 --camera-cx 640 --camera-cy 360 --show
```

### What it does

- Detects visible ArUco markers in each frame
- Matches marker IDs against the known floor map
- Solves for one camera pose in world coordinates using one or more markers
- Prints world-frame camera position in meters
- Shows reprojection error as a simple quality signal

### Notes

- Replace the example dimensions and intrinsics with your real measurements
- If only one marker is visible, pose will usually be noisier than with two or more markers
- This demo currently assumes zero distortion coefficients, so it is best paired with a later calibration-file upgrade
- Press `q` or `Esc` to close the preview

## Start ArUco + YOLOv8

Use this when you want both marker detection and object detection at the same time.

### Before you run it

Make sure these files exist in the project root:

- `yolov8s.onnx`
- `coco.names`

If the model is missing, download it with:

```bash
python scripts/download_yolo_onnx.py --model yolov8s --output yolov8s.onnx
```

### Recommended command

```bash
python -m src.runtime.pipeline --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family tag36h11 --yolo-model yolov8s.onnx --yolo-classes coco.names --yolo-input 640 --show
```

### What it does

- Connects to the RTSP stream from the Pi
- Runs ArUco / AprilTag detection in one worker thread
- Runs YOLOv8 detection in another worker thread
- Combines both outputs into a single preview

### Notes

- Default family in `pipeline` is `tag36h11`
- The pipeline saves ArUco crops to `artifacts/aruco_crops/`
- If YOLO files are missing, the pipeline can still run with ArUco only, but YOLO will be disabled

## Optional Telemetry

The repo also includes simple UDP telemetry scripts.

Start the receiver on your PC:

```bash
python scripts/pc_receive_telemetry.py --port 9000
```

On the Raspberry Pi, send telemetry to your PC:

```bash
python scripts/pi_send_telemetry.py --pc-ip <PC_IP> --port 9000 --rate-hz 5
```

Find your PC IP address on Linux or macOS with:

```bash
hostname -I
```

Expected receiver output looks like:

```text
seq=0 source=drone-pi ts=... payload={...}
```

## Generate Markers

Generate printable markers like this:

For AprilTag `tag36h11`:

```bash
python scripts/generate_aruco_markers.py --family tag36h11 --count 10 --size 267 --output-dir artifacts/aruco_markers
```

For standard ArUco `6x6_250`:

```bash
python scripts/generate_aruco_markers.py --family 6x6_250 --count 10 --size 267 --output-dir artifacts/aruco_markers
```

Important:

- The marker family must match the `--family` value used by the runtime
- Generated images are written to `artifacts/aruco_markers/`

## Troubleshooting

- No video stream: confirm the Pi feed works in `ffplay` first
- No telemetry: confirm both devices are on the same network and your PC firewall allows UDP on port `9000`
- ArUco not detecting markers: make sure the printed marker family matches the runtime `--family`
- YOLO not starting: confirm `yolov8s.onnx` and `coco.names` are present in the project root
- High video delay: use the RTSP path shown above rather than connecting directly to the camera

Tested on the laptop setup on 25 April 2026 with the current RTSP pipeline and runtime entry points in this repo.
