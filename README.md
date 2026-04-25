# Drone Project - DroneTastic RHUL

A drone control and monitoring system for hackathon.

## Project Structure

- `src/core/communication.py`: UDP telemetry packet format, sender, and receiver
- `src/stages/connect.py`: connection stage to ground station
- `src/stages/aruco_detector.py`: ArUco/AprilTag detector (PC side)
- `src/stages/yolo_detector.py`: YOLO detector (PC side)
- `src/runtime/pipeline.py`: threaded PC runtime (ArUco + YOLO in parallel)
- `src/runtime/aruco_demo.py`: ArUco-only demo runtime (no YOLO)
- `scripts/download_yolo_onnx.py`: download YOLO ONNX model on PC
- `scripts/pi_send_telemetry.py`: telemetry sender intended for Raspberry Pi
- `scripts/pc_receive_telemetry.py`: telemetry receiver intended for PC ground station

## Getting Started

Run these steps from the `Drone` project folder.

### 1) Install dependencies (PC and Pi)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Find your PC IP address

On Linux/macOS:

```bash
hostname -I
```

Use the IPv4 address from the same network as your Raspberry Pi (example: `192.168.1.42`).

### 3) Start telemetry receiver on your PC

```bash
python scripts/pc_receive_telemetry.py --port 9000
```

Keep this terminal open. It should print incoming telemetry.

### 4) Start telemetry sender on Raspberry Pi

On the Pi, open another terminal in the same project and run:

```bash
python scripts/pi_send_telemetry.py --pc-ip <PC_IP> --port 9000 --rate-hz 5
```

Example:

```bash
python scripts/pi_send_telemetry.py --pc-ip 192.168.1.42 --port 9000 --rate-hz 5
```

### 5) Confirm it works

On the PC receiver terminal you should see lines like:

```text
seq=0 source=drone-pi ts=... payload={...}
```

If no packets appear:

- Confirm both devices are on the same Wi-Fi/network.
- Confirm the `--pc-ip` value is correct.
- Allow inbound UDP port `9000` on the PC firewall.
- Ensure both sides use the same `--port` value.

## Connect-Then-Detect Pipeline

Architecture you asked for:

- Raspberry Pi: camera capture + frame streaming
- PC: receives frames + runs AI detection
- ArUco and YOLO run in separate detector files and separate threads

For the Pi camera, prefer the RTSP feed exposed by go2rtc:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop \
	-analyzeduration 0 -probesize 32 -vf setpts=0 \
	rtsp://dronetastic.local:8554/cam1
```

One-line versions:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -analyzeduration 0 -probesize 32 -vf setpts=0 rtsp://dronetastic.local:8554/cam1
```

```bash
python -m src.runtime.pipeline --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family tag36h11 --yolo-model yolov8s.onnx --yolo-classes coco.names --show
```

```bash
python -m src.runtime.aruco_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family 6x6_250
```

Use this flow:

Verified working on the laptop on 25 April 2026 after removing the vision package's optional demo imports that required `open3d`.

The pipeline also saves ArUco crops automatically to `artifacts/aruco_crops/`.

If you want printable ArUco/AprilTag symbols, generate them with:

```bash
python scripts/generate_aruco_markers.py --family tag36h11 --count 10 --size 800 --output-dir artifacts/aruco_markers
```

The PNG files will be written under `artifacts/aruco_markers/`.

Important: the marker family must match the pipeline `--family` value.
Example for standard ArUco markers:

```bash
python scripts/generate_aruco_markers.py --family 6x6_250 --count 10 --size 800 --output-dir artifacts/aruco_markers
python -m src.runtime.pipeline --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --family 6x6_250 --yolo-model yolov8s.onnx --yolo-classes coco.names --show
```

1. Start AI pipeline on PC (receiver + detector):

Download model once on PC (from project root):

```bash
python scripts/download_yolo_onnx.py --model yolov8s --output yolov8s.onnx
```

2. If you want the older UDP pipeline instead of RTSP, start the camera streamer on the Raspberry Pi:

```bash
python -m src.video_transmitter --ip <PC_IP> --port 5600 --width 640 --height 480 --fps 20 --quality 80
```

This runs in order:

- `src/runtime/pipeline.py` on PC receives the Pi RTSP stream and schedules AI workers.
- `src/stages/aruco_detector.py` runs ArUco detection thread.
- `src/stages/yolo_detector.py` runs YOLO detection thread.

If you still want the older UDP path, keep `src/video_transmitter.py` on the Pi and run the pipeline without `--rtsp-url`.

YOLO model note:

- Place your ONNX model file in the project root (example: `yolov8s.onnx`).
- If model/classes files are missing, pipeline still runs ArUco and prints a YOLO-disabled message.

## Features

- Low-latency UDP telemetry from Pi to PC
- JSON packet format with sequence numbers and timestamps

## Team

## License
