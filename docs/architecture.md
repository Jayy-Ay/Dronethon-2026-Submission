# System Architecture

## Overview

This project uses a PC-side threaded vision runtime that ingests frames from the Raspberry Pi, runs ArUco and YOLO in parallel, and publishes the latest fused view for preview and downstream logic.

PlantUML sources for the architecture are available in `docs/plantuml/`.

## Components

- Frame ingestion: `RtspFrameProvider` and `StreamFrameProvider` in `src/vision/frame_provider.py`
- Detection stages: `ArucoDetector` and `YoloDetector`
- Runtime threading: `DetectorWorker`, `FrameCache`, and `DisplayWorker` in `src/runtime/pipeline.py`
- Positioning: `object_geolocator.py` for ray-to-ground intersection and object NED estimates

## PlantUML Diagrams

- `docs/plantuml/01_system_overview.puml`: high-level component/data-flow view
- `docs/plantuml/02_threaded_pipeline_sequence.puml`: threaded frame processing sequence
- `docs/plantuml/03_yolo_detection_flow.puml`: detection/segmentation inference flow
- `docs/plantuml/04_object_geolocation_flow.puml`: detection-to-ground-position flow

## Data Flow

1. Frame provider receives RTSP or UDP camera frames.
2. The main loop submits each frame to ArUco and YOLO worker threads.
3. Workers publish freshest results to `FrameCache` (dropping stale pending frames to reduce latency).
4. Display thread renders the latest available combined overlays.
5. Optional geolocation converts `YoloDetection` reference pixels to ground-plane coordinates using telemetry pose.
