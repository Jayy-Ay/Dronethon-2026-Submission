# Runtime Demos

Short startup commands for the main runtime demos in `src/runtime/`.

## Table of Contents

- [Quick Start](#quick-start)
- [Start YOLO Only](#start-yolo-only)
- [Start YOLO Segmentation](#start-yolo-segmentation)
- [Start MAVSDK Object Position Demo](#start-mavsdk-object-position-demo)

## Quick Start

If you only want the main startup commands, use one of these:

YOLO only:

```bash
python -m src.runtime.yolo_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --yolo-model yolov8s.pt --yolo-classes coco.names --yolo-input 640 --show
```

YOLO segmentation:

```bash
python -m src.runtime.yolo_segmentation_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --yolo-model yolov8n-seg.pt --yolo-classes coco.names --yolo-input 640 --show
```

MAVSDK object position demo:

```bash
python -m src.runtime.object_position_demo_mavsdk --camera-source rtsp://dronetastic.local:8554/cam1 --yolo-model yolov8s.pt --yolo-classes coco.names --target-label person --marker-length-m 0.267 --area-width-m 0.60 --area-height-m 0.60 --show
```

## Start YOLO Only

Use this when you want object detection without segmentation or MAVSDK telemetry.

### Recommended command

```bash
python -m src.runtime.yolo_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --yolo-model yolov8s.pt --yolo-classes coco.names --yolo-input 640 --show
```

## Start YOLO Segmentation

Use this when you want segmentation masks instead of bounding boxes only.

### Recommended command

```bash
python -m src.runtime.yolo_segmentation_demo --rtsp-url rtsp://dronetastic.local:8554/cam1 --rtsp-width 1280 --rtsp-height 720 --yolo-model yolov8n-seg.pt --yolo-classes coco.names --yolo-input 640 --show
```

## Start MAVSDK Object Position Demo

Use this when you want YOLO detections projected onto the floor using ArUco markers, with Pixhawk pose as a fallback.

### Recommended command

```bash
python -m src.runtime.object_position_demo_mavsdk --camera-source rtsp://dronetastic.local:8554/cam1 --yolo-model yolov8s.pt --yolo-classes coco.names --target-label person --marker-length-m 0.267 --area-width-m 0.60 --area-height-m 0.60 --show
```
