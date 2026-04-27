# Vision Detection and Segmentation

This document explains how image detection and image segmentation work in this repo, which modules are involved, and how to run the relevant demos.

## Overview

The project has one shared YOLO inference path for both:

- object detection
- instance segmentation

The key idea is:

1. A frame source provides images from RTSP or the custom UDP stream.
2. `YoloDetector` runs inference on each frame.
3. Each result is converted into a shared `YoloDetection` object.
4. If the loaded model supports segmentation, each `YoloDetection` may also include a `mask`.

Detection and segmentation are therefore the same pipeline at the code level. The practical difference is the model you load and whether masks are returned.

## Main Files

- [frame_provider.py](/home/ja/Workspace/Drone/src/vision/frame_provider.py): receives frames from RTSP or UDP
- [yolo_detector.py](/home/ja/Workspace/Drone/src/stages/yolo_detector.py): shared YOLO inference code for detection and segmentation
- [yolo_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_demo.py): standalone object-detection demo
- [yolo_segmentation_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_segmentation_demo.py): standalone segmentation demo
- [pipeline.py](/home/ja/Workspace/Drone/src/runtime/pipeline.py): threaded ArUco + YOLO runtime
- [object_geolocator.py](/home/ja/Workspace/Drone/src/vision/object_geolocator.py): converts detections into ground-plane estimates
- [object_position_demo_mavsdk.py](/home/ja/Workspace/Drone/src/runtime/object_position_demo_mavsdk.py): combines YOLO detections with MAVSDK telemetry

## Frame Input

Frames enter the vision stack through one of these providers in [frame_provider.py](/home/ja/Workspace/Drone/src/vision/frame_provider.py):

- `RtspFrameProvider`
- `StreamFrameProvider`

`RtspFrameProvider` starts `ffmpeg`, decodes the RTSP stream into raw BGR frames, and publishes the newest frame to the runtime.

`StreamFrameProvider` listens for chunked UDP JPEG packets, reassembles them into one image, decodes them with OpenCV, and publishes the completed frame.

Both providers expose:

- `get_frame()`: return the latest frame without waiting
- `get_frame_with_timeout()`: wait for a fresh frame and return `None` on timeout

This means the rest of the vision code can stay independent of whether the camera feed came from RTSP or UDP.

## Shared Detection Data Model

YOLO results are normalized into the `YoloDetection` dataclass in [yolo_detector.py](/home/ja/Workspace/Drone/src/stages/yolo_detector.py).

Each detection contains:

- `class_id`
- `label`
- `confidence`
- `x1`, `y1`, `x2`, `y2`
- `mask`

`mask` is optional:

- `None` for box-only detections
- a boolean image mask for segmentation-capable models

That shared shape is what allows the demos and later geolocation code to work with either plain detections or segmented objects.

## How Object Detection Works

### 1. Model loading

`YoloDetector` supports two backends:

- Ultralytics for `.pt` models
- OpenCV DNN for `.onnx` models

Backend choice is automatic:

- `.pt` -> `ultralytics`
- anything else, typically `.onnx` -> `opencv-dnn`

For `.pt` models, the code tries to import `torch` and `ultralytics`, then prefers CUDA if available.

For `.onnx` models, the code uses `cv2.dnn.readNetFromONNX()` and prefers CUDA through OpenCV DNN when OpenCV was built with CUDA support. Otherwise it falls back to CPU.

### 2. Preprocessing

For the OpenCV DNN path, frames are:

1. resized with letterboxing into a square input
2. padded with value `114`
3. converted into a normalized blob with `cv2.dnn.blobFromImage()`

The detector also stores the resize scale and padding so predicted boxes can be mapped back to the original image coordinates.

For the Ultralytics `.pt` path, preprocessing is handled inside the Ultralytics prediction call.

### 3. Inference

`detect(frame)` is the main entrypoint.

- `.onnx` models run through OpenCV DNN forward pass
- `.pt` models run through `YOLO(...).predict(...)`

### 4. Postprocessing

For box detection, the detector:

1. reads box coordinates and class scores
2. handles both YOLOv5-style and YOLOv8-style output scoring
3. filters by confidence threshold
4. projects boxes back to the original frame
5. applies non-maximum suppression with `cv2.dnn.NMSBoxes`
6. returns a list of `YoloDetection`

This is the full object-detection flow used by [yolo_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_demo.py) and also by the combined [pipeline.py](/home/ja/Workspace/Drone/src/runtime/pipeline.py).

## How Image Segmentation Works

Segmentation uses the same detector class and the same `detect(frame)` call.

The difference is that the loaded model must produce masks, for example:

- `yolov8n-seg.pt`
- `yolov8s-seg.pt`

### Mask extraction

When using the Ultralytics backend, `YoloDetector` checks `result.masks` after inference.

If masks are present, `_extract_result_masks(...)`:

1. reads the mask tensor from the model result
2. resizes each mask to the original frame size
3. thresholds it at `> 0.5`
4. stores it as a boolean array in `YoloDetection.mask`

If masks are missing, the same detection is still returned, but with `mask=None`.

### Rendering

Segmentation visualization is handled by `draw_yolo_overlays(...)` in [yolo_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_demo.py).

When a detection has a mask:

- the masked pixels are tinted with a class color
- the mask contour is outlined
- the box and label are still drawn

So the segmentation demo is really a visualization and logging layer on top of the same detector output type.

### Important limitation

The current mask extraction path is implemented for Ultralytics `.pt` segmentation results.

OpenCV DNN `.onnx` detection models are supported for boxes, but this repo does not currently implement an ONNX segmentation-mask decode path. In practice that means:

- `.onnx` models work well for detection
- `.pt` segmentation models are the intended path for masks

The segmentation demo already warns you when detections appear but no masks are returned.

## Runtime Entry Points

### YOLO detection demo

[yolo_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_demo.py) is the simplest entrypoint for object detection.

It:

- opens the RTSP or UDP stream
- runs `detector.detect(frame)`
- draws boxes and labels
- prints a short summary once per second

Example:

```bash
python -m src.runtime.yolo_demo \
  --rtsp-url rtsp://dronetastic.local:8554/cam1 \
  --rtsp-width 1280 \
  --rtsp-height 720 \
  --yolo-model yolov8s.onnx \
  --yolo-classes coco.names \
  --yolo-input 640 \
  --show
```

### YOLO segmentation demo

[yolo_segmentation_demo.py](/home/ja/Workspace/Drone/src/runtime/yolo_segmentation_demo.py) uses the same detector, but expects a segmentation-capable model.

It:

- opens the stream
- runs `detector.detect(frame)`
- checks whether returned detections contain masks
- draws masks, contours, boxes, and labels
- logs `segments=<mask_count>/<detection_count>`

Example:

```bash
python -m src.runtime.yolo_segmentation_demo \
  --rtsp-url rtsp://dronetastic.local:8554/cam1 \
  --rtsp-width 1280 \
  --rtsp-height 720 \
  --yolo-model yolov8n-seg.pt \
  --yolo-classes coco.names \
  --yolo-input 640 \
  --show
```

### Combined ArUco + YOLO pipeline

[pipeline.py](/home/ja/Workspace/Drone/src/runtime/pipeline.py) runs ArUco and YOLO in separate threads.

The flow is:

1. a frame is pulled from the provider
2. the frame is submitted to the ArUco worker
3. the same frame is submitted to the YOLO worker
4. each worker keeps only the latest pending frame
5. results are published into `FrameCache`
6. the display thread renders the freshest combined view

This design keeps detection work off the main ingest loop and avoids large backlogs when inference is slower than the incoming stream.

One detail to know: the combined overlay currently draws YOLO bounding boxes and labels, but it does not render segmentation masks in `pipeline.py`. Mask rendering is implemented in `yolo_demo.py` and reused by `yolo_segmentation_demo.py`.

## Detection vs Segmentation Summary

| Feature | Detection | Segmentation |
|---|---|---|
| Shared detector class | Yes | Yes |
| Shared output dataclass | Yes | Yes |
| Requires boxes | Yes | Yes |
| Requires masks | No | Yes |
| Best-supported model type | `.onnx` or `.pt` | `.pt` segmentation model |
| Output field used | box coordinates | box coordinates + `mask` |

## How Geolocation Uses Detections

[object_geolocator.py](/home/ja/Workspace/Drone/src/vision/object_geolocator.py) builds on the same `YoloDetection` results.

The reference pixel for each object is chosen like this:

- if a segmentation mask exists, use the centroid of mask pixels
- otherwise, use the center of the bounding box

That reference pixel is converted into a camera ray, rotated into body and world coordinates, and intersected with the ground plane to estimate object position.

This is why segmentation can improve downstream positioning: the reference point can be based on the actual segmented object shape instead of the center of the bounding box.

## Useful CLI Parameters

The main YOLO-related runtime flags are:

- `--yolo-model`: path to `.pt` or `.onnx` model
- `--yolo-classes`: label file such as `coco.names`
- `--yolo-input`: square model input size
- `--yolo-conf`: confidence threshold
- `--yolo-nms`: non-maximum suppression threshold
- `--show`: open an annotated preview window

Frame-source flags are:

- `--rtsp-url`
- `--rtsp-width`
- `--rtsp-height`
- `--bind-ip`
- `--video-port`
- `--frame-timeout`
- `--max-no-frame-seconds`

## Recommended Usage

Use detection when:

- you only need object class + box
- you want broader model compatibility
- you want the simplest runtime setup

Use segmentation when:

- you want object masks overlaid on the image
- you want a more precise object reference point
- you are using a YOLOv8 segmentation `.pt` model

## Current Limitations

- Segmentation mask decoding is currently implemented through the Ultralytics `.pt` path.
- The threaded combined pipeline does not currently draw masks even if the model returns them.
- The segmentation result is a resized boolean mask, so very fine mask edges may be softened by interpolation before thresholding.

## Quick Mental Model

If you want a short version of the architecture, it is this:

- frame providers get images into the PC runtime
- `YoloDetector` runs one shared inference API
- object detection means `YoloDetection(mask=None)`
- segmentation means `YoloDetection(mask=<boolean mask>)`
- downstream code can choose whether to use only the box or the richer mask-aware reference point
