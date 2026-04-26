"""Export the YOLOv8 segmentation model to ONNX when run as a script."""

from ultralytics import YOLO

# This script intentionally performs the export at import/run time.
model = YOLO("yolov8s-seg.pt")
model.export(format="onnx", imgsz=320, opset=12, simplify=True)
