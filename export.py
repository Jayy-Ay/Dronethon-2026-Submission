from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt")
model.export(format="onnx", imgsz=320, opset=12, simplify=True)