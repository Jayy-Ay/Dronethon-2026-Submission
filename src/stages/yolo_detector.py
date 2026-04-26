"""YOLO detector running on PC-side frames."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import cv2
import numpy as np


@dataclass(frozen=True)
class YoloDetection:
    """Single object detection with class and confidence."""

    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class YoloDetector:
    """YOLO detector supporting Ultralytics `.pt` and OpenCV DNN `.onnx` backends."""

    def __init__(
        self,
        model_path: str,
        classes_path: str,
        input_size: int = 640,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
    ) -> None:
        """Load the model and class labels for later frame inference."""
        model = Path(model_path)
        classes = Path(classes_path)

        if not model.exists():
            raise FileNotFoundError(f"YOLO model not found: {model}")
        if not classes.exists():
            raise FileNotFoundError(f"Class names file not found: {classes}")

        self._backend = "ultralytics" if model.suffix.lower() == ".pt" else "opencv-dnn"
        self._net: Optional[cv2.dnn.Net] = None
        self._ultralytics_model: Optional[Any] = None
        self._labels = classes.read_text(encoding="utf-8").strip().splitlines()
        self._model_family = self._infer_model_family(model.stem.lower())
        self._input_size = int(input_size)
        self._conf_thresh = float(conf_thresh)
        self._nms_thresh = float(nms_thresh)
        self._device = self._configure_inference_device(model)

    def _configure_inference_device(self, model: Path) -> str:
        """Prefer CUDA inference by default, falling back to CPU when unavailable."""
        if self._backend == "ultralytics":
            return self._configure_ultralytics_backend(model)
        return self._configure_opencv_backend(model)

    def _configure_ultralytics_backend(self, model: Path) -> str:
        """Load a PyTorch-backed YOLOv8 model and prefer CUDA when available."""
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO .pt models require both 'torch' and 'ultralytics' to be installed in the active environment."
            ) from exc

        self._ultralytics_model = YOLO(str(model))
        if torch.cuda.is_available():
            self._ultralytics_model.to("cuda:0")
            return "cuda"
        return "cpu"

    def _configure_opencv_backend(self, model: Path) -> str:
        """Load an ONNX network for OpenCV DNN inference."""
        self._net = cv2.dnn.readNetFromONNX(str(model))
        cuda = getattr(cv2, "cuda", None)
        has_cuda_device = False
        if cuda is not None and hasattr(cuda, "getCudaEnabledDeviceCount"):
            try:
                has_cuda_device = cuda.getCudaEnabledDeviceCount() > 0
            except cv2.error:
                has_cuda_device = False

        if has_cuda_device:
            try:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                target_fp16 = getattr(cv2.dnn, "DNN_TARGET_CUDA_FP16", None)
                if target_fp16 is not None:
                    self._net.setPreferableTarget(target_fp16)
                    return "cuda-fp16"
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                return "cuda"
            except cv2.error:
                # Fall back when OpenCV exposes cv2.cuda but DNN CUDA support is not built in.
                pass

        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return "cpu"

    @property
    def device(self) -> str:
        """Return the active inference device."""
        return self._device

    @property
    def backend(self) -> str:
        """Return the active inference backend."""
        return self._backend

    def _preprocess(self, frame: np.ndarray):
        """Letterbox a frame and return the blob plus inverse mapping metadata."""
        h, w = frame.shape[:2]
        scale = self._input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_top = (self._input_size - nh) // 2
        pad_left = (self._input_size - nw) // 2
        canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self._input_size, self._input_size), swapRB=True)
        return blob, scale, pad_left, pad_top

    def _postprocess(self, output: np.ndarray, frame_shape, scale: float, pad_left: int, pad_top: int) -> List[YoloDetection]:
        """Project raw model output back into image-space YoloDetection items."""
        h, w = frame_shape[:2]
        pred = np.array(output)

        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            return []

        # Supports common ONNX layouts: [84, N] and [N, 84]
        if pred.shape[0] <= 100 and pred.shape[1] > pred.shape[0] and pred.shape[1] > 100:
            pred = pred.T

        if pred.shape[1] < 6:
            return []

        boxes_xywh = pred[:, :4]
        if pred.shape[1] >= 6 and self._uses_yolov5_scoring(pred):
            objectness = pred[:, 4]
            class_scores = pred[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        else:
            class_scores = pred[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences > self._conf_thresh

        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        bw = boxes_xywh[:, 2]
        bh = boxes_xywh[:, 3]

        x1 = ((x - bw / 2 - pad_left) / scale).clip(0, w - 1)
        y1 = ((y - bh / 2 - pad_top) / scale).clip(0, h - 1)
        x2 = ((x + bw / 2 - pad_left) / scale).clip(0, w - 1)
        y2 = ((y + bh / 2 - pad_top) / scale).clip(0, h - 1)

        nms_boxes = []
        for i in range(len(x1)):
            nms_boxes.append([
                int(x1[i]),
                int(y1[i]),
                int(max(1, x2[i] - x1[i])),
                int(max(1, y2[i] - y1[i])),
            ])

        idxs = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self._conf_thresh, self._nms_thresh)
        if len(idxs) == 0:
            return []

        idxs = np.array(idxs).reshape(-1)
        detections: List[YoloDetection] = []
        for idx in idxs:
            cid = int(class_ids[idx])
            label = self._labels[cid] if 0 <= cid < len(self._labels) else f"class_{cid}"
            detections.append(
                YoloDetection(
                    class_id=cid,
                    label=label,
                    confidence=float(confidences[idx]),
                    x1=int(x1[idx]),
                    y1=int(y1[idx]),
                    x2=int(x2[idx]),
                    y2=int(y2[idx]),
                )
            )

        return detections

    @staticmethod
    def _infer_model_family(model_name: str) -> str:
        """Infer the YOLO family from the model filename when possible."""
        if "yolov5" in model_name:
            return "yolov5"
        if "yolov8" in model_name:
            return "yolov8"
        return "auto"

    def _uses_yolov5_scoring(self, pred: np.ndarray) -> bool:
        """Return whether detections should use YOLOv5 objectness * class score."""
        if self._model_family == "yolov5":
            return True
        if self._model_family == "yolov8":
            return False
        return self._looks_like_yolov5(pred)

    @staticmethod
    def _looks_like_yolov5(pred: np.ndarray) -> bool:
        """Heuristically identify YOLOv5-style output with objectness at index 4."""
        if pred.shape[1] < 7:
            return False

        objectness = pred[:, 4]
        if objectness.size == 0:
            return False

        return bool(
            np.all((objectness >= 0.0) & (objectness <= 1.0))
            and np.any(objectness > 0.0)
        )

    def detect(self, frame: np.ndarray) -> List[YoloDetection]:
        """Run one forward pass and return NMS-filtered detections for a frame."""
        if self._backend == "ultralytics":
            return self._detect_ultralytics(frame)

        if self._net is None:
            raise RuntimeError("OpenCV DNN backend was not initialized")

        blob, scale, pad_left, pad_top = self._preprocess(frame)
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return self._postprocess(output, frame.shape, scale, pad_left, pad_top)

    def _detect_ultralytics(self, frame: np.ndarray) -> List[YoloDetection]:
        """Run YOLOv8 via Ultralytics and convert detections into the shared dataclass."""
        if self._ultralytics_model is None:
            raise RuntimeError("Ultralytics backend was not initialized")

        results = self._ultralytics_model.predict(
            source=frame,
            imgsz=self._input_size,
            conf=self._conf_thresh,
            iou=self._nms_thresh,
            device="cuda:0" if self._device == "cuda" else "cpu",
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: List[YoloDetection] = []
        names = result.names if hasattr(result, "names") else {}
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        for coords, confidence, class_id in zip(xyxy, confs, class_ids):
            x1, y1, x2, y2 = coords
            label = names.get(class_id, f"class_{class_id}") if isinstance(names, dict) else f"class_{class_id}"
            detections.append(
                YoloDetection(
                    class_id=int(class_id),
                    label=str(label),
                    confidence=float(confidence),
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )

        return detections
