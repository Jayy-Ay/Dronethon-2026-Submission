"""YOLO detector running on PC-side frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    """OpenCV DNN-based YOLO detector for ONNX models."""

    def __init__(
        self,
        model_path: str,
        classes_path: str,
        input_size: int = 640,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
    ) -> None:
        model = Path(model_path)
        classes = Path(classes_path)

        if not model.exists():
            raise FileNotFoundError(f"YOLO model not found: {model}")
        if not classes.exists():
            raise FileNotFoundError(f"Class names file not found: {classes}")

        self._net = cv2.dnn.readNetFromONNX(str(model))
        self._labels = classes.read_text(encoding="utf-8").strip().splitlines()
        self._input_size = int(input_size)
        self._conf_thresh = float(conf_thresh)
        self._nms_thresh = float(nms_thresh)

    def _preprocess(self, frame: np.ndarray):
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
        h, w = frame_shape[:2]
        pred = np.array(output)

        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            return []

        # Supports common ONNX layouts: [84, N] and [N, 84]
        if pred.shape[0] <= 100 and pred.shape[1] > pred.shape[0]:
            pred = pred.T

        if pred.shape[1] < 6:
            return []

        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
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

    def detect(self, frame: np.ndarray) -> List[YoloDetection]:
        blob, scale, pad_left, pad_top = self._preprocess(frame)
        self._net.setInput(blob)
        outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return self._postprocess(output, frame.shape, scale, pad_left, pad_top)
