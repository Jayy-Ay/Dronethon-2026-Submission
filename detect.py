import cv2
import numpy as np
from pathlib import Path
from src.vision.frame_provider import RtspFrameProvider

# ── Config ────────────────────────────────────────────────
DEFAULT_STREAM = "rtsp://dronetastic.local:8554/cam1"
MODEL_PATH = "yolov8s-seg.onnx"
INPUT_SIZE = 320
CONF_THRESH = 0.4
IOU_THRESH = 0.45
CLASSES = Path(__file__).with_name("coco.names").read_text(encoding="utf-8").strip().splitlines()

# Random colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)
# ─────────────────────────────────────────────────────────

def preprocess(img, input_size):
    """Letterbox an image for YOLO input and return scale plus padding."""
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top  = (input_size - nh) // 2
    pad_left = (input_size - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    return canvas, scale, pad_left, pad_top

def is_red(frame, box):
    """Heuristically classify whether a detected region contains mostly red pixels."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_pixels = cv2.countNonZero(mask1 + mask2)
    total_pixels = roi.shape[0] * roi.shape[1]
    return (red_pixels / total_pixels) > 0.3

def postprocess(outputs, orig_shape, scale, pad_left, pad_top):
    """Convert raw YOLOv8 segmentation outputs into detection dictionaries."""
    oh, ow = orig_shape[:2]

    pred = np.array(outputs[0])
    if pred.ndim == 3:
        pred = pred[0].T
    elif pred.ndim == 2:
        pred = pred.T

    proto = np.array(outputs[1])
    if proto.ndim == 4:
        proto = proto[0]

    # Dynamically get prototype dimensions
    proto_h, proto_w = proto.shape[1], proto.shape[2]

    boxes      = pred[:, :4]
    scores     = pred[:, 4:84]
    mask_coefs = pred[:, 84:]

    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > CONF_THRESH
    boxes, confidences, class_ids, mask_coefs = (
        boxes[mask], confidences[mask], class_ids[mask], mask_coefs[mask]
    )

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    x1 = ((x1 - pad_left) / scale).clip(0, ow)
    x2 = ((x2 - pad_left) / scale).clip(0, ow)
    y1 = ((y1 - pad_top)  / scale).clip(0, oh)
    y2 = ((y2 - pad_top)  / scale).clip(0, oh)

    bboxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(bboxes, confidences.tolist(), CONF_THRESH, IOU_THRESH)

    results = []
    for i in indices:
        mask_map = (mask_coefs[i] @ proto.reshape(32, -1)).reshape(proto_h, proto_w)
        mask_map = 1 / (1 + np.exp(-mask_map))
        mask_map = cv2.resize(mask_map, (INPUT_SIZE, INPUT_SIZE))
        mask_map = mask_map[pad_top:pad_top+int(oh*scale), pad_left:pad_left+int(ow*scale)]
        mask_map = cv2.resize(mask_map, (ow, oh))
        mask_bin = (mask_map > 0.5).astype(np.uint8)

        results.append({
            "box": [int(v) for v in bboxes[i]],
            "confidence": float(confidences[i]),
            "class_id": int(class_ids[i]),
            "label": CLASSES[int(class_ids[i])],
            "mask": mask_bin
        })
    return results

def draw(frame, detections):
    """Render segmentation masks, boxes, and labels onto a frame copy."""
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cid   = det["class_id"]
        color = tuple(int(c) for c in COLORS[cid])

        if det["label"] == "sports ball" and is_red(frame, det["box"]):
            label = f"RED BALL {det['confidence']:.2f}"
            color = (0, 0, 255)
        else:
            label = f"{det['label']} {det['confidence']:.2f}"

        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[det["mask"] == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return overlay

# ── Load model ────────────────────────────────────────────
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# ── Open RTSP stream ──────────────────────────────────────
provider = RtspFrameProvider(DEFAULT_STREAM, width=1280, height=720)
frame = provider.get_frame_with_timeout(timeout=5.0)

if frame is None:
    raise RuntimeError(f"Failed to read initial frame from {DEFAULT_STREAM}")

while True:
    frame = provider.get_frame()
    if frame is None:
        continue

    input_img, scale, pad_left, pad_top = preprocess(frame, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    detections = postprocess(outputs, frame.shape, scale, pad_left, pad_top)
    result = draw(frame, detections)

    cv2.imshow("YOLOv8 Segmentation", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

provider.close()
cv2.destroyAllWindows()
