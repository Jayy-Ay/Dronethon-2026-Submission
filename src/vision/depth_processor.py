import numpy as np
import torch
import open3d as o3d
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import cv2

# Load GLPN model
processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
model.eval()

def estimate_depth(frame):
    """Estimate depth map from a frame"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        d = model(**inputs).predicted_depth.squeeze().cpu().numpy()
    return cv2.resize(d, (frame.shape[1], frame.shape[0]))

def frame_to_point_cloud(frame, depth=None, step=2, z_scale=2.0, normalize_depth=True):
    """Convert frame (and optional depth map) to Open3D point cloud"""
    h, w, _ = frame.shape
    if depth is None:
        depth = estimate_depth(frame)
    d = depth.copy()
    if normalize_depth:
        d = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-8)
        d = 1.0 - d
    d *= z_scale

    fx, fy = 1.2 * w, 1.2 * h
    cx, cy = w / 2, h / 2

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image((d * 1000).astype(np.uint16)),
        depth_scale=1000.0,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    flip = np.array([[1, 0, 0, 0],
                     [0, -1, 0, h],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    pc.transform(flip)
    return pc