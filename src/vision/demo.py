
# Activate virtual environment with .\.venv312\Scripts\Activate.ps1
import open3d as o3d
import open3d.visualization as o3d_visualization
import cv2
import numpy as np
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
model.eval()

def estimate_depth(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        d = model(**inputs).predicted_depth.squeeze().cpu().numpy()
    return cv2.resize(d, (frame.shape[1], frame.shape[0]))

def webcam_to_open3d_demo():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    cap.release()
    h, w, _ = frame.shape
    step = 10
    points = [[x, y, 0] for y in range(0, h, step) for x in range(0, w, step)]
    colors = [[frame[y, x][2]/255, frame[y, x][1]/255, frame[y, x][0]/255] for y in range(0, h, step) for x in range(0, w, step)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points, float))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors, float))
    o3d_visualization.draw_geometries([pc], window_name="Webcam Point Cloud (Open3D)")

def webcam_to_open3d_live():
    cap = cv2.VideoCapture(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Webcam Point Cloud (Open3D)")
    pc = o3d.geometry.PointCloud()
    added = False
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        h, w, _ = frame.shape
        step = 10
        points = [[x, y, 0] for y in range(0, h, step) for x in range(0, w, step)]
        colors = [[frame[y, x][2]/255, frame[y, x][1]/255, frame[y, x][0]/255] for y in range(0, h, step) for x in range(0, w, step)]
        pc.points = o3d.utility.Vector3dVector(np.array(points, float))
        pc.colors = o3d.utility.Vector3dVector(np.array(colors, float))
        if not added:
            vis.add_geometry(pc)
            added = True
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        if not vis.poll_events():
            break
    cap.release()
    vis.destroy_window()


def webcam_to_open3d_depth_live():
    cap = cv2.VideoCapture(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live 3D Point Cloud (Open3D)")

    added = False
    paused = False

    step = 2          # dense cloud
    z_scale = 2.0
    normalize_depth = True
    pc = o3d.geometry.PointCloud()

    try:
        while True:
            # Capture only if not paused
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape

                depth = estimate_depth(frame)
                d = depth.copy()
                if normalize_depth:
                    d = (d - np.min(d)) / (np.max(d) - np.min(d) + 1e-8)
                    d = 1.0 - d
                d *= z_scale

                fx, fy = 1.2 * w, 1.2 * h
                cx, cy = w / 2, h / 2

                # Convert to RGBD and generate point cloud
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    o3d.geometry.Image((d * 1000).astype(np.uint16)),
                    depth_scale=1000.0,
                    convert_rgb_to_intensity=False,
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
                pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                # Flip the point cloud vertically (right side up)
                flip = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, h],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
                pc.transform(flip)

                if not added:
                    vis.add_geometry(pc)
                    added = True
                else:
                    vis.update_geometry(pc)

            # Refresh visualization
            vis.poll_events()
            vis.update_renderer()

            # --- Keyboard controls ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or not vis.poll_events():  # ESC or close window
                break
            elif key == 32:  # SPACE toggles pause
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")

    finally:
        cap.release()
        vis.destroy_window()

if __name__ == "__main__":
    webcam_to_open3d_depth_live()