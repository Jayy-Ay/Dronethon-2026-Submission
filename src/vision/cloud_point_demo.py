import open3d as o3d
from frame_provider import WebcamFrameProvider
from depth_processor import frame_to_point_cloud

if __name__ == "__main__":
    provider = WebcamFrameProvider()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live Depth Point Cloud")
    pc = o3d.geometry.PointCloud()
    added = False
    paused = False

    try:
        while True:
            if not paused:
                frame = provider.get_frame()
                if frame is None:
                    break
                pc = frame_to_point_cloud(frame)

                if not added:
                    vis.add_geometry(pc)
                    added = True
                else:
                    vis.update_geometry(pc)

            vis.poll_events()
            vis.update_renderer()

            key = 0xFF & int(vis.poll_events())  # Placeholder for key detection
            # Example: use ESC to exit or SPACE to pause
            # (you can implement proper keyboard callbacks in Open3D)
            
    finally:
        provider.release()
        vis.destroy_window()