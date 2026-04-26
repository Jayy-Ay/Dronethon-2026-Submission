"""Run ArUco world-frame localisation on a live Pi video stream."""

from __future__ import annotations
import argparse
import time
import cv2
import numpy as np
from src.localization import ArucoWorldLocalizer, MarkerMap
from src.stages.aruco_detector import ArucoDetection, ArucoDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the world-frame ArUco localisation demo."""
    parser = argparse.ArgumentParser(description="Run ArUco floor-map localisation demo")
    parser.add_argument("--rtsp-url", default="rtsp://dronetastic.local:8554/cam1", help="Optional RTSP URL")
    parser.add_argument("--rtsp-width", type=int, default=1280, help="RTSP decode width")
    parser.add_argument("--rtsp-height", type=int, default=720, help="RTSP decode height")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="UDP bind IP when --rtsp-url is empty")
    parser.add_argument("--video-port", type=int, default=5600, help="UDP bind port when --rtsp-url is empty")
    parser.add_argument("--family", default="6x6_250", help="ArUco family")
    parser.add_argument("--marker-length-m", type=float, required=True, help="Printed marker edge length in meters")
    parser.add_argument("--area-width-m", type=float, required=True, help="Distance from marker 0 center to marker 1 center")
    parser.add_argument("--area-height-m", type=float, required=True, help="Distance from marker 0 center to marker 2 center")
    parser.add_argument("--camera-fx", type=float, required=True, help="Camera focal length fx in pixels")
    parser.add_argument("--camera-fy", type=float, required=True, help="Camera focal length fy in pixels")
    parser.add_argument("--camera-cx", type=float, default=0.0, help="Camera principal point cx in pixels")
    parser.add_argument("--camera-cy", type=float, default=0.0, help="Camera principal point cy in pixels")
    parser.add_argument("--smoothing-alpha", type=float, default=0.2, help="0 disables smoothing; higher is smoother")
    parser.add_argument("--show", action="store_true", help="Show annotated preview window")
    parser.add_argument("--frame-timeout", type=float, default=3.0, help="Seconds to wait per frame")
    parser.add_argument("--max-no-frame-seconds", type=float, default=60.0, help="Exit after this no-frame gap")
    return parser.parse_args()


def draw_overlay(
    frame: np.ndarray,
    detections: list[ArucoDetection],
    pose_text: list[str],
) -> np.ndarray:
    """Draw detected markers plus current world-frame pose text."""
    vis = frame.copy()
    for det in detections:
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        x = int(det.center_x)
        y = int(det.center_y)
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"ID {det.tag_id}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    y = 24
    for line in pose_text:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y += 24
    return vis


def main() -> None:
    """Run the live localisation demo."""
    args = parse_args()

    provider: RtspFrameProvider | StreamFrameProvider
    if args.rtsp_url:
        provider = RtspFrameProvider(args.rtsp_url, width=args.rtsp_width, height=args.rtsp_height)
        print(f"Receiving RTSP frames from {args.rtsp_url}")
    else:
        provider = StreamFrameProvider(ip=args.bind_ip, port=args.video_port)
        print(f"Listening for UDP video stream on {args.bind_ip}:{args.video_port}")

    cx = args.camera_cx if args.camera_cx > 0 else args.rtsp_width / 2.0
    cy = args.camera_cy if args.camera_cy > 0 else args.rtsp_height / 2.0
    camera_matrix = np.array(
        [[args.camera_fx, 0.0, cx], [0.0, args.camera_fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros(5, dtype=np.float32)

    detector = ArucoDetector(
        family=args.family,
        marker_length_m=args.marker_length_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    marker_map = MarkerMap.rectangular_floor_map(
        marker_length_m=args.marker_length_m,
        area_width_m=args.area_width_m,
        area_height_m=args.area_height_m,
    )
    localizer = ArucoWorldLocalizer(
        marker_map=marker_map,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        smoothing_alpha=args.smoothing_alpha,
    )

    print("Localisation demo running")
    print("World frame: marker 0 at (0,0,0), +X toward marker 1, +Y toward marker 2, +Z upward")

    last_log = 0.0
    last_frame_time = time.time()
    last_no_frame_log = 0.0

    try:
        while True:
            frame = provider.get_frame_with_timeout(timeout=args.frame_timeout)
            if frame is None:
                now = time.time()
                gap = now - last_frame_time
                if now - last_no_frame_log >= 2.0:
                    print(f"Waiting for frame... no-frame-for={gap:.1f}s")
                    last_no_frame_log = now
                if gap >= args.max_no_frame_seconds:
                    print(f"No frame for {gap:.1f}s, exiting")
                    break
                continue

            last_frame_time = time.time()
            detections = detector.detect(frame)
            pose = localizer.estimate_pose(detections)

            pose_lines = [f"visible markers: {[det.tag_id for det in detections]}"]
            if pose.success:
                cam_x, cam_y, cam_z = pose.camera_position_world_m
                pose_lines.append(f"camera world xyz: {cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f} m")
                if pose.body_position_world_m is not None:
                    body_x, body_y, body_z = pose.body_position_world_m
                    pose_lines.append(f"drone world xyz: {body_x:.3f}, {body_y:.3f}, {body_z:.3f} m")
                if pose.reprojection_error_px is not None:
                    pose_lines.append(f"reprojection error: {pose.reprojection_error_px:.2f} px")
            else:
                pose_lines.append("pose unavailable")

            now = time.time()
            if now - last_log >= 1.0:
                if pose.success:
                    print(
                        "pose "
                        f"markers={pose.marker_ids} "
                        f"camera_xyz_m={tuple(round(v, 3) for v in pose.camera_position_world_m)} "
                        f"reproj_px={pose.reprojection_error_px:.2f}"
                    )
                else:
                    print(f"pose unavailable markers={[det.tag_id for det in detections]}")
                last_log = now

            if args.show:
                vis = draw_overlay(frame, detections, pose_lines)
                cv2.imshow("ArUco Localisation Demo", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        print("\nStopping localisation demo")
    finally:
        provider.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
