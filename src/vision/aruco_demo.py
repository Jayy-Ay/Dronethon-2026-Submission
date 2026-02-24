import cv2
from frame_provider import WebcamFrameProvider
from aruco_processor import ArucoProcessor

if __name__ == "__main__":
    # Initialize webcam
    provider = WebcamFrameProvider()

    # Initialize ArUco processor
    processor = ArucoProcessor(marker_length=0.1)

    # Optional: set camera calibration for pose estimation
    # cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist_coeffs = np.zeros(5)
    # processor.set_camera_parameters(cam_matrix, dist_coeffs)

    while True:
        frame = provider.get_frame()
        if frame is None:
            break

        # Detect markers and estimate pose
        corners, ids, rejected, rvecs, tvecs = processor.detect_markers(frame, estimate_pose=False)

        # Draw markers on frame
        output = processor.draw_markers(frame, corners, ids, rvecs, tvecs)

        cv2.imshow("ArUco Detection", output)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    provider.release()
    cv2.destroyAllWindows()