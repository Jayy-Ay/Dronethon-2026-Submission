import cv2
import numpy as np

# --- Step 1: Generate a few markers ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 200  # pixels
physical_marker_length = 0.1  # meters, adjust to your printed marker size

for marker_id in range(5):  # generate markers 0,1,2,3,4
    marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_size)
    cv2.imwrite(f"marker_{marker_id}.png", marker_img)

print("Generated markers 0-4 as PNG files.")

# --- Step 2: Open webcam and detect markers ---
cap = cv2.VideoCapture(0)
detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

# Optional: camera calibration (replace with your real values if available)
ret, frame = cap.read()
h, w, _ = frame.shape
fx = fy = 1.2 * w  # approximate focal length
cx, cy = w / 2, h / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)
dist_coeffs = np.zeros(5)  # assume no lens distortion

# 3D points of marker corners in marker coordinate frame
obj_points = np.array([
    [-physical_marker_length/2,  physical_marker_length/2, 0],
    [ physical_marker_length/2,  physical_marker_length/2, 0],
    [ physical_marker_length/2, -physical_marker_length/2, 0],
    [-physical_marker_length/2, -physical_marker_length/2, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    output = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            # Draw marker ID text
            c = corners[i][0]
            center = c.mean(axis=0).astype(int)
            cv2.putText(output, f"ID: {marker_id}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            # --- Estimate pose and distance ---
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                distance = np.linalg.norm(tvec)
                cv2.putText(output, f"Dist: {distance:.2f} m", (center[0], center[1]+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Optional: draw axes
                cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    cv2.imshow("ArUco Detection", output)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()