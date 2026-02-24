import cv2

class WebcamFrameProvider:
    """Simple webcam frame provider"""
    def __init__(self, cam_index=0, flip_vertically=False):
        self.cap = cv2.VideoCapture(cam_index)
        self.flip_vertically = flip_vertically

    def get_frame(self):
        """Return a single frame from the webcam"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.flip_vertically:
            frame = cv2.flip(frame, 0)
        return frame

    def release(self):
        self.cap.release()

class VideoFileFrameProvider:
    """Frame provider from a video file"""
    def __init__(self, filename, flip_vertically=False):
        self.cap = cv2.VideoCapture(filename)
        self.flip_vertically = flip_vertically

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.flip_vertically:
            frame = cv2.flip(frame, 0)
        return frame

    def release(self):
        self.cap.release()