import cv2
import socket
import numpy as np
import struct
import time
from collections import defaultdict
import threading


FRAME_TIMEOUT = 0.2  # 200ms timeout

class StreamFrameProvider:
    """Video provider from raw UDP JPEG stream"""

    def __init__(self, ip="127.0.0.1", port=5600):
        self.port = port
        self.ip = ip

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(2.0)
        print(f"Binding stream source at {self.ip}:{self.port}")

        self.frames = defaultdict(dict)
        self.expected_chunks = {}
        self.frame_timestamps = {}
        self.latest_frame = None
        self.start_receiver()
        self.HEADER_LENGTH = 16
        self.IDENTIFIER = b"MJPG_HDR"

    def get_frame(self):
        return self.latest_frame

    def _process_packet(self, packet):
        """Extract the data from the packet and combine it with the correct frame"""
        if len(packet) < self.HEADER_LENGTH:
            return None

        identifier, frame_id, total_chunks, chunk_id = struct.unpack(
            "!8sIHH", packet[:self.HEADER_LENGTH]
        )

        if identifier != self.IDENTIFIER:
            return None

        payload = packet[self.HEADER_LENGTH:]

        # Store timestamp for this frame
        if frame_id not in self.frame_timestamps:
            self.frame_timestamps[frame_id] = time.time()

        # Store chunk
        if frame_id not in self.frames:
            self.frames[frame_id] = {}

        self.frames[frame_id][chunk_id] = payload
        self.expected_chunks[frame_id] = total_chunks

        # If complete -> assemble
        if len(self.frames[frame_id]) == total_chunks:

            chunks = [
                self.frames[frame_id][i]
                for i in range(total_chunks)
                if i in self.frames[frame_id]
            ]

            frame_data = b"".join(chunks)

            # Cleanup
            del self.frames[frame_id]
            del self.expected_chunks[frame_id]
            del self.frame_timestamps[frame_id]

            npdata = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

            return frame
        return None

    def get_frame_with_timeout(self, timeout=2.0):
        """
        Keep trying to get a full frame until timeout is reached.
        Returns frame or None if timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            frame = self.get_frame()

            if frame is not None:
                return frame

        print("Frame timeout reached — no complete frame received.")
        return None

    def _receive_loop(self):
        while True:
            try:
                packet, _ = self.sock.recvfrom(65000)
            except socket.timeout:
                print("packet timeout")
                self._cleanup_old_frames()
                continue

            frame = self._process_packet(packet)

            if frame is not None:
                self.latest_frame = frame

    def start_receiver(self):
        """Start a new thread to constantly receive packets"""
        thread = threading.Thread(target=self._receive_loop, daemon=True)
        thread.start()

    def _cleanup_old_frames(self):
        """Remove frames that timed out before completion"""
        now = time.time()
        expired_frames = []

        for frame_id, timestamp in self.frame_timestamps.items():
            if now - timestamp > FRAME_TIMEOUT:
                expired_frames.append(frame_id)

        for frame_id in expired_frames:
            print(f"Dropping incomplete frame {frame_id}")
            self.frames.pop(frame_id, None)
            self.expected_chunks.pop(frame_id, None)
            self.frame_timestamps.pop(frame_id, None)

    def close(self):
        self.sock.close()

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
