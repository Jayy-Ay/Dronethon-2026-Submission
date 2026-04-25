import cv2
import socket
import numpy as np
import struct
import subprocess
import threading
import time
from collections import defaultdict
from typing import Optional


FRAME_TIMEOUT = 0.2  # 200ms timeout


def _start_ffmpeg_stream(url, width, height):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "0",
        "-probesize", "32",
        "-i", url,
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-f", "rawvideo",
        "pipe:1",
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def _read_exact(stdout, size):
    buf = b""
    while len(buf) < size:
        chunk = stdout.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_frame(stdout, width, height):
    frame_size = width * height * 3
    raw = _read_exact(stdout, frame_size)

    if raw is None:
        return None

    return np.frombuffer(raw, np.uint8).reshape((height, width, 3))

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


class RtspFrameProvider:
    """Low-latency frame provider backed by an ffmpeg RTSP pipe."""

    def __init__(self, url, width=1280, height=720):
        self.url = url
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self.latest_frame = None

        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def get_frame(self):
        return self.latest_frame

    def get_frame_with_timeout(self, timeout=2.0):
        if not self._frame_ready.wait(timeout=timeout):
            return None
        return self.get_frame()

    def _receive_loop(self):
        while not self._stop.is_set():
            process = _start_ffmpeg_stream(self.url, self.width, self.height)
            self._process = process

            try:
                while not self._stop.is_set():
                    frame = _read_frame(process.stdout, self.width, self.height)
                    if frame is None:
                        break

                    with self._lock:
                        self.latest_frame = frame
                        self._frame_ready.set()
            finally:
                self._terminate_process(process)

            if not self._stop.is_set():
                time.sleep(0.5)

    def _terminate_process(self, process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def close(self):
        self._stop.set()
        process = self._process
        if process is not None:
            self._terminate_process(process)
        self._thread.join(timeout=2.0)

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
