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
    """Launch ffmpeg to decode an RTSP stream into raw BGR frames on stdout."""
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
    """Read an exact number of bytes from a pipe, or return None on EOF."""
    buf = b""
    while len(buf) < size:
        chunk = stdout.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_frame(stdout, width, height):
    """Read one raw BGR frame from an ffmpeg stdout pipe."""
    frame_size = width * height * 3
    raw = _read_exact(stdout, frame_size)

    if raw is None:
        return None

    return np.frombuffer(raw, np.uint8).reshape((height, width, 3))

class StreamFrameProvider:
    """Video provider from raw UDP JPEG stream.

    Expects packets in the custom `MJPG_HDR` format and reassembles each frame
    from numbered chunks before JPEG decoding.
    """

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
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._frame_version = 0
        self._last_consumed_version = 0
        self.start_receiver()
        # Packet prefix: 8-byte identifier + frame_id + total_chunks + chunk_id.
        self.HEADER_LENGTH = 16
        self.IDENTIFIER = b"MJPG_HDR"

    def get_frame(self):
        """Return the most recently assembled frame without blocking."""
        with self._lock:
            return self.latest_frame

    def _process_packet(self, packet):
        """Accumulate one UDP packet and decode the JPEG once all chunks arrive."""
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

        # Once all chunks for a frame arrive, rebuild the JPEG payload in order.
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
        """Wait for and return the next unseen frame, or None on timeout."""
        deadline = time.time() + timeout
        with self._frame_ready:
            while self._frame_version <= self._last_consumed_version:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print("Frame timeout reached — no complete frame received.")
                    return None
                self._frame_ready.wait(timeout=remaining)

            self._last_consumed_version = self._frame_version
            return self.latest_frame

    def _receive_loop(self):
        """Receive UDP chunks forever and publish completed JPEG frames."""
        while True:
            try:
                packet, _ = self.sock.recvfrom(65000)
            except socket.timeout:
                print("packet timeout")
                self._cleanup_old_frames()
                continue

            frame = self._process_packet(packet)

            if frame is not None:
                with self._frame_ready:
                    self.latest_frame = frame
                    self._frame_version += 1
                    self._frame_ready.notify_all()

    def start_receiver(self):
        """Start the background thread that assembles incoming UDP frames."""
        thread = threading.Thread(target=self._receive_loop, daemon=True)
        thread.start()

    def _cleanup_old_frames(self):
        """Drop incomplete frames that have exceeded the chunk assembly timeout."""
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
        """Close the UDP socket used by the stream receiver."""
        self.sock.close()


class RtspFrameProvider:
    """Low-latency frame provider backed by an ffmpeg RTSP pipe."""

    def __init__(self, url, width=1280, height=720):
        self.url = url
        self.width = width
        self.height = height

        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self.latest_frame = None
        self._frame_version = 0
        self._last_consumed_version = 0

        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def get_frame(self):
        """Return the latest RTSP frame without waiting for a fresh one."""
        with self._lock:
            return self.latest_frame

    def get_frame_with_timeout(self, timeout=2.0):
        """Wait for a new RTSP frame version, or return None on timeout."""
        deadline = time.time() + timeout
        with self._frame_ready:
            while self._frame_version <= self._last_consumed_version and not self._stop.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._frame_ready.wait(timeout=remaining)

            if self._frame_version <= self._last_consumed_version:
                return None

            self._last_consumed_version = self._frame_version
            return self.latest_frame

    def _receive_loop(self):
        """Continuously reconnect to ffmpeg and publish frames until stopped."""
        while not self._stop.is_set():
            # Keep the provider resilient to transient RTSP/ffmpeg disconnects.
            process = _start_ffmpeg_stream(self.url, self.width, self.height)
            self._process = process

            try:
                while not self._stop.is_set():
                    frame = _read_frame(process.stdout, self.width, self.height)
                    if frame is None:
                        break

                    with self._frame_ready:
                        self.latest_frame = frame
                        self._frame_version += 1
                        self._frame_ready.notify_all()
            finally:
                self._terminate_process(process)

            if not self._stop.is_set():
                time.sleep(0.5)

    def _terminate_process(self, process):
        """Stop an ffmpeg subprocess, escalating to kill if needed."""
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def close(self):
        """Signal the receiver thread to stop and tear down ffmpeg cleanly."""
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        process = self._process
        if process is not None:
            self._terminate_process(process)
        self._thread.join(timeout=2.0)

class WebcamFrameProvider:
    """Simple webcam frame provider"""
    def __init__(self, cam_index=0, flip_vertically=False):
        """Open a local webcam device for synchronous frame reads."""
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
        """Release the webcam capture handle."""
        self.cap.release()

class VideoFileFrameProvider:
    """Frame provider from a video file"""
    def __init__(self, filename, flip_vertically=False):
        """Open a video file for sequential frame reads."""
        self.cap = cv2.VideoCapture(filename)
        self.flip_vertically = flip_vertically

    def get_frame(self):
        """Return the next decoded frame from the file, or None at EOF."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.flip_vertically:
            frame = cv2.flip(frame, 0)
        return frame

    def release(self):
        """Release the underlying video file handle."""
        self.cap.release()
