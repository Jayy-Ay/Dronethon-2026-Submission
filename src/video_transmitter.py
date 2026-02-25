import cv2
import socket
import argparse
import time
import math
import struct

MAX_UDP_PAYLOAD = 60000  # safe chunk size


class UdpVideoStreamer:
    def __init__(self, dest_ip="127.0.0.1", port=5600,
                 width=640, height=480, fps=30, quality=80):
        self.dest_ip = dest_ip
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.frame_id = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        print("Actual width:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Actual height:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0
        self.IDENTIFIER = b"MJPG_HDR" # Should be 8 chars long - or we got problems

    def start(self):
        print(f"Streaming to {self.dest_ip}:{self.port}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"JPEG Quality: {self.quality}")

        try:
            while True:
                start_time = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                )

                if not success:
                    continue

                self.send_frame_fragmented(buffer.tobytes())

                # FPS limiting
                elapsed = time.time() - start_time
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping stream...")

        finally:
            self.cap.release()
            self.sock.close()

    def send_frame_fragmented(self, frame_bytes):
        frame_id = self.frame_id
        self.frame_id += 1

        total_chunks = math.ceil(len(frame_bytes) / MAX_UDP_PAYLOAD)

        for chunk_id in range(total_chunks):
            start = chunk_id * MAX_UDP_PAYLOAD
            end = start + MAX_UDP_PAYLOAD
            chunk = frame_bytes[start:end]

            # Header:
            # IDENTIFIER (8 bytes) -- mainly for packet debugging
            # frame_id (4 bytes)
            # total_chunks (2 bytes)
            # chunk_id (2 bytes)

            header = struct.pack(
                "!8sIHH",
                self.IDENTIFIER,
                frame_id,
                total_chunks,
                chunk_id
            )

            packet = header + chunk
            self.sock.sendto(packet, (self.dest_ip, self.port))


def main():
    parser = argparse.ArgumentParser(description="UDP MJPEG Video Streamer")

    parser.add_argument("--ip", default="127.0.0.1",
                        help="Destination IP address")
    parser.add_argument("--port", type=int, default=5600,
                        help="Destination port")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second")
    parser.add_argument("--width", type=int, default=640,
                        help="Frame width")
    parser.add_argument("--height", type=int, default=480,
                        help="Frame height")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality (0-100)")

    args = parser.parse_args()

    streamer = UdpVideoStreamer(
        dest_ip=args.ip,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality
    )

    streamer.start()


if __name__ == "__main__":
    main()
