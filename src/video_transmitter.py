import cv2
import socket
import argparse
import time


class UdpVideoStreamer:
    def __init__(self, dest_ip="127.0.0.1", port=5600,
                 width=640, height=480, fps=30, quality=80):
        self.dest_ip = dest_ip
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.frame_interval = 1.0 / self.fps if self.fps > 0 else 0

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

                self.sock.sendto(buffer.tobytes(), (self.dest_ip, self.port))

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
