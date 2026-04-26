#!/usr/bin/env python3
"""Receive telemetry on a PC over UDP and print it."""

import argparse
import json
import time
from src.core.communication import UDPTelemetryReceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive telemetry from Raspberry Pi")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="Local IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Local UDP port (default: 9000)")
    parser.add_argument("--timeout", type=float, default=1.0, help="Receive timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receiver = UDPTelemetryReceiver(host=args.bind_ip, port=args.port, timeout_s=args.timeout)
    print(f"Listening for telemetry on {args.bind_ip}:{args.port}")

    last_seen = time.time()

    try:
        while True:
            packet = receiver.receive()
            if packet is None:
                if time.time() - last_seen > 5.0:
                    print("No telemetry in last 5s...")
                    last_seen = time.time()
                continue

            last_seen = time.time()
            print(
                f"seq={packet.seq} source={packet.source} ts={packet.timestamp:.3f} "
                f"payload={json.dumps(packet.payload, separators=(',', ':'))}"
            )
    except KeyboardInterrupt:
        print("\nStopping receiver")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
