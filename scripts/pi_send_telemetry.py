#!/usr/bin/env python3
"""Send drone telemetry from Raspberry Pi to a PC over UDP."""

import argparse
import random
import time

from src.core.communication import UDPTelemetrySender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send telemetry from Raspberry Pi to PC")
    parser.add_argument("--pc-ip", required=True, help="IPv4 address of your PC")
    parser.add_argument("--port", type=int, default=9000, help="UDP port on your PC (default: 9000)")
    parser.add_argument("--rate-hz", type=float, default=5.0, help="Messages per second (default: 5)")
    parser.add_argument("--source", default="drone-pi", help="Source label in telemetry packets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.2

    sender = UDPTelemetrySender(args.pc_ip, args.port, source=args.source)
    print(f"Sending telemetry to {args.pc_ip}:{args.port} at {args.rate_hz:.2f} Hz")

    try:
        altitude_m = 0.5
        battery_v = 16.8
        while True:
            altitude_m = max(0.0, altitude_m + random.uniform(-0.05, 0.08))
            battery_v = max(14.0, battery_v - random.uniform(0.0005, 0.003))
            payload = {
                "altitude_m": round(altitude_m, 3),
                "battery_v": round(battery_v, 3),
                "speed_mps": round(random.uniform(0.0, 6.0), 3),
                "gps": {
                    "lat": 51.52 + random.uniform(-0.0002, 0.0002),
                    "lon": -0.13 + random.uniform(-0.0002, 0.0002),
                },
                "status": "ok",
            }
            seq = sender.send(payload)
            print(f"sent seq={seq} altitude={payload['altitude_m']}m battery={payload['battery_v']}V")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping sender")
    finally:
        sender.close()


if __name__ == "__main__":
    main()
