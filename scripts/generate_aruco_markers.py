#!/usr/bin/env python3

"""Generate printable ArUco/AprilTag marker images.

Examples:
    python scripts/generate_aruco_markers.py --family tag36h11 --count 10 --size 800
    python scripts/generate_aruco_markers.py --family 6x6_250 --count 20 --output-dir artifacts/aruco_markers
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2


FAMILY_MAP = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate printable ArUco/AprilTag marker PNGs")
    parser.add_argument("--family", default="tag36h11", choices=sorted(FAMILY_MAP), help="Marker family")
    parser.add_argument("--count", type=int, default=5, help="Number of markers to generate")
    parser.add_argument("--size", type=int, default=800, help="Output image size in pixels")
    parser.add_argument("--output-dir", default="artifacts/aruco_markers", help="Directory for generated PNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[args.family])

    for marker_id in range(args.count):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, args.size)
        output_path = output_dir / f"{args.family}_id_{marker_id:03d}.png"
        cv2.imwrite(str(output_path), marker_img)

    print(f"Generated {args.count} markers in {output_dir}")


if __name__ == "__main__":
    main()