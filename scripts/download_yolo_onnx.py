#!/usr/bin/env python3
"""Download a YOLO ONNX model to the local Drone project (PC side).

Example:
    python scripts/download_yolo_onnx.py --model yolov8s --output yolov8s.onnx
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YOLO ONNX model for PC inference")
    parser.add_argument(
        "--model",
        default="yolov8s",
        help="YOLO model base name without extension (default: yolov8s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path (default: <model>.onnx in project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def candidate_urls(filename: str) -> list[str]:
    # Ultralytics hosts ONNX assets in releases; keep a small fallback list.
    return [
        f"https://github.com/ultralytics/assets/releases/latest/download/{filename}",
        f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{filename}",
        f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{filename}",
    ]


def download_file(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Drone-YOLO-Downloader"})
    with urllib.request.urlopen(req, timeout=30) as response, target.open("wb") as out:
        out.write(response.read())


def export_with_ultralytics(model_name: str, output: Path) -> bool:
    """Fallback: download <model>.pt and export to ONNX using ultralytics."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics export unavailable: {exc}")
        return False

    try:
        print(f"Trying Ultralytics export fallback for {model_name}.pt -> ONNX")
        model = YOLO(f"{model_name}.pt")
        exported = model.export(format="onnx")
        exported_path = Path(str(exported)).resolve()
        if not exported_path.exists():
            print("Ultralytics export finished but ONNX file was not found.")
            return False

        output.parent.mkdir(parents=True, exist_ok=True)
        if exported_path != output.resolve():
            if output.exists():
                output.unlink()
            exported_path.replace(output)
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"Exported model to: {output}")
        print(f"Model size: {size_mb:.1f} MB")
        return True
    except Exception as exc:
        print(f"Ultralytics export failed: {exc}")
        return False


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    filename = f"{args.model}.onnx"
    output = Path(args.output) if args.output else project_root / filename
    if not output.is_absolute():
        output = project_root / output

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        print(f"Model already exists at: {output}")
        print("Use --force to re-download.")
        return 0

    urls = candidate_urls(filename)
    errors: list[str] = []

    tmp_path = output.with_suffix(output.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    for url in urls:
        print(f"Trying: {url}")
        try:
            download_file(url, tmp_path)
            size_mb = tmp_path.stat().st_size / (1024 * 1024)
            if size_mb < 1.0:
                raise RuntimeError("Downloaded file is unexpectedly small")
            tmp_path.replace(output)
            print(f"Downloaded model to: {output}")
            print(f"Model size: {size_mb:.1f} MB")
            return 0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            errors.append(f"{url} -> {exc}")
            if tmp_path.exists():
                tmp_path.unlink()

    print("Direct ONNX download failed; attempting Ultralytics export fallback.")
    if export_with_ultralytics(args.model, output):
        return 0

    print("Failed to prepare YOLO ONNX model.")
    for err in errors:
        print(f"  - {err}")
    print("Tip: try another model (e.g. --model yolov8n) or check network access.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
