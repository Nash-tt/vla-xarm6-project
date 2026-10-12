#!/usr/bin/env python3
"""Collect synchronized Intel RealSense RGB-D episodes for VLA datasets."""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def device_info(device: rs.device) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "product_line": rs.camera_info.product_line,
        "usb_type": rs.camera_info.usb_type_descriptor,
    }
    result: dict[str, str] = {}
    for key, field in fields.items():
        if device.supports(field):
            result[key] = device.get_info(field)
    return result


def intrinsics_dict(profile: rs.video_stream_profile) -> dict[str, Any]:
    intr = profile.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def read_action(path: Path | None) -> Any:
    """Read the latest robot action/state JSON without stopping camera capture."""
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def unique_episode_dir(root: Path, requested_id: str | None) -> Path:
    stem = requested_id or datetime.now().strftime("episode_%Y%m%d_%H%M%S")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect aligned RGB and depth frames from an Intel RealSense camera."
    )
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--episode-id", help="Episode directory name; generated if omitted")
    parser.add_argument("--instruction", default="", help="Language instruction for this episode")
    parser.add_argument("--serial", help="RealSense serial number; first device is used if omitted")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 means until Q/Ctrl+C")
    parser.add_argument("--warmup", type=int, default=30, help="Frames discarded before recording")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--action-file", type=Path, help="Optional JSON file polled once per frame")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV preview window")
    parser.add_argument("--color-only", action="store_true", help="Do not save depth images")
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height and fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("jpeg-quality must be between 1 and 100")

    context = rs.context()
    devices = context.query_devices()
    if args.list_devices:
        print(json.dumps([device_info(d) for d in devices], ensure_ascii=False, indent=2))
        return 0
    if not devices:
        raise SystemExit("No RealSense camera found. Check USB cable, driver, and camera permissions.")

    selected = None
    if args.serial:
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == args.serial:
                selected = dev
                break
        if selected is None:
            raise SystemExit(f"RealSense serial {args.serial!r} was not found")
    else:
        selected = devices[0]
    serial = selected.get_info(rs.camera_info.serial_number)

    episode_dir = unique_episode_dir(args.output.resolve(), args.episode_id)
    color_dir = episode_dir / "color"
    depth_dir = episode_dir / "depth"
    color_dir.mkdir(parents=True)
    if not args.color_only:
        depth_dir.mkdir()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.color_only:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    profile = None
    frames_file = None
    captured = 0
    dropped = 0
    started_monotonic_ns = 0
    started_utc = utc_now()

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color) if not args.color_only else None
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_scale = None
        depth_intrinsics = None
        if not args.color_only:
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intrinsics = intrinsics_dict(depth_profile)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        metadata = {
            "schema_version": "1.0",
            "episode_id": episode_dir.name,
            "instruction": args.instruction,
            "created_utc": started_utc,
            "device": device_info(profile.get_device()),
            "capture": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "color_format": "BGR8 saved as JPEG",
                "depth_format": None if args.color_only else "Z16 saved as uint16 PNG",
                "depth_scale_meters_per_unit": depth_scale,
                "depth_aligned_to_color": not args.color_only,
                "color_intrinsics": intrinsics_dict(color_profile),
                "native_depth_intrinsics": depth_intrinsics,
            },
            "host": {"python": sys.version, "platform": platform.platform()},
        }
        (episode_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"Warming up camera for {args.warmup} frames...")
        for _ in range(args.warmup):
            if stop_requested:
                break
            pipeline.wait_for_frames(5000)

        frames_file = (episode_dir / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        started_monotonic_ns = time.monotonic_ns()
        print(f"Recording to: {episode_dir}")
        if not args.no_preview:
            print("Press Q or Esc in the preview window to stop.")

        while not stop_requested:
            if args.duration > 0 and (time.monotonic_ns() - started_monotonic_ns) / 1e9 >= args.duration:
                break
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                dropped += 1
                print(f"Frame timeout: {exc}", file=sys.stderr)
                continue
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = None if args.color_only else frames.get_depth_frame()
            if not color_frame or (not args.color_only and not depth_frame):
                dropped += 1
                continue

            host_monotonic_ns = time.monotonic_ns()
            host_unix_ns = time.time_ns()
            color = np.asanyarray(color_frame.get_data())
            color_name = f"{captured:06d}.jpg"
            color_ok = cv2.imwrite(
                str(color_dir / color_name), color, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if not color_ok:
                raise RuntimeError(f"Failed to write {color_dir / color_name}")

            depth_name = None
            depth = None
            if depth_frame is not None:
                depth = np.asanyarray(depth_frame.get_data())
                depth_name = f"{captured:06d}.png"
                if not cv2.imwrite(str(depth_dir / depth_name), depth):
                    raise RuntimeError(f"Failed to write {depth_dir / depth_name}")

            record = {
                "frame_index": captured,
                "host_unix_ns": host_unix_ns,
                "host_monotonic_ns": host_monotonic_ns,
                "elapsed_s": (host_monotonic_ns - started_monotonic_ns) / 1e9,
                "device_timestamp_ms": color_frame.get_timestamp(),
                "device_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                "device_frame_number": color_frame.get_frame_number(),
                "color_path": f"color/{color_name}",
                "depth_path": None if depth_name is None else f"depth/{depth_name}",
                "instruction": args.instruction,
                "action": read_action(args.action_file),
            }
            frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            captured += 1

            if not args.no_preview:
                preview = color
                if depth is not None:
                    depth_preview = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_TURBO
                    )
                    preview = np.hstack((color, depth_preview))
                cv2.putText(
                    preview,
                    f"frames={captured}  Q/Esc: stop",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("RealSense VLA Collector", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture_ended_monotonic_ns = time.monotonic_ns()
        if frames_file is not None:
            frames_file.close()
        if profile is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        elapsed = (
            0.0
            if not started_monotonic_ns
            else (capture_ended_monotonic_ns - started_monotonic_ns) / 1e9
        )
        summary = {
            "completed_utc": utc_now(),
            "frame_count": captured,
            "dropped_or_timeout_count": dropped,
            "elapsed_s": elapsed,
            "average_fps": captured / elapsed if elapsed > 0 else 0.0,
        }
        (episode_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
