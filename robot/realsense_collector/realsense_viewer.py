#!/usr/bin/env python3
"""Display live Intel RealSense frames without recording any data."""

from __future__ import annotations

import argparse
import signal
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live RealSense color and depth frames without saving them."
    )
    parser.add_argument("--serial", help="RealSense serial number; first device is used if omitted")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-only", action="store_true", help="Display only the color stream")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height and fps must be positive")

    devices = rs.context().query_devices()
    if not devices:
        raise SystemExit("No RealSense camera found. Check USB cable and camera permissions.")

    if args.serial:
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        if args.serial not in serials:
            raise SystemExit(f"RealSense serial {args.serial!r} was not found")
        serial = args.serial
    else:
        serial = devices[0].get_info(rs.camera_info.serial_number)

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
    started = False

    try:
        pipeline.start(config)
        started = True
        align = None if args.color_only else rs.align(rs.stream.color)
        print(f"Showing RealSense {serial}. Press Q or Esc to close.")

        while not stop_requested:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                print(f"Frame timeout: {exc}")
                continue

            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = None if args.color_only else frames.get_depth_frame()
            if not color_frame or (not args.color_only and not depth_frame):
                continue

            color = np.asanyarray(color_frame.get_data())
            display = color
            if depth_frame is not None:
                depth = np.asanyarray(depth_frame.get_data())
                depth_color = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_TURBO
                )
                display = np.hstack((color, depth_color))

            cv2.imshow("RealSense Live Viewer", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
