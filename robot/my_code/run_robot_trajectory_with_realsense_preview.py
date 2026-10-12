#!/usr/bin/env python3
"""Run the fitted xArm trajectory with a non-recording RealSense RGB preview.

This is a robot-only safety-test entry point. It does not launch PhaseLink and
does not save RealSense frames. The preview has its own thread and never waits
inside the 20 ms Cartesian servo loop.
"""

import argparse
import os
import sys
import threading
import time


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402
from run_fitted_trajectory_vla_sync import prepare_robot_at_start  # noqa: E402
from run_single_curve_from_latest_points_servo_soft_pose import (  # noqa: E402
    DRY_RUN_LIFT_MM,
    PATH_POINT_SPACING_MM,
    ROBOT_IP,
    SERVO_ACC_MM_S2,
    SERVO_PERIOD_SECONDS,
    SERVO_SPEED_MM_S,
    START_DELAY_SECONDS,
    TEACH_POINTS,
    TEACH_POINTS_PATH,
    TEACH_ROBOT_IP,
    TEACH_TCP_OFFSET,
    build_curve,
    path_length,
)


class RealSenseRgbPreview(threading.Thread):
    """Display latest RGB frames without recording or blocking robot motion."""

    def __init__(self, stop_event, window_name="RealSense live RGB preview"):
        super().__init__(name="realsense-rgb-preview", daemon=True)
        self.stop_event = stop_event
        self.window_name = window_name
        self.started_event = threading.Event()
        self.closed_event = threading.Event()
        self.error = None
        self.closed_by_user = False
        self.frame_count = 0

    def run(self):
        pipeline = None
        cv2 = None
        try:
            import cv2 as cv2_module
            import numpy as np
            import pyrealsense2 as rs

            cv2 = cv2_module
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 960, 540)
            self.started_event.set()

            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                display = image.copy()
                cv2.putText(
                    display,
                    "Preview only - Q/ESC closes camera view; robot continues",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, display)
                self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                try:
                    window_visible = cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                except Exception:
                    window_visible = 0
                if key in (ord("q"), ord("Q"), 27) or window_visible < 1:
                    self.closed_by_user = True
                    print(
                        "RealSense preview closed; robot test continues without "
                        "camera display."
                    )
                    break
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
        finally:
            self.started_event.set()
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            if cv2 is not None:
                try:
                    cv2.destroyWindow(self.window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass
            self.closed_event.set()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 30 mm raised fitted robot trajectory while showing a "
            "non-recording RealSense RGB preview."
        )
    )
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--allow-robot-ip-mismatch", action="store_true")
    parser.add_argument(
        "--start-only",
        action="store_true",
        help="Move only to the raised fitted-trajectory start pose, then stop.",
    )
    return parser.parse_args()


def run_servo_trajectory(arm, curve):
    next_deadline_ns = time.monotonic_ns()
    period_ns = int(round(SERVO_PERIOD_SECONDS * 1e9))

    for index, pose in enumerate(curve):
        code = arm.set_servo_cartesian(
            pose,
            speed=SERVO_SPEED_MM_S,
            mvacc=SERVO_ACC_MM_S2,
        )
        if code != 0:
            raise RuntimeError(
                "set_servo_cartesian failed at index {}, code={}".format(
                    index, code
                )
            )
        if index % 100 == 0 or index == len(curve) - 1:
            print("servo #{:04d}/{}".format(index, len(curve) - 1))

        next_deadline_ns += period_ns
        remaining_ns = next_deadline_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)


def main():
    args = parse_args()
    if args.robot_ip != TEACH_ROBOT_IP and not args.allow_robot_ip_mismatch:
        raise ValueError(
            "--robot-ip {} differs from teach-point robot_ip {}. Refusing "
            "motion; verify the robot and pass --allow-robot-ip-mismatch only "
            "if intentional.".format(args.robot_ip, TEACH_ROBOT_IP)
        )

    curve = build_curve()
    print("Robot trajectory test with RealSense preview")
    print("  robot IP:", args.robot_ip)
    print("  teach points file:", TEACH_POINTS_PATH)
    print("  teach points:", len(TEACH_POINTS))
    print("  trajectory points:", len(curve))
    print("  path length: {:.1f} mm".format(path_length(curve)))
    print("  lift: {:.1f} mm".format(DRY_RUN_LIFT_MM))
    print("  servo speed: {:.1f} mm/s".format(SERVO_SPEED_MM_S))
    print("  servo acceleration: {:.1f} mm/s^2".format(SERVO_ACC_MM_S2))
    print("  servo period: {:.3f} s".format(SERVO_PERIOD_SECONDS))
    print("  point spacing: {:.3f} mm".format(PATH_POINT_SPACING_MM))
    print("  TCP offset:", TEACH_TCP_OFFSET)
    print("  mode:", "start-only" if args.start_only else "complete trajectory")

    preview_stop = threading.Event()
    preview = RealSenseRgbPreview(preview_stop)
    arm = None
    try:
        print("Starting non-recording RealSense RGB preview...")
        preview.start()
        preview.started_event.wait(timeout=20)
        if preview.error is not None:
            raise RuntimeError("RealSense preview failed: {}".format(preview.error))
        if not preview.started_event.is_set():
            raise RuntimeError("RealSense preview startup timed out")

        print("Connecting xArm...")
        arm = XArmAPI(
            args.robot_ip,
            enable_report=True,
            report_type="real",
        )
        time.sleep(1.0)
        if not arm.connected:
            raise RuntimeError("failed to connect xArm at {}".format(args.robot_ip))

        print("Robot will move in {} seconds.".format(START_DELAY_SECONDS))
        print("Keep the emergency stop reachable. Press Ctrl+C now to cancel.")
        for left in range(START_DELAY_SECONDS, 0, -1):
            print("{}...".format(left))
            time.sleep(1)

        prepare_robot_at_start(arm, curve, TEACH_TCP_OFFSET)
        if args.start_only:
            print("Raised start pose reached; --start-only test completed.")
        else:
            print("Running complete fitted Cartesian servo trajectory...")
            run_servo_trajectory(arm, curve)
            print("Complete raised trajectory finished.")
    except KeyboardInterrupt:
        print("\nInterrupted; returning control mode safely.")
    finally:
        if arm is not None:
            try:
                print("Returning xArm to position mode...")
                arm.set_mode(0)
                arm.set_state(0)
            finally:
                arm.disconnect()

        preview_stop.set()
        if preview.is_alive():
            preview.join(timeout=10)
        if preview.is_alive():
            print("WARNING: RealSense preview thread did not stop in time.")
        if preview.error is not None:
            print("RealSense preview error:", preview.error)
        print("RealSense preview frames shown:", preview.frame_count)


if __name__ == "__main__":
    main()
