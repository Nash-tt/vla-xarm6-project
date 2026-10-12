#!/usr/bin/env python3
"""
Collect any number of teaching points.

Usage:
  1. Run this script.
  2. Move the robot by hand in teach mode.
  3. Press Enter to save the current TCP pose.
  4. Type q then Enter to finish.

Output:
  my_code/teach_points_any_count.json
"""

import json
import os
import sys
import time


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402


ROBOT_IP = "192.168.1.100"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "teach_points_3_3.json")

# No probe installed: flange TCP.
# After mounting the probe, change this to the probe contact-point offset.
TCP_OFFSET = [0, 0, 0, 0, 0, 0]

TEACH_SENSITIVITY = 5


def save_points(points):
    payload = {
        "robot_ip": ROBOT_IP,
        "created_at": time.time(),
        "tcp_offset": TCP_OFFSET,
        "count": len(points),
        "unit": {
            "position": "mm",
            "orientation": "degree",
        },
        "points": points,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_status(arm, title):
    print()
    print("==== {} ====".format(title))
    print("connected:", arm.connected)
    print("mode:", arm.mode)
    print("state:", arm.state)
    print("error_code:", arm.error_code)
    print("warn_code:", arm.warn_code)
    print("====================")
    print()


def make_point_name(index):
    return "P{:03d}".format(index)


def main():
    print("Any-count teaching point collector")
    print("Robot IP:", ROBOT_IP)
    print("Output:", OUTPUT_FILE)
    print("Keep emergency stop reachable before running.")
    input("Press Enter to connect and enter teach mode...")

    arm = XArmAPI(ROBOT_IP)
    points = []

    try:
        time.sleep(0.5)
        print_status(arm, "after connect")

        if arm.warn_code != 0:
            print("clean_warn:", arm.clean_warn())
        if arm.error_code != 0:
            print("clean_error:", arm.clean_error())

        print("motion_enable:", arm.motion_enable(True))
        print("set_mode 0:", arm.set_mode(0))
        print("set_state 0:", arm.set_state(0))
        time.sleep(0.5)

        print("set_tcp_offset:", arm.set_tcp_offset(TCP_OFFSET, wait=True))

        print("set_mode 2:", arm.set_mode(2))
        print("set_state 0:", arm.set_state(0))
        print("set_teach_sensitivity:", arm.set_teach_sensitivity(TEACH_SENSITIVITY))
        time.sleep(1.0)
        print_status(arm, "teach mode")

        print("Move the robot by hand.")
        print("Press Enter to save a point.")
        print("Type q then Enter to finish.")

        while True:
            default_name = make_point_name(len(points))
            answer = input("\nMove to {}, press Enter to save, or q to quit: ".format(default_name)).strip()
            if answer.lower() in ("q", "quit", "exit"):
                break

            name = answer if answer else default_name
            code, pose = arm.get_position(is_radian=False)
            code_j, joints = arm.get_servo_angle(is_radian=False)

            if code != 0:
                print("{} save failed, get_position code={}".format(name, code))
                print_status(arm, "save failed")
                continue

            point = {
                "name": name,
                "index": len(points),
                "timestamp": time.time(),
                "tcp_pose": pose,
                "joint_angles": joints if code_j == 0 else None,
            }
            points.append(point)
            save_points(points)

            print("{} saved".format(name))
            print("  tcp_pose:", pose)
            print("  total:", len(points))

        print("Returning to position mode...")
        print("set_mode 0:", arm.set_mode(0))
        print("set_state 0:", arm.set_state(0))

    finally:
        try:
            arm.disconnect()
        finally:
            save_points(points)

    print()
    print("Done. Saved {} point(s) to:".format(len(points)))
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
