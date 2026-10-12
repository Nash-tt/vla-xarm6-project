#!/usr/bin/env python3
"""
Softer servo-continuous scan from the latest 9 teaching points.

This version is tuned to reduce small end-effector shake:
  - lower speed
  - lower acceleration
  - point spacing matched to speed * servo period
  - optional fixed orientation to avoid wrist micro-adjustments
"""

import json
import math
import os
import sys
import time


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402


ROBOT_IP = "192.168.1.100"

# First run should be raised. Set to 0.0 only after checking the path is safe.
DRY_RUN_LIFT_MM = 30.0

# Softer servo streaming parameters.
SERVO_SPEED_MM_S = 10.0
SERVO_ACC_MM_S2 = 500.0
SERVO_PERIOD_SECONDS = 0.02

# Match spacing to speed * period: 10 mm/s * 0.02 s = 0.20 mm.
PATH_POINT_SPACING_MM = 0.2

POINTS_PER_SEGMENT = 50
START_DELAY_SECONDS = 5

# If True, use the first point orientation for the whole curve.
# This often reduces wrist shake when teaching orientations are not reliable.
USE_FIXED_ORIENTATION = True

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "single_curve_latest_servo_soft_trajectory.json")


TEACH_POINTS = [
    {"name": "P00", "pose": [208.458328, -126.490364, 300.37561, 178.948184, 0.167934, 1.606975]},
    {"name": "P10", "pose": [205.94548, -54.406677, 313.332001, 174.266087, 0.635238, 1.245037]},
    {"name": "P20", "pose": [205.98436, -8.69558, 324.340576, 176.26571, 0.591407, 1.266409]},
    {"name": "P01", "pose": [205.98436, 42.09827, 337.162079, 175.128561, 0.616388, 1.254434]},
    {"name": "P11", "pose": [205.984436, 149.523331, 352.639526, 171.022688, 0.704681, 1.207107]},
    {"name": "P21", "pose": [205.984543, 204.788605, 352.036713, 166.531959, 0.797042, 1.148207]},
    {"name": "P02", "pose": [205.984589, 244.543594, 357.876129, 163.437643, 0.85789, 1.103517]},
    {"name": "P12", "pose": [205.984589, 321.566925, 356.515686, 158.946571, 0.941656, 1.032928]},
    {"name": "P22", "pose": [205.984589, 381.667084, 347.52121, 153.978397, 1.027543, 0.9475]},
]


def unwrap_degrees(values):
    result = [values[0]]
    for value in values[1:]:
        prev = result[-1]
        while value - prev > 180.0:
            value -= 360.0
        while value - prev < -180.0:
            value += 360.0
        result.append(value)
    return result


def wrap_degree(value):
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def pose_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def lerp_pose(a, b, ratio):
    pose = [a[i] + (b[i] - a[i]) * ratio for i in range(6)]
    pose[3] = wrap_degree(pose[3])
    pose[4] = wrap_degree(pose[4])
    pose[5] = wrap_degree(pose[5])
    return pose


def make_raw_curve(points, points_per_segment, lift_mm):
    poses = [item["pose"][:] for item in points]

    for axis in [3, 4, 5]:
        unwrapped = unwrap_degrees([pose[axis] for pose in poses])
        for pose, value in zip(poses, unwrapped):
            pose[axis] = value

    curve = []
    n = len(poses)
    for segment in range(n - 1):
        p0 = poses[max(segment - 1, 0)]
        p1 = poses[segment]
        p2 = poses[segment + 1]
        p3 = poses[min(segment + 2, n - 1)]
        for step in range(points_per_segment):
            if segment > 0 and step == 0:
                continue
            t = step / float(points_per_segment)
            pose = [catmull_rom(p0[i], p1[i], p2[i], p3[i], t) for i in range(6)]
            pose[2] += lift_mm
            pose[3] = wrap_degree(pose[3])
            pose[4] = wrap_degree(pose[4])
            pose[5] = wrap_degree(pose[5])
            curve.append(pose)

    final_pose = poses[-1][:]
    final_pose[2] += lift_mm
    final_pose[3] = wrap_degree(final_pose[3])
    final_pose[4] = wrap_degree(final_pose[4])
    final_pose[5] = wrap_degree(final_pose[5])
    curve.append(final_pose)

    if USE_FIXED_ORIENTATION and curve:
        fixed = curve[0][3:6]
        for pose in curve:
            pose[3:6] = fixed[:]

    return curve


def resample_by_distance(raw_curve, spacing_mm):
    if len(raw_curve) <= 1:
        return raw_curve[:]

    result = [raw_curve[0]]
    carried = 0.0
    prev = raw_curve[0]

    for current in raw_curve[1:]:
        seg_len = pose_distance(prev, current)
        if seg_len < 1e-9:
            prev = current
            continue

        while carried + seg_len >= spacing_mm:
            ratio = (spacing_mm - carried) / seg_len
            new_pose = lerp_pose(prev, current, ratio)
            result.append(new_pose)
            prev = new_pose
            seg_len = pose_distance(prev, current)
            carried = 0.0

        carried += seg_len
        prev = current

    if pose_distance(result[-1], raw_curve[-1]) > 1e-6:
        result.append(raw_curve[-1])
    return result


def path_length(curve):
    return sum(pose_distance(curve[i - 1], curve[i]) for i in range(1, len(curve)))


def build_curve():
    raw_curve = make_raw_curve(TEACH_POINTS, POINTS_PER_SEGMENT, DRY_RUN_LIFT_MM)
    return resample_by_distance(raw_curve, PATH_POINT_SPACING_MM)


def save_trajectory(curve):
    payload = {
        "robot_ip": ROBOT_IP,
        "mode": "single_curve_latest_servo_soft",
        "created_at": time.time(),
        "dry_run_lift_mm": DRY_RUN_LIFT_MM,
        "servo_speed_mm_s": SERVO_SPEED_MM_S,
        "servo_acc_mm_s2": SERVO_ACC_MM_S2,
        "servo_period_seconds": SERVO_PERIOD_SECONDS,
        "path_point_spacing_mm": PATH_POINT_SPACING_MM,
        "use_fixed_orientation": USE_FIXED_ORIENTATION,
        "path_length_mm": path_length(curve),
        "teach_points": TEACH_POINTS,
        "trajectory": [{"index": i, "pose": pose} for i, pose in enumerate(curve)],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    curve = build_curve()
    save_trajectory(curve)

    print("Soft servo-continuous trajectory generated")
    print("  trajectory points:", len(curve))
    print("  path length: {:.1f} mm".format(path_length(curve)))
    print("  lift:", DRY_RUN_LIFT_MM, "mm")
    print("  servo speed:", SERVO_SPEED_MM_S, "mm/s")
    print("  servo acceleration:", SERVO_ACC_MM_S2, "mm/s^2")
    print("  servo period:", SERVO_PERIOD_SECONDS, "s")
    print("  point spacing:", PATH_POINT_SPACING_MM, "mm")
    print("  fixed orientation:", USE_FIXED_ORIENTATION)
    print("  output:", OUTPUT_PATH)
    print()

    print("Robot will move in {} seconds.".format(START_DELAY_SECONDS))
    print("Keep emergency stop reachable. Press Ctrl+C now to cancel.")
    for left in range(START_DELAY_SECONDS, 0, -1):
        print("{}...".format(left))
        time.sleep(1)

    arm = XArmAPI(ROBOT_IP)
    try:
        if arm.warn_code != 0:
            print("clean_warn:", arm.clean_warn())
        if arm.error_code != 0:
            print("clean_error:", arm.clean_error())

        print("motion_enable:", arm.motion_enable(True))
        print("set_mode 0:", arm.set_mode(0))
        print("set_state 0:", arm.set_state(0))
        time.sleep(0.5)

        print("Moving to start pose...")
        start = curve[0]
        code = arm.set_position(
            x=start[0],
            y=start[1],
            z=start[2],
            roll=start[3],
            pitch=start[4],
            yaw=start[5],
            speed=10,
            wait=True,
        )
        if code != 0:
            raise RuntimeError("move to start failed, code={}".format(code))

        print("Switching to servo Cartesian mode...")
        print("set_mode 1:", arm.set_mode(1))
        print("set_state 0:", arm.set_state(0))
        time.sleep(0.5)

        next_time = time.monotonic()
        for index, pose in enumerate(curve):
            ret = arm.set_servo_cartesian(
                pose,
                speed=SERVO_SPEED_MM_S,
                mvacc=SERVO_ACC_MM_S2,
            )
            if ret != 0:
                raise RuntimeError("set_servo_cartesian failed at index {}, code={}".format(index, ret))

            if index % 100 == 0 or index == len(curve) - 1:
                print("servo #{:04d}/{}".format(index, len(curve) - 1))

            next_time += SERVO_PERIOD_SECONDS
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Returning to position mode...")
        print("set_mode 0:", arm.set_mode(0))
        print("set_state 0:", arm.set_state(0))
        print("Soft servo-continuous run finished.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
