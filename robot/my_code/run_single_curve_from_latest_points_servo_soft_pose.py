#!/usr/bin/env python3
"""
Soft servo-continuous scan with orientation following the taught poses.

This version keeps the softer speed/spacing tuning, but it DOES NOT fix
roll/pitch/yaw. Orientation is interpolated along the fitted path.
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

# Run at the original taught height (no dry-run lift).
DRY_RUN_LIFT_MM = 0.0

# Soft servo streaming parameters.
SERVO_SPEED_MM_S = 10.0
SERVO_ACC_MM_S2 = 500.0
SERVO_PERIOD_SECONDS = 0.02
PATH_POINT_SPACING_MM = 0.2
APPROACH_SPEED_MM_S = 10.0
DESCENT_SPEED_MM_S = 10.0

POINTS_PER_SEGMENT = 50
START_DELAY_SECONDS = 5

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "single_curve_latest_servo_soft_pose_trajectory.json")

# TEACH_POINTS_PATH = os.path.join(
#     os.path.dirname(__file__),
#     r"generated_set_3\intermediate_02_03.json",
# )

TEACH_POINTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "teach_points_1_1.json",
)


def load_teach_points(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    source_points = payload.get("points")
    if not isinstance(source_points, list) or len(source_points) < 2:
        raise ValueError("points must be a list containing at least 2 teaching points")

    teach_points = []
    for index, item in enumerate(source_points):
        pose = item.get("tcp_pose")
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError("points[{}].tcp_pose must contain exactly 6 values".format(index))
        if not all(isinstance(value, (int, float)) for value in pose):
            raise ValueError("points[{}].tcp_pose contains a non-numeric value".format(index))
        teach_points.append({
            "name": item.get("name", "P{:03d}".format(index)),
            "pose": [float(value) for value in pose],
        })
    return teach_points


def load_teach_tcp_offset(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    offset = payload.get("tcp_offset")
    if not isinstance(offset, list) or len(offset) != 6:
        raise ValueError("tcp_offset must contain exactly 6 values")
    if not all(isinstance(value, (int, float)) for value in offset):
        raise ValueError("tcp_offset contains a non-numeric value")
    return [float(value) for value in offset]


def load_teach_robot_ip(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    robot_ip = payload.get("robot_ip")
    if not isinstance(robot_ip, str) or not robot_ip.strip():
        raise ValueError("robot_ip is missing from the teach-point file")
    return robot_ip.strip()


TEACH_POINTS = load_teach_points(TEACH_POINTS_PATH)
TEACH_TCP_OFFSET = load_teach_tcp_offset(TEACH_POINTS_PATH)
TEACH_ROBOT_IP = load_teach_robot_ip(TEACH_POINTS_PATH)


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

    # Unwrap before interpolation so roll near +/-180 does not jump abruptly.
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


def make_staged_approach_poses(current_pose, trajectory_start_pose):
    """Keep current Z for the lateral/orientation approach, then change only Z."""
    if len(current_pose) < 6 or len(trajectory_start_pose) < 6:
        raise ValueError("current and trajectory-start poses must contain 6 values")
    if current_pose[2] < trajectory_start_pose[2]:
        raise RuntimeError(
            "current TCP Z {:.3f} mm is below trajectory-start Z {:.3f} mm; "
            "refusing staged descent".format(
                current_pose[2], trajectory_start_pose[2]
            )
        )
    above_start = list(trajectory_start_pose[:6])
    above_start[2] = float(current_pose[2])
    return above_start, list(trajectory_start_pose[:6])


def read_current_tcp_pose(arm):
    code, pose = arm.get_position()
    if code != 0 or not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise RuntimeError(
            "failed to read current TCP pose, code={}, pose={}".format(code, pose)
        )
    return [float(value) for value in pose[:6]]


def restore_position_mode_ready(
    arm,
    timeout_seconds=10.0,
    stable_seconds=0.5,
    poll_seconds=0.05,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Re-arm mode 0 after the countdown and require stable READY feedback."""
    mode_code = arm.set_mode(0)
    print("set_mode 0 immediately before approach:", mode_code)
    if mode_code != 0:
        raise RuntimeError(
            "set_mode 0 immediately before approach failed, code={}".format(
                mode_code
            )
        )

    # set_mode can put the controller into state 5 (MODE_CHANGED). This must
    # follow set_mode, even when the requested mode was already mode 0.
    state_code = arm.set_state(0)
    print("set_state 0 immediately before approach:", state_code)
    if state_code != 0:
        raise RuntimeError(
            "set_state 0 immediately before approach failed, code={}".format(
                state_code
            )
        )

    deadline = monotonic() + timeout_seconds
    ready_since = None
    last_state = None
    while monotonic() < deadline:
        state_query_code, state = arm.get_state()
        last_state = {
            "state_query_code": state_query_code,
            "state": state,
            "mode": arm.mode,
            "error_code": arm.error_code,
            "warn_code": arm.warn_code,
        }
        ready = (
            state_query_code == 0
            and state == 2
            and arm.mode == 0
            and arm.error_code == 0
            and arm.warn_code == 0
        )
        if ready:
            if ready_since is None:
                ready_since = monotonic()
            if monotonic() - ready_since >= stable_seconds:
                print("xArm position mode READY (stable):", last_state)
                return last_state
        else:
            ready_since = None
        sleep(poll_seconds)

    raise RuntimeError(
        "xArm did not become stably READY in position mode immediately "
        "before approach; last_status={}".format(last_state)
    )


def save_trajectory(curve, current_pose=None, above_start_pose=None):
    payload = {
        "robot_ip": ROBOT_IP,
        "mode": "single_curve_latest_servo_soft_pose",
        "created_at": time.time(),
        "dry_run_lift_mm": DRY_RUN_LIFT_MM,
        "servo_speed_mm_s": SERVO_SPEED_MM_S,
        "servo_acc_mm_s2": SERVO_ACC_MM_S2,
        "servo_period_seconds": SERVO_PERIOD_SECONDS,
        "path_point_spacing_mm": PATH_POINT_SPACING_MM,
        "orientation_mode": "interpolated_from_teaching_points",
        "tcp_offset": TEACH_TCP_OFFSET,
        "approach_start_pose": current_pose,
        "staged_approach": {
            "constant_z_target_pose": above_start_pose,
            "vertical_descent_target_pose": curve[0],
            "approach_speed_mm_s": APPROACH_SPEED_MM_S,
            "descent_speed_mm_s": DESCENT_SPEED_MM_S,
        },
        "path_length_mm": path_length(curve),
        "teach_points": TEACH_POINTS,
        "trajectory": [{"index": i, "pose": pose} for i, pose in enumerate(curve)],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    curve = build_curve()
    save_trajectory(curve)

    print("Soft pose-following servo trajectory generated")
    print("  trajectory points:", len(curve))
    print("  path length: {:.1f} mm".format(path_length(curve)))
    print("  lift:", DRY_RUN_LIFT_MM, "mm")
    print("  servo speed:", SERVO_SPEED_MM_S, "mm/s")
    print("  servo acceleration:", SERVO_ACC_MM_S2, "mm/s^2")
    print("  servo period:", SERVO_PERIOD_SECONDS, "s")
    print("  point spacing:", PATH_POINT_SPACING_MM, "mm")
    print("  constant-Z approach speed:", APPROACH_SPEED_MM_S, "mm/s")
    print("  vertical descent speed:", DESCENT_SPEED_MM_S, "mm/s")
    print("  orientation: interpolated from taught poses")
    print("  teach points file:", TEACH_POINTS_PATH)
    print("  teach points count:", len(TEACH_POINTS))
    print("  output:", OUTPUT_PATH)
    print()
    print("  start:", [round(v, 3) for v in curve[0]])
    print("  end:  ", [round(v, 3) for v in curve[-1]])
    print()

    arm = XArmAPI(ROBOT_IP)
    try:
        if arm.warn_code or arm.error_code:
            raise RuntimeError(
                "xArm has warn/error before motion (warn={}, error={}); "
                "inspect and clear it deliberately before rerunning.".format(
                    arm.warn_code, arm.error_code
                )
            )

        setup_calls = (
            ("motion_enable", lambda: arm.motion_enable(True)),
            ("set_mode 0", lambda: arm.set_mode(0)),
        )
        for name, operation in setup_calls:
            setup_code = operation()
            print("{}:".format(name), setup_code)
            if setup_code != 0:
                raise RuntimeError(
                    "{} failed, code={}".format(name, setup_code)
                )
        tcp_code = arm.set_tcp_offset(TEACH_TCP_OFFSET, wait=True)
        print("set_tcp_offset from teach points:", tcp_code, TEACH_TCP_OFFSET)
        if tcp_code != 0:
            raise RuntimeError(
                "failed to restore taught TCP offset, code={}".format(tcp_code)
            )

        # set_tcp_offset is a critical configuration change and makes the
        # controller report state 5 (MODE_CHANGED). Restore motion state only
        # after the TCP and mode configuration is complete.
        state_code = arm.set_state(0)
        print("set_state 0 after TCP/mode configuration:", state_code)
        if state_code != 0:
            raise RuntimeError(
                "set_state 0 after TCP/mode configuration failed, code={}"
                .format(state_code)
        )
        time.sleep(0.5)

        print("Robot will move in {} seconds.".format(START_DELAY_SECONDS))
        print("Keep emergency stop reachable. Press Ctrl+C now to cancel.")
        for left in range(START_DELAY_SECONDS, 0, -1):
            print("{}...".format(left))
            time.sleep(1)

        # The controller can return to state 5 during the countdown. Re-arm
        # immediately before reading the actual start pose and issuing the
        # first Cartesian position command.
        restore_position_mode_ready(arm)
        current_pose = read_current_tcp_pose(arm)
        above_start, start = make_staged_approach_poses(current_pose, curve[0])
        save_trajectory(curve, current_pose, above_start)
        print("Actual approach-start TCP pose:", current_pose)

        print("Stage 1: moving above trajectory start while keeping current Z...")
        code = arm.set_position(
            x=above_start[0],
            y=above_start[1],
            z=above_start[2],
            roll=above_start[3],
            pitch=above_start[4],
            yaw=above_start[5],
            speed=APPROACH_SPEED_MM_S,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(
                "constant-Z approach failed, code={}".format(code)
            )

        print("Stage 2: descending vertically to trajectory start...")
        code = arm.set_position(
            x=start[0],
            y=start[1],
            z=start[2],
            roll=start[3],
            pitch=start[4],
            yaw=start[5],
            speed=DESCENT_SPEED_MM_S,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(
                "vertical descent to start failed, code={}".format(code)
            )

        print("Switching to servo Cartesian mode...")
        mode_code = arm.set_mode(1)
        state_code = arm.set_state(0)
        print("set_mode 1:", mode_code)
        print("set_state 0:", state_code)
        if mode_code != 0 or state_code != 0:
            raise RuntimeError(
                "failed to enter servo mode: mode_code={}, state_code={}".format(
                    mode_code, state_code
                )
            )
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
        print("Soft pose-following servo run finished.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
