#!/usr/bin/env python3
"""
Recollect every teaching point after a selected point.

The selected point and all points before it are kept unchanged. Newly captured
points replace the old suffix only after an explicit final confirmation. The
original JSON is backed up immediately before the replacement.

Examples:
  python recollect_teach_points_after.py
  python recollect_teach_points_after.py --after P010
  python recollect_teach_points_after.py --list
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402


DEFAULT_POINTS_FILE = os.path.join(
    os.path.dirname(__file__), "teach_points_any_count_1.json"
)
ANCHOR_MOVE_SPEED_MM_S = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recollect and replace all teaching points after one point."
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_POINTS_FILE,
        help="teaching-point JSON file (default: teach_points_any_count.json)",
    )
    parser.add_argument(
        "--after",
        help="last point to keep, specified by point name, list position, or index",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show the existing points and exit without connecting to the robot",
    )
    return parser.parse_args()


def load_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("points must be a non-empty list")

    for position, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError("points[{}] must be an object".format(position))
        pose = point.get("tcp_pose")
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError(
                "points[{}].tcp_pose must contain exactly 6 values".format(position)
            )
        if not all(isinstance(value, (int, float)) for value in pose):
            raise ValueError(
                "points[{}].tcp_pose contains a non-numeric value".format(position)
            )

    robot_ip = payload.get("robot_ip")
    if not isinstance(robot_ip, str) or not robot_ip.strip():
        raise ValueError("robot_ip is missing from the teaching-point file")

    tcp_offset = payload.get("tcp_offset")
    if not isinstance(tcp_offset, list) or len(tcp_offset) != 6:
        raise ValueError("tcp_offset must contain exactly 6 values")
    if not all(isinstance(value, (int, float)) for value in tcp_offset):
        raise ValueError("tcp_offset contains a non-numeric value")

    return payload


def point_name(point, position):
    return str(point.get("name", "P{:03d}".format(position)))


def print_points(points):
    print()
    print("Existing teaching points:")
    print("  pos  stored-index  name       x        y        z       roll    pitch      yaw")
    for position, point in enumerate(points):
        pose = point["tcp_pose"]
        print(
            "  {:>3}  {:>12}  {:<8} {:>8.2f} {:>8.2f} {:>8.2f} "
            "{:>8.2f} {:>8.2f} {:>8.2f}".format(
                position,
                point.get("index", "-"),
                point_name(point, position),
                *pose
            )
        )
    print()


def resolve_point(points, selector):
    selector = selector.strip()
    if not selector:
        raise ValueError("point selector cannot be empty")

    name_matches = [
        position
        for position, point in enumerate(points)
        if point_name(point, position).lower() == selector.lower()
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        raise ValueError("point name {!r} is not unique".format(selector))

    try:
        number = int(selector)
    except ValueError:
        raise ValueError("no point named or numbered {!r}".format(selector))

    if 0 <= number < len(points):
        return number

    index_matches = [
        position
        for position, point in enumerate(points)
        if point.get("index") == number
    ]
    if len(index_matches) == 1:
        return index_matches[0]
    raise ValueError("point number {} was not found".format(number))


def choose_keep_position(points, requested):
    selector = requested
    while True:
        if selector is None:
            selector = input(
                "Enter the last point to KEEP (for example P010 or 10), or q to cancel: "
            ).strip()
        if selector.lower() in ("q", "quit", "exit", "cancel"):
            return None
        try:
            return resolve_point(points, selector)
        except ValueError as exc:
            print("Invalid selection:", exc)
            if requested is not None:
                raise
            selector = None


def check_code(label, code):
    print("{}: {}".format(label, code))
    if code != 0:
        raise RuntimeError("{} failed, code={}".format(label, code))


def print_robot_status(arm, title):
    print()
    print("==== {} ====".format(title))
    print("connected:", arm.connected)
    print("mode:", arm.mode)
    print("state:", arm.state)
    print("error_code:", arm.error_code)
    print("warn_code:", arm.warn_code)
    print("====================")
    print()


def default_new_name(old_suffix, new_offset, absolute_position):
    if new_offset < len(old_suffix):
        return point_name(old_suffix[new_offset], absolute_position)
    return "P{:03d}".format(absolute_position)


def show_candidate(prefix, new_points):
    print()
    print("Candidate trajectory: {} kept + {} new = {} points".format(
        len(prefix), len(new_points), len(prefix) + len(new_points)
    ))
    for point in new_points:
        pose = point["tcp_pose"]
        print(
            "  NEW {:>3} {:<8} [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(
                point["index"], point["name"], *pose
            )
        )
    print()


def collect_suffix(arm, prefix, old_suffix):
    new_points = []
    prefix_names = {point_name(point, i).lower() for i, point in enumerate(prefix)}

    print("The robot remains in position mode; hand-guiding is disabled.")
    print("Move the robot to each NEW point with xArm Studio or another jog control.")
    print("Commands: Enter=save, u=undo new point, l=list candidate, q=finish, c=cancel")

    while True:
        absolute_position = len(prefix) + len(new_points)
        default_name = default_new_name(
            old_suffix, len(new_points), absolute_position
        )
        answer = input(
            "\nNext {}: Enter to save, type a custom name, or command [u/l/q/c]: ".format(
                default_name
            )
        ).strip()
        command = answer.lower()

        if command in ("c", "cancel"):
            return None
        if command in ("q", "quit", "finish"):
            return new_points
        if command in ("u", "undo"):
            if new_points:
                removed = new_points.pop()
                print("Removed newly captured point:", removed["name"])
            else:
                print("There is no newly captured point to undo.")
            continue
        if command in ("l", "list"):
            show_candidate(prefix, new_points)
            continue

        name = answer if answer else default_name
        used_names = prefix_names | {point["name"].lower() for point in new_points}
        if name.lower() in used_names:
            print("Point name {!r} is already used; choose another name.".format(name))
            continue

        code_pose, pose = arm.get_position(is_radian=False)
        code_joints, joints = arm.get_servo_angle(is_radian=False)
        if code_pose != 0:
            print("Save failed: get_position code={}".format(code_pose))
            print_robot_status(arm, "save failed")
            continue

        point = {
            "name": name,
            "index": absolute_position,
            "timestamp": time.time(),
            "tcp_pose": [float(value) for value in pose],
            "joint_angles": (
                [float(value) for value in joints]
                if code_joints == 0 and joints is not None
                else None
            ),
        }
        new_points.append(point)
        print("{} saved at index {}".format(name, absolute_position))
        print("  tcp_pose:", point["tcp_pose"])
        print("  new points captured:", len(new_points))


def unique_backup_path(path):
    directory = os.path.dirname(os.path.abspath(path))
    stem, extension = os.path.splitext(os.path.basename(path))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(
        directory, "{}.backup_{}{}".format(stem, timestamp, extension)
    )
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(
            directory,
            "{}.backup_{}_{}{}".format(stem, timestamp, counter, extension),
        )
        counter += 1
    return candidate


def atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".teach_points_",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def build_updated_payload(payload, keep_position, new_points, backup_path):
    old_points = payload["points"]
    prefix = old_points[: keep_position + 1]
    updated = dict(payload)
    updated["points"] = prefix + new_points
    updated["count"] = len(updated["points"])
    updated["modified_at"] = time.time()

    history = payload.get("edit_history", [])
    if not isinstance(history, list):
        history = []
    updated["edit_history"] = history + [
        {
            "operation": "recollect_points_after",
            "modified_at": updated["modified_at"],
            "kept_through_position": keep_position,
            "kept_through_name": point_name(old_points[keep_position], keep_position),
            "old_suffix_count": len(old_points) - len(prefix),
            "new_suffix_count": len(new_points),
            "backup_file": os.path.basename(backup_path),
        }
    ]
    return updated


def main():
    args = parse_args()
    points_path = os.path.abspath(args.file)
    payload = load_payload(points_path)
    points = payload["points"]
    print("Teaching-point file:", points_path)
    print("Robot IP:", payload["robot_ip"])
    print("TCP offset:", payload["tcp_offset"])
    print_points(points)

    if args.list:
        return 0

    keep_position = choose_keep_position(points, args.after)
    if keep_position is None:
        print("Cancelled; the teaching-point file was not changed.")
        return 0

    prefix = points[: keep_position + 1]
    old_suffix = points[keep_position + 1 :]
    anchor = points[keep_position]
    print("Keeping positions 0 through {} (ending at {}).".format(
        keep_position, point_name(anchor, keep_position)
    ))
    print("The existing {} point(s) after it will be replaced.".format(len(old_suffix)))
    print("Kept anchor pose:", anchor["tcp_pose"])
    print(
        "After confirmation, the robot will move to the kept anchor at {:.1f} mm/s."
        .format(ANCHOR_MOVE_SPEED_MM_S)
    )
    print("It will remain in position mode; hand-guiding will NOT be enabled.")
    confirm = input(
        "Type YES to connect in position mode and move to the kept anchor: "
    ).strip()
    if confirm.upper() != "YES":
        print("Cancelled; the teaching-point file was not changed.")
        return 0

    arm = None
    new_points = None
    try:
        arm = XArmAPI(payload["robot_ip"].strip())
        time.sleep(0.5)
        print_robot_status(arm, "after connect")
        if arm.warn_code or arm.error_code:
            raise RuntimeError(
                "xArm has warn/error (warn={}, error={}); resolve it before retrying"
                .format(arm.warn_code, arm.error_code)
            )

        check_code("motion_enable", arm.motion_enable(True))
        check_code("set_mode 0", arm.set_mode(0))
        check_code("set_state 0", arm.set_state(0))
        check_code(
            "set_tcp_offset",
            arm.set_tcp_offset(payload["tcp_offset"], wait=True),
        )
        # Changing the TCP offset can put the controller into state 5. Restore
        # the ready state before commanding the anchor move.
        check_code("set_state 0 after TCP configuration", arm.set_state(0))
        time.sleep(0.5)

        anchor_pose = anchor["tcp_pose"]
        print("Moving to kept anchor {}...".format(
            point_name(anchor, keep_position)
        ))
        check_code(
            "move to kept anchor",
            arm.set_position(
                x=anchor_pose[0],
                y=anchor_pose[1],
                z=anchor_pose[2],
                roll=anchor_pose[3],
                pitch=anchor_pose[4],
                yaw=anchor_pose[5],
                speed=ANCHOR_MOVE_SPEED_MM_S,
                wait=True,
            ),
        )
        print_robot_status(arm, "at kept anchor (position mode)")
        new_points = collect_suffix(arm, prefix, old_suffix)
    except KeyboardInterrupt:
        print("\nInterrupted; the teaching-point file was not changed.")
        return 130
    finally:
        if arm is not None:
            try:
                print("Returning to position mode...")
                print("set_mode 0:", arm.set_mode(0))
                print("set_state 0:", arm.set_state(0))
            except Exception as exc:
                print("Could not restore position mode:", exc)
            finally:
                arm.disconnect()

    if new_points is None:
        print("Cancelled; the teaching-point file was not changed.")
        return 0

    show_candidate(prefix, new_points)
    print("Old total: {} points".format(len(points)))
    print("New total: {} points".format(len(prefix) + len(new_points)))
    if not new_points:
        print("WARNING: finishing now will delete the entire old suffix.")
    confirm = input("Type OVERWRITE to back up and replace the JSON: ").strip()
    if confirm.upper() != "OVERWRITE":
        print("Not saved; the teaching-point file was not changed.")
        return 0

    backup_path = unique_backup_path(points_path)
    shutil.copy2(points_path, backup_path)
    updated = build_updated_payload(payload, keep_position, new_points, backup_path)
    atomic_write_json(points_path, updated)
    print("Updated:", points_path)
    print("Backup:", backup_path)
    print("Saved {} total point(s).".format(updated["count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
