#!/usr/bin/env python3
"""
Build fixed-rate VLA samples from independently timestamped raw streams.

Robot poses are interpolated. RealSense and ultrasound frames use nearest
capture time. Device clocks are linearly mapped to the host clock to remove
arrival-time jitter while retaining a common cross-device time base.
"""

import argparse
import bisect
import json
import math
import os
import statistics


DEFAULT_INSTRUCTION = (
    "Move the ultrasound probe along the demonstrated path while maintaining "
    "stable contact and image quality."
)

CLOCK_FIT_MARGIN_NS = 5_000_000_000


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # A power loss can leave one incomplete final line. Earlier
                # complete records remain usable.
                if line_no == len(lines):
                    break
                raise ValueError("{}:{}: {}".format(path, line_no, exc))
    return records


def as_int(value):
    if value is None:
        return None
    return int(value)


def records_in_clock_fit_window(
    records, capture_start_ns, capture_end_ns, margin_ns=CLOCK_FIT_MARGIN_NS
):
    """Select arrival records near the valid capture interval for clock fitting.

    PhaseLink may emit non-frame/control packets while the UI is being closed.
    Their packet tails do not necessarily contain a device timestamp.  Limiting
    the regression interval prevents those packets from poisoning an otherwise
    valid device-to-host clock fit, while all raw records remain preserved.
    """
    lower_ns = capture_start_ns - margin_ns
    upper_ns = capture_end_ns + margin_ns
    selected = []
    for record in records:
        host_value = record.get("host_unix_ns")
        if host_value is None:
            continue
        host_ns = as_int(host_value)
        if lower_ns <= host_ns <= upper_ns:
            selected.append(record)
    return selected


def fit_device_clock(records, field, ticks_per_second):
    pairs = []
    for record in records:
        device_value = record.get(field)
        host_value = record.get("host_unix_ns")
        if device_value is not None and host_value is not None:
            # RealSense timestamps contain sub-millisecond fractions. Keep
            # them instead of truncating to an integer millisecond.
            pairs.append((float(device_value), as_int(host_value)))

    result = {
        "device_field": field,
        "ticks_per_second": ticks_per_second,
        "pair_count": len(pairs),
        "used": False,
    }
    if len(pairs) < 10:
        return None, result

    # Use relative values so uint64 device ticks and Unix nanoseconds do not
    # lose precision during the least-squares calculation.
    device0, host0 = pairs[0]
    xs = [(device - device0) / float(ticks_per_second) for device, _ in pairs]
    ys = [(host - host0) / 1e9 for _, host in pairs]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x <= 0:
        return None, result
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance_x
    intercept = mean_y - slope * mean_x

    residual_ms = [
        (y - (intercept + slope * x)) * 1000.0 for x, y in zip(xs, ys)
    ]
    rms_ms = math.sqrt(statistics.fmean(value * value for value in residual_ms))
    max_abs_ms = max(abs(value) for value in residual_ms)
    drift_ppm = (slope - 1.0) * 1e6
    result.update({
        "slope": slope,
        "drift_ppm": drift_ppm,
        "arrival_residual_rms_ms": rms_ms,
        "arrival_residual_max_abs_ms": max_abs_ms,
    })

    # A bad/reset device clock is worse than raw host arrival timestamps.
    if not 0.95 <= slope <= 1.05:
        result["rejected_reason"] = "device clock slope outside 0.95..1.05"
        return None, result

    def mapper(record):
        device_value = float(record[field])
        relative_s = (device_value - device0) / float(ticks_per_second)
        mapped_relative_s = intercept + slope * relative_s
        return host0 + int(round(mapped_relative_s * 1e9))

    result["used"] = True
    return mapper, result


def attach_sync_times(records, mapper, host_field="host_unix_ns"):
    for record in records:
        record["_sync_unix_ns"] = (
            mapper(record) if mapper is not None else as_int(record[host_field])
        )
    records.sort(key=lambda item: item["_sync_unix_ns"])


def nearest_record(records, times, target_ns):
    index = bisect.bisect_left(times, target_ns)
    candidates = []
    if index < len(records):
        candidates.append(records[index])
    if index > 0:
        candidates.append(records[index - 1])
    record = min(
        candidates, key=lambda item: abs(item["_sync_unix_ns"] - target_ns)
    )
    return record, (record["_sync_unix_ns"] - target_ns) / 1e6


def active_action(records, times, target_ns):
    """Return the latest command issued at or before the target timestamp."""
    index = bisect.bisect_right(times, target_ns) - 1
    if index < 0:
        return None, None
    record = records[index]
    return record, (record["_sync_unix_ns"] - target_ns) / 1e6


def action_is_valid(action_record, action_skew_ms, servo_max_skew_ms):
    if action_record is None or action_record.get("return_code", 0) != 0:
        return False
    if action_record.get("type") == "position_target":
        return True
    return abs(action_skew_ms) <= servo_max_skew_ms


def interpolate_scalar(a, b, alpha):
    return float(a) + (float(b) - float(a)) * alpha


def interpolate_angle_degrees(a, b, alpha):
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    value = float(a) + delta * alpha
    return (value + 180.0) % 360.0 - 180.0


def interpolate_vector(a, b, alpha, angle_start=None):
    if a is None or b is None or len(a) != len(b):
        return a if alpha < 0.5 else b
    values = []
    for index, (left, right) in enumerate(zip(a, b)):
        if angle_start is not None and index >= angle_start:
            values.append(interpolate_angle_degrees(left, right, alpha))
        else:
            values.append(interpolate_scalar(left, right, alpha))
    return values


def interpolate_robot(records, times, target_ns):
    right_index = bisect.bisect_left(times, target_ns)
    if right_index <= 0:
        left = right = records[0]
        alpha = 0.0
    elif right_index >= len(records):
        left = right = records[-1]
        alpha = 0.0
    else:
        left = records[right_index - 1]
        right = records[right_index]
        span = right["_sync_unix_ns"] - left["_sync_unix_ns"]
        alpha = 0.0 if span <= 0 else (
            (target_ns - left["_sync_unix_ns"]) / float(span)
        )

    nearest = min(
        (left, right), key=lambda item: abs(item["_sync_unix_ns"] - target_ns)
    )
    return {
        "tcp_pose": interpolate_vector(
            left.get("tcp_pose"), right.get("tcp_pose"), alpha, angle_start=3
        ),
        "joint_angles": interpolate_vector(
            left.get("joint_angles"), right.get("joint_angles"), alpha, angle_start=0
        ),
        "joint_torque": nearest.get("joint_torque"),
        "ft_ext_force": nearest.get("ft_ext_force"),
        "ft_raw_force": nearest.get("ft_raw_force"),
        "state": nearest.get("state"),
        "mode": nearest.get("mode"),
        "error_code": nearest.get("error_code"),
        "warn_code": nearest.get("warn_code"),
        "source_before_index": left.get("record_index"),
        "source_after_index": right.get("record_index"),
        "interpolation_alpha": alpha,
        "nearest_skew_ms": (
            nearest["_sync_unix_ns"] - target_ns
        ) / 1e6,
    }


def public_record(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override metadata.json instruction for every aligned sample.",
    )
    parser.add_argument("--robot-max-skew-ms", type=float, default=80.0)
    parser.add_argument("--realsense-max-skew-ms", type=float, default=50.0)
    parser.add_argument("--ultrasound-max-skew-ms", type=float, default=50.0)
    parser.add_argument("--action-max-skew-ms", type=float, default=80.0)
    parser.add_argument("--drop-invalid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    episode_dir = os.path.abspath(args.episode)
    metadata = read_json(os.path.join(episode_dir, "metadata.json"))
    instruction = args.instruction or metadata.get("instruction") or DEFAULT_INSTRUCTION
    robot = read_jsonl(os.path.join(episode_dir, "robot_records.jsonl"))
    realsense = read_jsonl(os.path.join(episode_dir, "realsense_frames.jsonl"))
    ultrasound = read_jsonl(os.path.join(episode_dir, "ultrasound_frames.jsonl"))
    actions_path = os.path.join(episode_dir, "robot_actions.jsonl")
    actions = read_jsonl(actions_path) if os.path.isfile(actions_path) else []
    if not robot or not realsense or not ultrasound or not actions:
        raise RuntimeError(
            "Robot, RealSense, ultrasound, and actual command streams must "
            "all contain records: robot={}, realsense={}, ultrasound={}, "
            "actions={}".format(
                len(robot), len(realsense), len(ultrasound), len(actions)
            )
        )

    window = metadata.get("capture_window") or {}
    if window.get("start_unix_ns") is None or window.get("end_unix_ns") is None:
        raise RuntimeError(
            "metadata.json must contain capture_window.start_unix_ns and "
            "capture_window.end_unix_ns; refusing to align setup or "
            "move-to-start data as valid VLA samples"
        )
    requested_start = as_int(window["start_unix_ns"])
    requested_end = as_int(window["end_unix_ns"])
    if requested_end <= requested_start:
        raise RuntimeError("capture_window end must be later than start")

    fit_window_start = requested_start - CLOCK_FIT_MARGIN_NS
    fit_window_end = requested_end + CLOCK_FIT_MARGIN_NS
    rs_fit_records = records_in_clock_fit_window(
        realsense, requested_start, requested_end
    )
    us_fit_records = records_in_clock_fit_window(
        ultrasound, requested_start, requested_end
    )

    rs_mapper, rs_clock = fit_device_clock(
        rs_fit_records, "device_timestamp_ms", ticks_per_second=1000.0
    )
    us_mapper, us_clock = fit_device_clock(
        us_fit_records,
        "device_timestamp_ticks_200mhz",
        ticks_per_second=200000000.0,
    )
    for clock_fit, total_count in (
        (rs_clock, len(realsense)),
        (us_clock, len(ultrasound)),
    ):
        clock_fit.update({
            "total_record_count": total_count,
            "fit_window_margin_s": CLOCK_FIT_MARGIN_NS / 1e9,
            "fit_window_start_unix_ns": fit_window_start,
            "fit_window_end_unix_ns": fit_window_end,
        })
    attach_sync_times(robot, None)
    attach_sync_times(realsense, rs_mapper)
    attach_sync_times(ultrasound, us_mapper)
    # A command is considered issued/effective only after the SDK call has
    # returned, not merely when the host started the call.
    for action in actions:
        action.setdefault("command_end_unix_ns", action.get("host_unix_ns"))
        action.setdefault(
            "effective_unix_ns", action.get("command_end_unix_ns")
        )
    attach_sync_times(actions, None, host_field="effective_unix_ns")

    robot_times = [item["_sync_unix_ns"] for item in robot]
    rs_times = [item["_sync_unix_ns"] for item in realsense]
    us_times = [item["_sync_unix_ns"] for item in ultrasound]
    action_times = [item["_sync_unix_ns"] for item in actions]
    overlap_start = max(requested_start, robot_times[0], rs_times[0], us_times[0])
    overlap_end = min(requested_end, robot_times[-1], rs_times[-1], us_times[-1])
    if overlap_end <= overlap_start:
        raise RuntimeError("The three streams have no common capture interval")

    period_ns = int(round(1e9 / args.fps))
    output_path = os.path.join(episode_dir, "aligned_samples.jsonl")
    sample_count = 0
    invalid_count = 0
    skipped_count = 0
    target_ns = overlap_start

    with open(output_path, "w", encoding="utf-8", buffering=1) as output:
        while target_ns <= overlap_end:
            robot_state = interpolate_robot(robot, robot_times, target_ns)
            rs_record, rs_skew_ms = nearest_record(
                realsense, rs_times, target_ns
            )
            us_record, us_skew_ms = nearest_record(
                ultrasound, us_times, target_ns
            )
            action_record, action_skew_ms = active_action(
                actions, action_times, target_ns
            )
            if action_record is not None:
                action_payload = {
                    **public_record(action_record),
                    "alignment_skew_ms": action_skew_ms,
                }
                # Servo targets are refreshed every 20 ms, so a large age
                # indicates missing commands. A blocking position target is a
                # long-lived action and remains active until the next command.
                action_valid = action_is_valid(
                    action_record,
                    action_skew_ms,
                    args.action_max_skew_ms,
                )
            else:
                action_skew_ms = None
                action_payload = None
                action_valid = False

            valid = (
                abs(robot_state["nearest_skew_ms"]) <= args.robot_max_skew_ms
                and abs(rs_skew_ms) <= args.realsense_max_skew_ms
                and abs(us_skew_ms) <= args.ultrasound_max_skew_ms
                and action_valid
            )
            if not valid:
                invalid_count += 1
                if args.drop_invalid:
                    skipped_count += 1
                    target_ns += period_ns
                    continue

            sample = {
                "schema": "ultrasound_vla_aligned_sample_v1",
                "sample_index": sample_count,
                "timestamp_unix_ns": target_ns,
                "timestamp_from_start_s": (target_ns - overlap_start) / 1e9,
                "instruction": instruction,
                "observation": {
                    "robot": {
                        **robot_state,
                        "alignment_skew_ms": robot_state["nearest_skew_ms"],
                    },
                    "realsense": {
                        **public_record(rs_record),
                        "alignment_skew_ms": rs_skew_ms,
                    },
                    "ultrasound": {
                        **public_record(us_record),
                        "packet_file": "ultrasound_packets.bin",
                        "alignment_skew_ms": us_skew_ms,
                    },
                },
                # This is always an actual issued position/servo command. An
                # observed future robot state is never substituted for action.
                "action": action_payload,
                "quality": {
                    "valid": valid,
                    "robot_skew_ms": robot_state["nearest_skew_ms"],
                    "realsense_skew_ms": rs_skew_ms,
                    "ultrasound_skew_ms": us_skew_ms,
                    "action_skew_ms": action_skew_ms,
                },
            }
            output.write(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            sample_count += 1
            target_ns += period_ns

    alignment = {
        "schema": "ultrasound_vla_alignment_v1",
        "episode_dir": episode_dir,
        "fps": args.fps,
        "period_ns": period_ns,
        "overlap_start_unix_ns": overlap_start,
        "overlap_end_unix_ns": overlap_end,
        "overlap_duration_s": (overlap_end - overlap_start) / 1e9,
        "input_counts": {
            "robot": len(robot),
            "realsense": len(realsense),
            "ultrasound": len(ultrasound),
            "robot_actions": len(actions),
        },
        "output_sample_count": sample_count,
        "invalid_before_drop_count": invalid_count,
        "skipped_invalid_count": skipped_count,
        "clock_fits": {
            "realsense": rs_clock,
            "ultrasound": us_clock,
        },
        "tolerances_ms": {
            "robot": args.robot_max_skew_ms,
            "realsense": args.realsense_max_skew_ms,
            "ultrasound": args.ultrasound_max_skew_ms,
            "action": args.action_max_skew_ms,
        },
        "action_source": "commanded_position_or_cartesian_servo_target",
        "action_validity_policy": {
            "position_target": "active until the next issued command; age limit is not applied",
            "cartesian_servo_target": "must be no older than action_max_skew_ms",
        },
        "output": "aligned_samples.jsonl",
    }
    with open(
        os.path.join(episode_dir, "alignment.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(alignment, handle, ensure_ascii=False, indent=2)

    print("Aligned samples:", sample_count)
    print("Invalid before drop:", invalid_count)
    print("Output:", output_path)
    print("RealSense device-clock fit used:", rs_clock["used"])
    print("Ultrasound device-clock fit used:", us_clock["used"])


if __name__ == "__main__":
    main()
