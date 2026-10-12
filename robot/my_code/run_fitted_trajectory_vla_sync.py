#!/usr/bin/env python3
"""
Run the fitted taught-point trajectory while recording xArm, RealSense, and
PhaseLink ultrasound as independently timestamped streams.

This is the synchronized collector for the automatic scan stage. It reuses
the exact trajectory and servo parameters from
run_single_curve_from_latest_points_servo_soft_pose.py.
"""

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402
from collect_teach_vla_sync import (  # noqa: E402
    DEFAULT_PHASELINK_EXE,
    RobotReportRecorder,
    launch_phaselink,
)
from realsense_sync_recorder import RealSenseRecorder, ensure_dir  # noqa: E402
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


DEFAULT_OUTPUT_ROOT = os.environ.get(
    "VLA_OUTPUT_ROOT", r"D:\fitted_trajectory_vla_runs"
)
DEFAULT_MIN_FREE_GB = 10.0
TASK_INSTRUCTION = (
    "Move from the configured capture-start pose to the first taught scan "
    "point using a constant-height approach, descend vertically, then follow "
    "the fitted trajectory while collecting synchronized RGB-D and ultrasound."
)
DEFAULT_APPROACH_SPEED_MM_S = 10.0
DEFAULT_DESCENT_SPEED_MM_S = 10.0
ROBOT_READY_TIMEOUT_SECONDS = 10.0
ROBOT_READY_STABLE_SECONDS = 0.5
ROBOT_READY_POLL_SECONDS = 0.05
ULTRASOUND_READY_TIMEOUT_SECONDS = 10.0
ULTRASOUND_READY_POLL_SECONDS = 0.1
ULTRASOUND_PRESET_NAME = "contact_scan_v1"
ULTRASOUND_REQUESTED_SETTINGS = {
    "group": 0,
    "basic": {
        "gain_db": 8.0,
        "range_start_mm": 0.0,
        "range_end_mm": 50.0,
        "probe_pitch_mm": 0.60,
        "probe_primary_elements": 64,
        "probe_frequency_mhz": 5.0,
        "workpiece_longitudinal_velocity_m_s": 3000.0,
        "wedge_velocity_m_s": 2330.0,
        "wedge_enabled": True,
        "wedge_x_sdk_mm": 85.60,
        "wedge_z_mm": 0.0,
        "frame_rate_per_beam_hz": 20.0,
        "voltage_v": 50,
        "pulse_width_ns": 100,
        "polarity": "negative",
        "high_pass_mhz": 1,
        "low_pass_mhz": 25,
        "rectifier": "full_wave",
        "display_max_amplitude_percent": 800,
        "sample_points": 1024,
        "zero_position_ns": 1350.0,
    },
    "array": {
        "scan_type": "linear",
        "wave_type": "longitudinal",
        "beam_mode": "single",
        "aperture": 16,
        "element_step": 1,
        "first_element": 1,
        "last_element": 64,
        "angle_deg": 0.0,
        "focus_mode": "none",
        "focus_depth_mm": 0.0,
        "transmit_channel": 1,
        "receive_channel": 1,
    },
    "gate_a": {
        "enabled": True,
        "synchronization": "pulser",
        "start_mm": 14.41,
        "width_mm": 32.70,
        "end_mm": 47.11,
        "threshold_percent": 11.0,
        "measure": "peak",
        "synchronous_acquisition": False,
    },
    "tcg": {
        "enabled": True,
        "points": [
            {"index": 1, "depth_mm": 30.0, "gain_db": 5.0},
            {"index": 2, "depth_mm": 20.0, "gain_db": 7.0},
        ],
    },
}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_episode_id():
    return time.strftime("fitted_scan_vla_%Y%m%d_%H%M%S")


def create_unique_episode_dir(output_root):
    ensure_dir(output_root)
    base_id = make_episode_id()
    for suffix in range(1000):
        episode_id = base_id if suffix == 0 else "{}_{:03d}".format(base_id, suffix)
        episode_dir = os.path.abspath(os.path.join(output_root, episode_id))
        try:
            os.makedirs(episode_dir, exist_ok=False)
            return episode_id, episode_dir
        except FileExistsError:
            continue
    raise RuntimeError("could not create a unique episode directory")


def prepare_output_root(output_root, min_free_gb=DEFAULT_MIN_FREE_GB):
    """Resolve the episode root and reject insufficient storage before hardware starts."""
    if min_free_gb < 0:
        raise ValueError("--min-free-gb must be non-negative")

    resolved_root = os.path.abspath(
        os.path.expandvars(os.path.expanduser(output_root))
    )
    drive, _ = os.path.splitdrive(resolved_root)
    if drive and not os.path.exists(drive + os.sep):
        raise RuntimeError("output drive does not exist: {}".format(drive))

    ensure_dir(resolved_root)
    usage = shutil.disk_usage(resolved_root)
    required_free_bytes = int(min_free_gb * 1024 ** 3)
    if usage.free < required_free_bytes:
        raise RuntimeError(
            "insufficient free space at {}: {:.2f} GB available, {:.2f} GB "
            "required before collection".format(
                resolved_root, usage.free / 1024 ** 3, min_free_gb
            )
        )
    return resolved_root, {
        "output_root": resolved_root,
        "free_bytes_before_episode": usage.free,
        "free_gb_before_episode": usage.free / 1024 ** 3,
        "minimum_free_gb": min_free_gb,
    }


def wait_for_ultrasound_packets(
    episode_dir,
    timeout=ULTRASOUND_READY_TIMEOUT_SECONDS,
    poll_seconds=ULTRASOUND_READY_POLL_SECONDS,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Require evidence of raw PhaseLink packets before servo motion starts."""
    packet_path = os.path.join(episode_dir, "ultrasound_packets.bin")
    index_path = os.path.join(episode_dir, "ultrasound_frames.jsonl")
    metadata_path = os.path.join(episode_dir, "ultrasound_metadata.json")
    deadline = monotonic() + timeout
    packet_bytes = 0
    index_bytes = 0

    while monotonic() < deadline:
        if os.path.isfile(packet_path):
            packet_bytes = os.path.getsize(packet_path)
        if os.path.isfile(index_path):
            index_bytes = os.path.getsize(index_path)
        if packet_bytes > 0 and index_bytes > 0:
            return {
                "packet_file_bytes": packet_bytes,
                "index_file_bytes": index_bytes,
            }
        sleep(poll_seconds)

    recorder_error = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                recorder_error = json.load(handle).get("last_error")
        except (OSError, ValueError):
            pass
    detail = " recorder_error={}".format(recorder_error) if recorder_error else ""
    raise RuntimeError(
        "PhaseLink raw stream is not reaching Client::DataPacket: "
        "ultrasound_packets.bin={} bytes, ultrasound_frames.jsonl={} bytes "
        "after {:.1f} s. Do not run the robot trajectory. Verify that the "
        "device is producing acquisition packets, not only that the Start "
        "scan button changed text.{}".format(
            packet_bytes, index_bytes, timeout, detail
        )
    )


class ActionRecorder:
    """Serialize action JSONL off the time-critical servo loop."""

    _STOP = object()

    def __init__(self, output_path):
        self.output_path = output_path
        self.records = queue.SimpleQueue()
        self.writer_thread = threading.Thread(
            target=self._writer, name="xarm-action-writer", daemon=True
        )
        self.started_event = threading.Event()
        self.record_count = 0
        self.writer_error = None

    def start(self):
        self.writer_thread.start()
        self.started_event.wait(timeout=5)
        if not self.started_event.is_set():
            raise RuntimeError("xArm action writer startup timed out")
        if self.writer_error is not None:
            raise RuntimeError(
                "xArm action writer failed: {}".format(self.writer_error)
            )

    def record(self, action):
        if self.writer_error is not None:
            raise RuntimeError(
                "xArm action writer failed: {}".format(self.writer_error)
            )
        self.records.put(action)

    def _writer(self):
        try:
            with open(self.output_path, "w", encoding="utf-8", buffering=1) as handle:
                self.started_event.set()
                while True:
                    record = self.records.get()
                    if record is self._STOP:
                        return
                    handle.write(
                        json.dumps(
                            record, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                    self.record_count += 1
        except Exception as exc:
            self.writer_error = exc
            self.started_event.set()

    def stop(self):
        self.records.put(self._STOP)
        self.writer_thread.join(timeout=10)
        if self.writer_thread.is_alive():
            raise RuntimeError("xArm action writer did not stop")
        if self.writer_error is not None:
            raise RuntimeError(
                "xArm action writer failed: {}".format(self.writer_error)
            )


def countdown(seconds):
    print("Robot will move in {} seconds.".format(seconds))
    print("Keep the emergency stop reachable. Press Ctrl+C now to cancel.")
    for left in range(seconds, 0, -1):
        print("{}...".format(left))
        time.sleep(1)


def robot_status_snapshot(arm):
    """Read the controller state used to gate the first motion command."""
    state_code, state = arm.get_state()
    error_code, error_warn = arm.get_err_warn_code()
    if error_code == 0:
        error, warn = error_warn
    else:
        error, warn = arm.error_code, arm.warn_code

    axis = arm.axis
    motor_enable = list(arm.motor_enable_states)
    active_motor_enable = motor_enable[:axis] if axis > 0 else motor_enable
    motor_states_available = bool(active_motor_enable) and all(
        value in (0, 1) for value in active_motor_enable
    )
    motors_ready = motor_states_available and all(
        value == 1 for value in active_motor_enable
    )
    protocol_ready = bool(arm.core.state_is_ready)
    return {
        "state_query_code": state_code,
        "state": state,
        "mode": arm.mode,
        "error_query_code": error_code,
        "error_code": error,
        "warn_code": warn,
        "axis": axis,
        "motor_enable_states": motor_enable,
        "motor_states_available": motor_states_available,
        "motors_ready": motors_ready,
        "protocol_ready": protocol_ready,
    }


def wait_for_robot_ready(
    arm,
    expected_mode=0,
    timeout=ROBOT_READY_TIMEOUT_SECONDS,
    stable_seconds=ROBOT_READY_STABLE_SECONDS,
    poll_seconds=ROBOT_READY_POLL_SECONDS,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Wait until state 2 and all motor/protocol readiness flags are stable."""
    deadline = monotonic() + timeout
    ready_since = None
    last_status = None
    while monotonic() < deadline:
        last_status = robot_status_snapshot(arm)
        is_ready = (
            last_status["state_query_code"] == 0
            and last_status["error_query_code"] == 0
            and last_status["state"] == 2
            and last_status["mode"] == expected_mode
            and last_status["error_code"] == 0
            and last_status["warn_code"] == 0
            and (
                not last_status["motor_states_available"]
                or last_status["motors_ready"]
            )
            and last_status["protocol_ready"]
        )
        if is_ready:
            if ready_since is None:
                ready_since = monotonic()
            if monotonic() - ready_since >= stable_seconds:
                print("xArm READY (stable):", last_status)
                return last_status
        else:
            ready_since = None
        sleep(poll_seconds)

    raise RuntimeError(
        "xArm did not become stably READY in mode {} after final "
        "set_state(0); ".format(expected_mode)
        + "last_status={}".format(last_status)
    )


def prepare_robot_for_capture(arm, taught_tcp_offset):
    """Configure the robot without changing its Cartesian pose."""
    if arm.warn_code or arm.error_code:
        raise RuntimeError(
            "xArm has warn/error before motion (warn={}, error={}). Inspect "
            "and clear it deliberately in the controller/Studio, then rerun."
            .format(arm.warn_code, arm.error_code)
        )

    # set_mode and set_tcp_offset are critical configuration changes. The xArm
    # controller enters feedback state 5 (MODE_CHANGED) after either call and
    # cannot accept motion until set_state(0) is sent afterwards. Keep
    # set_state(0) last, then wait for feedback state 2 (READY).
    setup_calls = (
        ("motion_enable", lambda: arm.motion_enable(True)),
        ("set_mode 0", lambda: arm.set_mode(0)),
    )
    for name, operation in setup_calls:
        code = operation()
        print("{}:".format(name), code)
        if code != 0:
            raise RuntimeError("{} failed, code={}".format(name, code))
    tcp_code = arm.set_tcp_offset(taught_tcp_offset, wait=True)
    print("set_tcp_offset from teach points:", tcp_code, taught_tcp_offset)
    if tcp_code != 0:
        raise RuntimeError(
            "failed to restore taught TCP offset, code={}".format(tcp_code)
        )

    state_code = arm.set_state(0)
    print("set_state 0 after TCP/mode configuration:", state_code)
    if state_code != 0:
        raise RuntimeError(
            "set_state 0 after TCP/mode configuration failed, code={}".format(
                state_code
            )
        )
    wait_for_robot_ready(arm, expected_mode=0)


def restore_position_mode_ready(arm):
    """Re-arm position mode immediately before the first capture motion.

    PhaseLink setup can leave the robot waiting for long enough that the
    controller is no longer in READY state.  set_mode may itself produce
    state 5 (MODE_CHANGED), so set_state(0) must be sent afterwards.
    """
    mode_code = arm.set_mode(0)
    print("set_mode 0 immediately before capture:", mode_code)
    if mode_code != 0:
        raise RuntimeError(
            "set_mode 0 immediately before capture failed, code={}".format(
                mode_code
            )
        )

    state_code = arm.set_state(0)
    print("set_state 0 immediately before capture:", state_code)
    if state_code != 0:
        raise RuntimeError(
            "set_state 0 immediately before capture failed, code={}".format(
                state_code
            )
        )
    return wait_for_robot_ready(arm, expected_mode=0)



def read_current_tcp_pose(arm):
    code, pose = arm.get_position()
    if code != 0 or not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise RuntimeError(
            "failed to read current TCP pose before capture, code={}, pose={}".format(
                code, pose
            )
        )
    return [float(value) for value in pose[:6]]


def record_position_action(
    arm,
    action_recorder,
    action_index,
    target_pose,
    speed_mm_s,
    phase,
):
    """Issue one blocking position target and preserve its true timing."""
    command_start_monotonic_ns = time.monotonic_ns()
    command_start_unix_ns = time.time_ns()
    return_code = arm.set_position(
        x=target_pose[0],
        y=target_pose[1],
        z=target_pose[2],
        roll=target_pose[3],
        pitch=target_pose[4],
        yaw=target_pose[5],
        speed=speed_mm_s,
        wait=True,
    )
    command_end_monotonic_ns = time.monotonic_ns()
    command_end_unix_ns = time.time_ns()
    action_recorder.record({
        "schema": "xarm_position_action_v1",
        "action_index": action_index,
        "host_unix_ns": command_start_unix_ns,
        "host_monotonic_ns": command_start_monotonic_ns,
        # A wait=True call returns only after arrival, but the target becomes
        # active near call entry. Keep both the effective and completion times.
        "effective_unix_ns": command_start_unix_ns,
        "effective_monotonic_ns": command_start_monotonic_ns,
        "command_end_unix_ns": command_end_unix_ns,
        "command_end_monotonic_ns": command_end_monotonic_ns,
        "scheduled_monotonic_ns": command_start_monotonic_ns,
        "schedule_lag_ms": 0.0,
        "type": "position_target",
        "phase": phase,
        "target_tcp_pose": list(target_pose),
        "speed": speed_mm_s,
        "acc": None,
        "servo_period": None,
        "speed_mm_s": speed_mm_s,
        "acc_mm_s2": None,
        "servo_period_s": None,
        "wait": True,
        "return_code": return_code,
    })
    if return_code != 0:
        raise RuntimeError(
            "{} position move failed, code={}".format(phase, return_code)
        )


def move_to_trajectory_start_staged(
    arm,
    curve,
    capture_start_pose,
    action_recorder,
    approach_speed_mm_s,
    descent_speed_mm_s,
):
    """Approach the first taught point without lowering Z until directly above it."""
    first_pose = list(curve[0])
    if capture_start_pose[2] < first_pose[2]:
        raise RuntimeError(
            "capture-start Z {:.3f} mm is below trajectory-start Z {:.3f} mm; "
            "refusing staged descent".format(capture_start_pose[2], first_pose[2])
        )

    above_first_pose = list(first_pose)
    above_first_pose[2] = capture_start_pose[2]
    print("VLA stage 1: moving above first taught point at constant Z...")
    record_position_action(
        arm,
        action_recorder,
        action_index=0,
        target_pose=above_first_pose,
        speed_mm_s=approach_speed_mm_s,
        phase="constant_z_approach",
    )

    print("VLA stage 2: descending vertically to first taught point...")
    record_position_action(
        arm,
        action_recorder,
        action_index=1,
        target_pose=first_pose,
        speed_mm_s=descent_speed_mm_s,
        phase="vertical_descent",
    )

    print("Switching to Cartesian servo mode...")
    code_mode = arm.set_mode(1)
    code_state = arm.set_state(0)
    print("set_mode 1:", code_mode)
    print("set_state 0:", code_state)
    if code_mode != 0 or code_state != 0:
        raise RuntimeError(
            "failed to enter servo mode: mode_code={}, state_code={}".format(
                code_mode, code_state
            )
        )
    wait_for_robot_ready(arm, expected_mode=1)
    return 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--allow-robot-ip-mismatch", action="store_true")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Episode parent directory (default: %(default)s; environment "
            "override: VLA_OUTPUT_ROOT)."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help="Minimum free space required before any device or robot starts.",
    )
    parser.add_argument(
        "--ultrasound-ready-timeout",
        type=float,
        default=ULTRASOUND_READY_TIMEOUT_SECONDS,
        help=(
            "Seconds to wait for non-empty PhaseLink raw packet and index "
            "files before allowing the servo trajectory."
        ),
    )
    parser.add_argument("--robot-report-type", choices=("real", "rich"), default="real")
    parser.add_argument(
        "--approach-speed",
        type=float,
        default=DEFAULT_APPROACH_SPEED_MM_S,
        help="Constant-Z approach speed in mm/s (default: %(default)s).",
    )
    parser.add_argument(
        "--descent-speed",
        type=float,
        default=DEFAULT_DESCENT_SPEED_MM_S,
        help="Final vertical descent speed in mm/s (default: %(default)s).",
    )
    parser.add_argument("--phaselink-exe", default=DEFAULT_PHASELINK_EXE)
    parser.add_argument("--no-launch-phaselink", action="store_true")
    parser.add_argument("--no-realsense", action="store_true")
    parser.add_argument(
        "--realsense-width",
        type=int,
        default=1280,
        help="RealSense color/depth stream width (default: %(default)s).",
    )
    parser.add_argument(
        "--realsense-height",
        type=int,
        default=720,
        help="RealSense color/depth stream height (default: %(default)s).",
    )
    parser.add_argument(
        "--realsense-fps",
        type=int,
        default=30,
        help="RealSense color/depth stream rate (default: %(default)s).",
    )
    parser.add_argument(
        "--no-realsense-preview",
        action="store_true",
        help="Record RGB-D without opening the live RGB preview window.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.realsense_width <= 0
        or args.realsense_height <= 0
        or args.realsense_fps <= 0
    ):
        raise ValueError("RealSense width, height and fps must be positive")
    if args.approach_speed <= 0 or args.descent_speed <= 0:
        raise ValueError("approach and descent speeds must be positive")
    if args.robot_ip != TEACH_ROBOT_IP and not args.allow_robot_ip_mismatch:
        raise ValueError(
            "--robot-ip {} differs from teach-point robot_ip {}. Refusing "
            "motion; verify the robot and pass --allow-robot-ip-mismatch only "
            "if this is intentional.".format(args.robot_ip, TEACH_ROBOT_IP)
        )
    output_root, storage_metadata = prepare_output_root(
        args.output_root, args.min_free_gb
    )
    curve = build_curve()
    episode_id, episode_dir = create_unique_episode_dir(output_root)

    trajectory_payload = {
        "schema": "fitted_pose_trajectory_v1",
        "source_teach_points_file": os.path.abspath(TEACH_POINTS_PATH),
        "teach_points": TEACH_POINTS,
        "tcp_offset": TEACH_TCP_OFFSET,
        "dry_run_lift_mm": DRY_RUN_LIFT_MM,
        "capture_start_pose": None,
        "capture_start_pose_source": "xArm get_position immediately before capture_window.start",
        "staged_approach": {
            "constant_z_target_pose": None,
            "constant_z_source": "actual capture-start TCP Z",
            "vertical_descent_target_pose": curve[0],
            "approach_speed_mm_s": args.approach_speed,
            "descent_speed_mm_s": args.descent_speed,
        },
        "servo_speed_mm_s": SERVO_SPEED_MM_S,
        "servo_acc_mm_s2": SERVO_ACC_MM_S2,
        "servo_period_seconds": SERVO_PERIOD_SECONDS,
        "path_point_spacing_mm": PATH_POINT_SPACING_MM,
        "path_length_mm": path_length(curve),
        "trajectory": [
            {"trajectory_index": index, "target_tcp_pose": pose}
            for index, pose in enumerate(curve)
        ],
    }
    write_json(os.path.join(episode_dir, "trajectory.json"), trajectory_payload)

    metadata = {
        "schema": "xarm_realsense_phaselink_fitted_trajectory_episode_v1",
        "episode_id": episode_id,
        "created_at_unix_ns": time.time_ns(),
        "instruction": TASK_INSTRUCTION,
        "storage": storage_metadata,
        "robot_ip": args.robot_ip,
        "robot_report_type": args.robot_report_type,
        "capture_start_pose": None,
        "capture_start_pose_source": "xArm get_position immediately before capture_window.start",
        "staged_approach": {
            "stage_1": "move XY/RPY above first trajectory point while keeping capture-start Z",
            "stage_2": "change only Z to reach the first trajectory point",
            "approach_speed_mm_s": args.approach_speed,
            "descent_speed_mm_s": args.descent_speed,
            "included_in_capture_window": True,
        },
        "realsense_live_preview": (
            False if args.no_realsense else not args.no_realsense_preview
        ),
        "realsense_requested_stream": (
            None
            if args.no_realsense
            else {
                "width": args.realsense_width,
                "height": args.realsense_height,
                "fps": args.realsense_fps,
                "depth_aligned_to_color": True,
            }
        ),
        "ultrasound_preset": {
            "name": ULTRASOUND_PRESET_NAME,
            "requested_settings": ULTRASOUND_REQUESTED_SETTINGS,
            "application": "PhaseLink SDK configuration before connection",
        },
        "trajectory": {
            "file": "trajectory.json",
            "teach_points_file": os.path.abspath(TEACH_POINTS_PATH),
            "teach_points_count": len(TEACH_POINTS),
            "tcp_offset": TEACH_TCP_OFFSET,
            "trajectory_points": len(curve),
            "path_length_mm": path_length(curve),
            "dry_run_lift_mm": DRY_RUN_LIFT_MM,
            "capture_start_pose": None,
            "approach_speed_mm_s": args.approach_speed,
            "descent_speed_mm_s": args.descent_speed,
            "servo_speed_mm_s": SERVO_SPEED_MM_S,
            "servo_acc_mm_s2": SERVO_ACC_MM_S2,
            "servo_period_seconds": SERVO_PERIOD_SECONDS,
        },
        "clock_policy": {
            "alignment_clock": "host_unix_ns",
            "host_monotonic_recorded": True,
            "xarm_host_timestamp": "sdk_parser_before_callback_dispatch",
            "ultrasound_device_clock_hz": 200000000,
        },
        "records": {
            "robot_observations": "robot_records.jsonl",
            "robot_actions": "robot_actions.jsonl",
            "realsense": None if args.no_realsense else "realsense_frames.jsonl",
            "ultrasound": "ultrasound_frames.jsonl",
            "ultrasound_packets": "ultrasound_packets.bin",
        },
    }
    metadata_path = os.path.join(episode_dir, "metadata.json")
    write_json(metadata_path, metadata)

    print("Fitted-trajectory synchronized VLA collection")
    print("  episode:", episode_id)
    print("  output:", episode_dir)
    print(
        "  free space before episode: {:.2f} GB".format(
            storage_metadata["free_gb_before_episode"]
        )
    )
    print("  teach points:", len(TEACH_POINTS))
    print("  trajectory points:", len(curve))
    print("  path length: {:.1f} mm".format(path_length(curve)))
    print("  lift:", DRY_RUN_LIFT_MM, "mm")
    print("  VLA capture-start pose: current xArm TCP read immediately before motion")
    print("  staged approach: keep current Z, then descend to Z={:.3f} mm".format(
        curve[0][2]
    ))
    print("  expected motion duration: {:.2f} s".format(
        len(curve) * SERVO_PERIOD_SECONDS
    ))

    camera_stop = threading.Event()
    camera = None
    arm = None
    robot_recorder = None
    action_recorder = None
    phaselink_process = None
    capture_started = False

    try:
        if not args.no_launch_phaselink:
            print("Launching PhaseLink with raw-packet export...")
            phaselink_process = launch_phaselink(
                args.phaselink_exe,
                episode_dir,
                preset=ULTRASOUND_PRESET_NAME,
            )
            print(
                "PhaseLink preset requested: {} (actual readback will be saved "
                "in ultrasound_metadata.json)".format(ULTRASOUND_PRESET_NAME)
            )
        else:
            print(
                "PhaseLink auto-launch disabled. Its process must have "
                "PHASELINK_VLA_DIR={} set before startup.".format(episode_dir)
            )

        if not args.no_realsense:
            preview_enabled = not args.no_realsense_preview
            print(
                "Starting RealSense {}x{} @ {} FPS{}...".format(
                    args.realsense_width,
                    args.realsense_height,
                    args.realsense_fps,
                    " with live RGB preview" if preview_enabled else "",
                )
            )
            camera = RealSenseRecorder(
                episode_dir,
                camera_stop,
                width=args.realsense_width,
                height=args.realsense_height,
                fps=args.realsense_fps,
                preview=preview_enabled,
                preview_window_name="RealSense RGB - {}".format(episode_id),
            )
            camera.start()
            camera.started_event.wait(timeout=20)
            if camera.error is not None:
                raise RuntimeError("RealSense failed: {}".format(camera.error))
            if not camera.started_event.is_set():
                raise RuntimeError("RealSense startup timed out")

        print("Connecting xArm...")
        arm = XArmAPI(
            args.robot_ip,
            enable_report=True,
            report_type=args.robot_report_type,
        )
        time.sleep(1.0)
        robot_recorder = RobotReportRecorder(
            arm, os.path.join(episode_dir, "robot_records.jsonl")
        )
        robot_recorder.start()

        prepare_robot_for_capture(arm, TEACH_TCP_OFFSET)
        input(
            "\nIn PhaseLink: connect the ultrasound device and click Start scan. "
            "The robot's current TCP pose will become the VLA start pose. "
            "Then press Enter to arm the valid capture window, constant-Z "
            "approach, vertical descent, and fitted trajectory..."
        )

        print("Verifying that PhaseLink raw packets are reaching disk...")
        ultrasound_ready = wait_for_ultrasound_packets(
            episode_dir, timeout=args.ultrasound_ready_timeout
        )
        metadata["ultrasound_ready_check"] = ultrasound_ready
        print(
            "PhaseLink raw stream verified: {} packet bytes, {} index bytes."
            .format(
                ultrasound_ready["packet_file_bytes"],
                ultrasound_ready["index_file_bytes"],
            )
        )

        countdown(START_DELAY_SECONDS)
        # The initial READY check occurs before the operator configures
        # PhaseLink. Re-arm and verify the controller again at the last safe
        # instant before reading the capture-start pose and opening the valid
        # window. These calls configure state/mode only; they do not move the
        # robot.
        capture_start_robot_status = restore_position_mode_ready(arm)
        capture_start_pose = read_current_tcp_pose(arm)
        if capture_start_pose[2] < curve[0][2]:
            raise RuntimeError(
                "current TCP Z {:.3f} mm is below trajectory-start Z {:.3f} mm; "
                "refusing staged descent".format(
                    capture_start_pose[2], curve[0][2]
                )
            )
        above_first_pose = list(curve[0])
        above_first_pose[2] = capture_start_pose[2]
        metadata["capture_start_pose"] = capture_start_pose
        metadata["capture_start_robot_status"] = capture_start_robot_status
        metadata["trajectory"]["capture_start_pose"] = capture_start_pose
        metadata["staged_approach"]["constant_z_target_pose"] = above_first_pose
        metadata["staged_approach"]["vertical_descent_target_pose"] = list(curve[0])
        trajectory_payload["capture_start_pose"] = capture_start_pose
        trajectory_payload["staged_approach"]["constant_z_target_pose"] = above_first_pose
        write_json(os.path.join(episode_dir, "trajectory.json"), trajectory_payload)
        print("Actual VLA capture-start TCP pose:", capture_start_pose)

        actions_path = os.path.join(episode_dir, "robot_actions.jsonl")
        action_recorder = ActionRecorder(actions_path)
        action_recorder.start()
        start_monotonic_ns = time.monotonic_ns()
        start_unix_ns = time.time_ns()
        metadata["capture_window"] = {
            "start_unix_ns": start_unix_ns,
            "start_monotonic_ns": start_monotonic_ns,
            "start_pose": capture_start_pose,
        }
        capture_started = True

        approach_start_unix_ns = time.time_ns()
        action_index_offset = move_to_trajectory_start_staged(
            arm=arm,
            curve=curve,
            capture_start_pose=capture_start_pose,
            action_recorder=action_recorder,
            approach_speed_mm_s=args.approach_speed,
            descent_speed_mm_s=args.descent_speed,
        )
        servo_start_monotonic_ns = time.monotonic_ns()
        servo_start_unix_ns = time.time_ns()
        metadata["motion_phases"] = {
            "staged_approach_start_unix_ns": approach_start_unix_ns,
            "servo_trajectory_start_unix_ns": servo_start_unix_ns,
            "position_action_count": action_index_offset,
        }
        next_deadline_ns = servo_start_monotonic_ns

        for index, pose in enumerate(curve):
            scheduled_monotonic_ns = (
                servo_start_monotonic_ns
                + int(round(index * SERVO_PERIOD_SECONDS * 1e9))
            )
            command_start_monotonic_ns = time.monotonic_ns()
            command_start_unix_ns = time.time_ns()
            return_code = arm.set_servo_cartesian(
                pose,
                speed=SERVO_SPEED_MM_S,
                mvacc=SERVO_ACC_MM_S2,
            )
            command_end_monotonic_ns = time.monotonic_ns()
            command_end_unix_ns = time.time_ns()
            action = {
                "schema": "xarm_cartesian_servo_action_v1",
                "action_index": action_index_offset + index,
                "trajectory_index": index,
                "host_unix_ns": command_start_unix_ns,
                "host_monotonic_ns": command_start_monotonic_ns,
                "effective_unix_ns": command_end_unix_ns,
                "effective_monotonic_ns": command_end_monotonic_ns,
                "command_end_unix_ns": command_end_unix_ns,
                "command_end_monotonic_ns": command_end_monotonic_ns,
                "scheduled_monotonic_ns": scheduled_monotonic_ns,
                "schedule_lag_ms": (
                    command_start_monotonic_ns - scheduled_monotonic_ns
                ) / 1e6,
                "type": "cartesian_servo_target",
                "target_tcp_pose": pose,
                "speed": SERVO_SPEED_MM_S,
                "acc": SERVO_ACC_MM_S2,
                "servo_period": SERVO_PERIOD_SECONDS,
                "speed_mm_s": SERVO_SPEED_MM_S,
                "acc_mm_s2": SERVO_ACC_MM_S2,
                "servo_period_s": SERVO_PERIOD_SECONDS,
                "return_code": return_code,
            }
            action_recorder.record(action)

            if return_code != 0:
                raise RuntimeError(
                    "set_servo_cartesian failed at index {}, code={}".format(
                        index, return_code
                    )
                )
            if camera is not None and camera.error is not None:
                raise RuntimeError(
                    "RealSense failed during trajectory: {}".format(camera.error)
                )
            if robot_recorder.writer_error is not None:
                raise RuntimeError(
                    "xArm report writer failed during trajectory: {}".format(
                        robot_recorder.writer_error
                    )
                )
            if index % 100 == 0 or index == len(curve) - 1:
                print("servo #{:04d}/{}".format(index, len(curve) - 1))

            next_deadline_ns += int(round(SERVO_PERIOD_SECONDS * 1e9))
            remaining_ns = next_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1e9)

        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        end_monotonic_ns = time.monotonic_ns()
        end_unix_ns = time.time_ns()
        metadata["capture_window"].update({
            "end_unix_ns": end_unix_ns,
            "end_monotonic_ns": end_monotonic_ns,
            "duration_s": (end_monotonic_ns - start_monotonic_ns) / 1e9,
        })
        metadata["trajectory_completed"] = True
        print("Fitted trajectory and synchronized capture completed.")
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the robot collection safely.")
        metadata["trajectory_completed"] = False
        metadata["interrupted"] = True
        if capture_started:
            metadata.setdefault("capture_window", {}).update({
                "end_unix_ns": time.time_ns(),
                "end_monotonic_ns": time.monotonic_ns(),
            })
    except Exception as exc:
        metadata["trajectory_completed"] = False
        metadata["error"] = str(exc)
        if capture_started:
            metadata.setdefault("capture_window", {}).update({
                "end_unix_ns": time.time_ns(),
                "end_monotonic_ns": time.monotonic_ns(),
            })
        raise
    finally:
        if action_recorder is not None:
            try:
                action_recorder.stop()
                metadata["robot_action_count"] = action_recorder.record_count
            except Exception as exc:
                metadata["action_stop_error"] = str(exc)
        if robot_recorder is not None:
            try:
                robot_recorder.stop()
                metadata["robot_record_count"] = robot_recorder.record_count
                metadata["robot_dropped_count"] = robot_recorder.dropped_count
            except Exception as exc:
                metadata["robot_stop_error"] = str(exc)

        if arm is not None:
            try:
                print("Returning xArm to position mode...")
                arm.set_mode(0)
                arm.set_state(0)
            finally:
                arm.disconnect()

        if camera is not None:
            camera_stop.set()
            camera.join(timeout=45)
            if camera.is_alive():
                metadata["realsense_stop_error"] = "recorder did not stop"
            metadata["realsense_frame_count"] = camera.frame_count
            metadata["realsense_received_count"] = camera.received_count
            metadata["realsense_dropped_count"] = camera.dropped_count
            metadata["realsense_queue_capacity"] = camera.queue_capacity
            metadata["realsense_depth_png_compression"] = (
                camera.depth_png_compression
            )
            metadata["realsense_preview_closed_by_user"] = (
                camera.preview_closed_by_user
            )
            if camera.preview_error is not None:
                metadata["realsense_preview_error"] = str(camera.preview_error)
            if camera.error is not None:
                metadata["realsense_error"] = str(camera.error)

        write_json(metadata_path, metadata)

    if phaselink_process is not None and phaselink_process.poll() is None:
        print(
            "\nIn PhaseLink: click Stop scan and close the window so the final "
            "ultrasound metadata and buffers are flushed."
        )
    print("Raw episode:", episode_dir)
    print(
        "Align at 15 Hz with:\n  python align_vla_episode.py --episode "
        "\"{}\" --fps 15".format(episode_dir)
    )


if __name__ == "__main__":
    main()
