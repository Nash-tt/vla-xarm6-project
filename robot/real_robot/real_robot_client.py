#!/usr/bin/env python
"""Remote pi0 inference client for the real xArm + RealSense + PhaseLink stack.

The default mode is SHADOW: live observations are sent to the remote policy and
predicted actions are logged, but no robot command is issued.  Real motion needs
both --execute and the exact --confirm phrase.

This script never moves the robot to a start pose.  Put the probe at the intended
task start pose with an existing, validated positioning procedure before enabling
execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ULTRASOUND_SETTINGS_METADATA = (
    HERE / "config" / "training_ultrasound_metadata.json"
)
DEFAULT_ULTRASOUND_LIVE_DIR = HERE / "live_ultrasound_current"
DEFAULT_XARM_ROOT = Path(
    r"<xArm-SDK>"
)
XARM_ROOT = Path(os.environ.get("XARM_SDK_ROOT", str(DEFAULT_XARM_ROOT))).resolve()
MY_CODE = XARM_ROOT / "my_code"
for candidate in (HERE, XARM_ROOT, MY_CODE):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import remote_client as transport  # noqa: E402
from export_ultrasound_images import (  # noqa: E402
    DEFAULT_PALETTE_SOURCE,
    calculate_beam_count,
    decode_s_scan,
    load_phaselink_palette,
)


CAM_BASE = "base_0_rgb"
CAM_WRIST = "left_wrist_0_rgb"
TASK = "Follow the fitted scan trajectory with the ultrasound probe."
EXECUTE_CONFIRMATION = "EXECUTE_REAL_XARM_POLICY"


class SafetyStop(RuntimeError):
    pass


def finite_pose(value, label="pose"):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise SafetyStop(f"{label} must contain six finite values, got {value!r}")
    return array


def angle_delta_deg(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def pose_distance(a, b):
    a = finite_pose(a)
    b = finite_pose(b)
    linear = float(np.linalg.norm(a[:3] - b[:3]))
    angular = float(np.linalg.norm(angle_delta_deg(a[3:], b[3:])))
    return linear, angular


def wrap_degrees(value):
    """Return equivalent degree values in the controller-friendly [-180, 180) range."""
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


class PolicyTargetRateLimiter:
    """Convert distant absolute policy poses into a continuous pose stream.

    The limiter is deliberately opt-in.  In the default/direct mode it returns
    the raw policy target unchanged, preserving the existing safety behavior.
    """

    def __init__(self, enabled, fps, linear_speed_mm_s, angular_speed_deg_s):
        self.enabled = bool(enabled)
        self.period_s = 1.0 / float(fps)
        self.linear_speed_mm_s = float(linear_speed_mm_s)
        self.angular_speed_deg_s = float(angular_speed_deg_s)
        self.max_linear_step_mm = self.linear_speed_mm_s * self.period_s
        self.max_angular_step_deg = self.angular_speed_deg_s * self.period_s
        self.last_command = None

    @staticmethod
    def _limit_vector(delta, max_norm):
        delta = np.asarray(delta, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > max_norm and norm > 0:
            return delta * (max_norm / norm), norm, True
        return delta, norm, False

    def apply(self, raw_target, actual_pose):
        raw_target = finite_pose(raw_target, "raw policy target")
        actual_pose = finite_pose(actual_pose, "actual TCP")
        if not self.enabled:
            return raw_target.copy(), {
                "enabled": False,
                "raw_target_distance_mm": pose_distance(raw_target, actual_pose)[0],
                "raw_target_angle_deg": pose_distance(raw_target, actual_pose)[1],
            }

        base = actual_pose if self.last_command is None else self.last_command
        linear_delta, linear_requested, linear_limited = self._limit_vector(
            raw_target[:3] - base[:3], self.max_linear_step_mm
        )
        angular_delta, angular_requested, angular_limited = self._limit_vector(
            angle_delta_deg(raw_target[3:], base[3:]), self.max_angular_step_deg
        )
        command = base.copy()
        command[:3] += linear_delta
        command[3:] = wrap_degrees(base[3:] + angular_delta)
        self.last_command = command.copy()
        raw_linear, raw_angular = pose_distance(raw_target, actual_pose)
        return command, {
            "enabled": True,
            "linear_speed_mm_s": self.linear_speed_mm_s,
            "angular_speed_deg_s": self.angular_speed_deg_s,
            "linear_step_mm": float(np.linalg.norm(linear_delta)),
            "angular_step_deg": float(np.linalg.norm(angular_delta)),
            "requested_from_previous_mm": linear_requested,
            "requested_from_previous_deg": angular_requested,
            "linear_limited": linear_limited,
            "angular_limited": angular_limited,
            "raw_target_distance_mm": raw_linear,
            "raw_target_angle_deg": raw_angular,
        }


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class LivePolicyConfig:
    img_size: int = 224
    cam_base: str = CAM_BASE
    cam_wrist: str = CAM_WRIST
    task: str = TASK


class SafetyEnvelope:
    """Validate live policy targets, optionally against a reference trajectory."""

    def __init__(
        self,
        trajectory_path,
        tube_mm,
        tube_deg,
        max_step_mm,
        max_step_deg,
        max_tracking_error_mm,
        max_tracking_error_deg,
        start_tolerance_mm,
        start_tolerance_deg,
        max_backtrack_points,
        max_forward_jump_points,
    ):
        self.points = None
        self.trajectory_path = None
        if trajectory_path:
            document = read_json(trajectory_path)
            points = document.get("trajectory")
            if points and isinstance(points[0], dict):
                points = [item["target_tcp_pose"] for item in points]
            if not points:
                points = [item["pose"] for item in document.get("teach_points", [])]
            self.points = np.asarray(points, dtype=np.float64)
            if self.points.ndim != 2 or self.points.shape[1] != 6:
                raise ValueError("safety trajectory must contain Nx6 poses")
            if not np.all(np.isfinite(self.points)):
                raise ValueError("safety trajectory contains non-finite values")
            self.trajectory_path = str(Path(trajectory_path).resolve())
        self.tube_mm = float(tube_mm)
        self.tube_deg = float(tube_deg)
        self.max_step_mm = float(max_step_mm)
        self.max_step_deg = float(max_step_deg)
        self.max_tracking_error_mm = float(max_tracking_error_mm)
        self.max_tracking_error_deg = float(max_tracking_error_deg)
        self.start_tolerance_mm = float(start_tolerance_mm)
        self.start_tolerance_deg = float(start_tolerance_deg)
        self.max_backtrack_points = int(max_backtrack_points)
        self.max_forward_jump_points = int(max_forward_jump_points)
        self.progress_index = None
        self.last_target = None

    def nearest(self, pose):
        if self.points is None:
            raise RuntimeError("reference trajectory is disabled")
        pose = finite_pose(pose)
        linear = np.linalg.norm(self.points[:, :3] - pose[:3], axis=1)
        angular = np.linalg.norm(
            angle_delta_deg(self.points[:, 3:], pose[3:]), axis=1
        )
        score = linear / max(self.tube_mm, 1e-9) + angular / max(self.tube_deg, 1e-9)
        index = int(np.argmin(score))
        return index, float(linear[index]), float(angular[index])

    def require_at_start(self, actual_pose):
        actual_pose = finite_pose(actual_pose, "initial actual pose")
        if self.points is not None:
            linear, angular = pose_distance(actual_pose, self.points[0])
            if linear > self.start_tolerance_mm or angular > self.start_tolerance_deg:
                raise SafetyStop(
                    "robot is not at the demonstrated start: "
                    f"translation={linear:.2f} mm (limit {self.start_tolerance_mm:.2f}), "
                    f"rotation={angular:.2f} deg (limit {self.start_tolerance_deg:.2f})"
                )
            self.progress_index = 0
        else:
            self.progress_index = None
        self.last_target = finite_pose(actual_pose, "initial actual pose")

    def validate(self, target, actual_pose):
        target = finite_pose(target, "policy action")
        actual_pose = finite_pose(actual_pose, "actual TCP")
        index = None
        tube_linear = None
        tube_angular = None
        if self.points is not None:
            index, tube_linear, tube_angular = self.nearest(target)
            if tube_linear > self.tube_mm or tube_angular > self.tube_deg:
                raise SafetyStop(
                    f"policy target left trajectory tube at reference #{index}: "
                    f"{tube_linear:.2f} mm, {tube_angular:.2f} deg"
                )

        tracking_linear, tracking_angular = pose_distance(target, actual_pose)
        if (
            tracking_linear > self.max_tracking_error_mm
            or tracking_angular > self.max_tracking_error_deg
        ):
            raise SafetyStop(
                "policy target too far from actual TCP: "
                f"{tracking_linear:.2f} mm, {tracking_angular:.2f} deg"
            )

        if self.last_target is not None:
            step_linear, step_angular = pose_distance(target, self.last_target)
            if step_linear > self.max_step_mm or step_angular > self.max_step_deg:
                raise SafetyStop(
                    "policy action jump too large: "
                    f"{step_linear:.2f} mm, {step_angular:.2f} deg"
                )

        if self.points is not None and self.progress_index is not None:
            lower = self.progress_index - self.max_backtrack_points
            upper = self.progress_index + self.max_forward_jump_points
            if index < lower or index > upper:
                raise SafetyStop(
                    f"trajectory progress jump {self.progress_index} -> {index} "
                    f"outside [{lower}, {upper}]"
                )
        if self.points is not None:
            self.progress_index = max(index, self.progress_index or 0)
        self.last_target = target.copy()
        return {
            "reference_index": index,
            "tube_distance_mm": tube_linear,
            "tube_angle_deg": tube_angular,
            "tracking_error_mm": tracking_linear,
            "tracking_angle_deg": tracking_angular,
        }


class RealSenseSource:
    def __init__(self, width, height, fps, image_size, serial=""):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.image_size = int(image_size)
        self.serial = serial
        self.lock = threading.Lock()
        self.frame = None
        self.frame_monotonic = None
        self.error = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="realsense-live", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15.0):
            raise RuntimeError("RealSense startup timed out")
        if self.error:
            raise RuntimeError(f"RealSense failed: {self.error}")

    def _run(self):
        pipeline = None
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.rgb8,
                self.fps,
            )
            pipeline.start(config)
            for _ in range(15):
                pipeline.wait_for_frames(5000)
            self.ready.set()
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                captured = time.monotonic()
                image = np.asanyarray(color.get_data())
                resized = cv2.resize(
                    image,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                with self.lock:
                    self.frame = np.ascontiguousarray(resized)
                    self.frame_monotonic = captured
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def latest(self, max_age_s):
        if self.error:
            raise SafetyStop(f"RealSense thread failed: {self.error}")
        with self.lock:
            if self.frame is None:
                raise SafetyStop("no RealSense RGB frame")
            frame = self.frame.copy()
            timestamp = self.frame_monotonic
        age = time.monotonic() - timestamp
        if age > max_age_s:
            raise SafetyStop(f"RealSense frame is stale: {age:.3f}s")
        return frame, age

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)


def _last_complete_jsonl_record(path, tail_bytes=262144):
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if end <= 0:
            raise SafetyStop("ultrasound index is empty")
        start = max(0, end - tail_bytes)
        handle.seek(start)
        data = handle.read()
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    for line in reversed(lines):
        try:
            return json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise SafetyStop("no complete ultrasound index record")


class UltrasoundSource:
    def __init__(self, live_dir, settings_metadata, image_size):
        self.live_dir = Path(live_dir).resolve()
        self.index_path = self.live_dir / "ultrasound_frames.jsonl"
        self.packet_path = self.live_dir / "ultrasound_packets.bin"
        metadata = read_json(settings_metadata)
        settings = metadata.get("applied_settings")
        if settings is None:
            settings = metadata.get("ultrasound_preset", {}).get("requested_settings")
        if settings is None:
            raise ValueError("settings metadata has no applied/requested ultrasound settings")
        self.settings = settings
        self.beam_count = calculate_beam_count(settings)
        self.display_max = float(settings["basic"]["display_max_amplitude_percent"])
        self.palette_bgr = load_phaselink_palette(DEFAULT_PALETTE_SOURCE)
        self.image_size = int(image_size)
        self.last_packet_index = None
        self.last_image = None
        self.last_host_unix_ns = None

    def latest(self, max_age_s):
        if not self.index_path.is_file() or not self.packet_path.is_file():
            raise SafetyStop(
                "PhaseLink live files are missing; start PhaseLink with "
                f"PHASELINK_VLA_DIR={self.live_dir}"
            )
        record = _last_complete_jsonl_record(self.index_path)
        packet_index = int(record["packet_index"])
        host_unix_ns = int(record["host_unix_ns"])
        age = (time.time_ns() - host_unix_ns) / 1e9
        if age < -1.0 or age > max_age_s:
            raise SafetyStop(f"ultrasound packet is stale: {age:.3f}s")
        if packet_index != self.last_packet_index:
            offset = int(record["binary_offset"])
            size = int(record["binary_size"])
            with open(self.packet_path, "rb") as packets:
                packets.seek(offset)
                packet = packets.read(size)
            if len(packet) != size:
                raise SafetyStop(
                    f"ultrasound packet is incomplete: {len(packet)}/{size} bytes"
                )
            gray = decode_s_scan(packet, self.beam_count, self.display_max)
            display = cv2.resize(
                gray,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            bgr = self.palette_bgr[display]
            self.last_image = np.ascontiguousarray(bgr[:, :, ::-1])
            self.last_packet_index = packet_index
            self.last_host_unix_ns = host_unix_ns
        return self.last_image.copy(), age, packet_index


class XArmInterface:
    def __init__(self, ip, report_type, max_force_n, max_torque_nm, allow_zero_ft):
        self.ip = ip
        self.report_type = report_type
        self.max_force_n = float(max_force_n)
        self.max_torque_nm = float(max_torque_nm)
        self.allow_zero_ft = bool(allow_zero_ft)
        self.arm = None
        self.lock = threading.Lock()
        self.latest_report = None
        self.latest_report_monotonic = None
        self.executing = False

    def connect(self):
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(
            self.ip,
            enable_report=True,
            report_type=self.report_type,
            is_radian=False,
        )
        if not self.arm.connected:
            raise RuntimeError(f"could not connect to xArm at {self.ip}")
        if not self.arm.register_report_callback(self._on_report):
            raise RuntimeError("xArm report callback registration failed")
        deadline = time.monotonic() + 5.0
        while self.latest_report is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.latest_report is None:
            raise RuntimeError("no xArm report received")

    def _on_report(self, report):
        record = {
            "pose": report.get("cartesian"),
            "state": report.get("state"),
            "mode": report.get("mode"),
            "error_code": report.get("error_code"),
            "warn_code": report.get("warn_code"),
            "ft_ext_force": report.get("ft_ext_force"),
            "ft_raw_force": report.get("ft_raw_force"),
        }
        with self.lock:
            self.latest_report = record
            self.latest_report_monotonic = time.monotonic()

    def snapshot(self, max_age_s=0.25):
        with self.lock:
            report = dict(self.latest_report or {})
            timestamp = self.latest_report_monotonic
        if not report or timestamp is None:
            raise SafetyStop("no xArm report")
        age = time.monotonic() - timestamp
        if age > max_age_s:
            raise SafetyStop(f"xArm report is stale: {age:.3f}s")
        pose = finite_pose(report.get("pose"), "xArm reported TCP")
        report["pose"] = pose
        report["age_s"] = age
        return report

    def _require_healthy(self, require_ft, allow_state5=False):
        report = self.snapshot()
        if not self.arm.connected:
            raise SafetyStop("xArm disconnected")
        if int(report.get("error_code") or 0) != 0:
            raise SafetyStop(f"xArm error_code={report.get('error_code')}")
        if int(report.get("warn_code") or 0) != 0:
            raise SafetyStop(f"xArm warn_code={report.get('warn_code')}")
        state = report.get("state")
        if state in (4, 6) or (state == 5 and not allow_state5):
            raise SafetyStop(f"xArm unsafe state={report.get('state')}")
        force = report.get("ft_ext_force")
        if require_ft:
            force = np.asarray(force, dtype=np.float64).reshape(-1)
            if force.shape != (6,) or not np.all(np.isfinite(force)):
                raise SafetyStop("force/torque report is unavailable")
            if not self.allow_zero_ft and np.max(np.abs(force)) < 1e-6:
                raise SafetyStop("force/torque report is all zero")
            force_norm = float(np.linalg.norm(force[:3]))
            torque_norm = float(np.linalg.norm(force[3:]))
            if force_norm > self.max_force_n:
                raise SafetyStop(
                    f"external force {force_norm:.2f} N exceeds {self.max_force_n:.2f} N"
                )
            if torque_norm > self.max_torque_nm:
                raise SafetyStop(
                    f"external torque {torque_norm:.2f} Nm exceeds {self.max_torque_nm:.2f} Nm"
                )
        return report

    def prepare_execute(self, envelope):
        # State 5 is the controller's MODE_CHANGED/Stop feedback.  It is safe
        # to inspect here because no target has been sent yet; set_mode(1)
        # followed by set_state(0) below must leave it before execution starts.
        report = self._require_healthy(require_ft=True, allow_state5=True)
        envelope.require_at_start(report["pose"])
        config_code, _ = self.arm.get_ft_sensor_config()
        data_code, data = self.arm.get_ft_sensor_data(is_raw=False)
        if config_code != 0 or data_code != 0:
            raise SafetyStop(
                f"force sensor check failed: config_code={config_code}, data_code={data_code}"
            )
        if not self.allow_zero_ft and np.max(np.abs(np.asarray(data, dtype=float))) < 1e-6:
            raise SafetyStop("direct force sensor read is all zero")
        code = self.arm.motion_enable(True)
        if code != 0:
            raise SafetyStop(f"motion_enable failed, code={code}")
        # From this point onward any setup failure must take the emergency-stop
        # cleanup path; motors have been enabled even if servo mode is not ready.
        self.executing = True
        for label, code in (
            ("set_mode(1)", self.arm.set_mode(1)),
            ("set_state(0)", self.arm.set_state(0)),
        ):
            if code != 0:
                raise SafetyStop(f"{label} failed, code={code}")
        time.sleep(0.2)
        self._require_healthy(require_ft=True)

    def send_target(self, target, envelope, speed, acceleration, period):
        report = self._require_healthy(require_ft=True)
        safety = envelope.validate(target, report["pose"])
        before_ns = time.time_ns()
        code = self.arm.set_servo_cartesian(
            target.tolist(),
            speed=float(speed),
            mvacc=float(acceleration),
            mvtime=float(period),
            is_radian=False,
        )
        after_ns = time.time_ns()
        if code != 0:
            raise SafetyStop(f"set_servo_cartesian failed, code={code}")
        return report, safety, code, before_ns, after_ns

    def stop_motion(self, emergency):
        if self.arm is None:
            return
        try:
            if emergency and self.executing:
                self.arm.set_state(4)
                self.arm.set_mode(0)
            elif self.executing:
                self.arm.set_mode(0)
                self.arm.set_state(0)
        finally:
            self.executing = False

    def close(self):
        if self.arm is None:
            return
        try:
            self.arm.release_report_callback(self._on_report)
        except Exception:
            pass
        try:
            self.arm.disconnect()
        except Exception:
            pass


class RealRobotClient(transport.LeanRobotClient):
    def __init__(
        self, args, config, robot, camera, ultrasound, envelope, rate_limiter, log_handle
    ):
        super().__init__(args, config)
        self.robot = robot
        self.camera = camera
        self.ultrasound = ultrasound
        self.envelope = envelope
        self.rate_limiter = rate_limiter
        self.log_handle = log_handle
        self.start_monotonic = None
        self.last_action_monotonic = None
        self.action_count = 0
        self.safety_error = None

    def get_observation(self):
        # Shadow mode is read-only.  During execute startup the first
        # observation is also collected before prepare_execute() deliberately
        # changes mode/state, so state 5 may be observed only while no command
        # execution has begun.
        report = self.robot._require_healthy(
            require_ft=self.args.execute,
            allow_state5=not self.robot.executing,
        )
        rgb, rgb_age = self.camera.latest(self.args.sensor_max_age)
        us, us_age, packet_index = self.ultrasound.latest(self.args.sensor_max_age)
        pose = report["pose"]
        observation = {name: float(pose[i]) for i, name in enumerate(transport.DIMS)}
        observation[self.b.cam_base] = rgb
        observation[self.b.cam_wrist] = us
        self.observed_frames.append(self.action_count)
        self._log(
            {
                "type": "observation",
                "host_unix_ns": time.time_ns(),
                "tcp_pose": pose.tolist(),
                "rgb_age_s": rgb_age,
                "ultrasound_age_s": us_age,
                "ultrasound_packet_index": packet_index,
                "ft_ext_force": report.get("ft_ext_force"),
            }
        )
        return observation

    def pop_action(self):
        timed = self._pop_action()
        if timed is None:
            return None
        raw_target = finite_pose(timed.get_action(), "remote policy action")
        with self.latest_action_lock:
            self.latest_action = timed.get_timestep()
        now = time.monotonic()
        limiter_report = self.robot.snapshot()
        command_target, rate_limit = self.rate_limiter.apply(
            raw_target, limiter_report["pose"]
        )
        result = {
            "type": "action",
            "host_unix_ns": time.time_ns(),
            "action_timestep": int(timed.get_timestep()),
            # Keep target_tcp_pose as the raw model output for compatibility
            # with existing log-analysis scripts. commanded_tcp_pose is what
            # is validated and, in execute mode, actually sent to xArm.
            "target_tcp_pose": raw_target.tolist(),
            "raw_policy_target": raw_target.tolist(),
            "commanded_tcp_pose": command_target.tolist(),
            "rate_limit": rate_limit,
            "execute": bool(self.args.execute),
        }
        if self.args.execute:
            report, safety, code, before_ns, after_ns = self.robot.send_target(
                command_target,
                self.envelope,
                self.args.servo_speed,
                self.args.servo_acceleration,
                self.environment_dt,
            )
            result.update(
                {
                    "actual_tcp_pose": report["pose"].tolist(),
                    "safety": safety,
                    "return_code": code,
                    "call_before_unix_ns": before_ns,
                    "call_after_unix_ns": after_ns,
                }
            )
        else:
            report = limiter_report
            try:
                result["safety"] = self.envelope.validate(
                    command_target, report["pose"]
                )
                result["shadow_valid"] = True
            except SafetyStop as exc:
                result["shadow_valid"] = False
                result["shadow_rejection"] = str(exc)
        self.action_count += 1
        self.last_action_monotonic = now
        self._log(result)
        return command_target

    def _log(self, record):
        self.log_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.log_handle.flush()

    def run_control_loop(self):
        self.start_monotonic = time.monotonic()
        self.last_action_monotonic = self.start_monotonic
        next_tick = self.start_monotonic
        try:
            while self.running:
                now = time.monotonic()
                if now - self.start_monotonic >= self.args.max_seconds:
                    break
                if self.observation_error:
                    raise SafetyStop(self.observation_error)
                if self.rx_error:
                    raise SafetyStop(self.rx_error)
                if self.actions_available():
                    self.pop_action()
                elif self.args.execute and self.args.queue_empty_stop > 0:
                    empty_for = now - self.last_action_monotonic
                    if empty_for > self.args.queue_empty_stop:
                        raise SafetyStop(
                            f"action queue empty for {empty_for:.3f}s; no action is repeated"
                        )
                if self._ready_to_send_observation():
                    self.control_loop_observation()
                with self.action_queue_lock:
                    self.action_queue_size.append(self.action_queue.qsize())
                next_tick += self.environment_dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self.safety_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.shutdown_event.set()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="100.64.0.1:8080")
    parser.add_argument("--robot-ip", default="192.168.1.100")
    parser.add_argument("--robot-report-type", choices=("real", "rich"), default="real")
    parser.add_argument("--ckpt", default=transport.DEFAULT_CKPT)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.9)
    parser.add_argument("--grpc-compression", choices=("none", "gzip"), default="gzip")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--initial-action-timeout", type=float, default=180.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--realsense-width", type=int, default=1280)
    parser.add_argument("--realsense-height", type=int, default=720)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--realsense-serial", default="")
    parser.add_argument(
        "--ultrasound-live-dir",
        default=str(DEFAULT_ULTRASOUND_LIVE_DIR),
        help=(
            "PhaseLink live packet directory; defaults to the fixed path "
            "<deploy-dir>\\live_ultrasound_current"
        ),
    )
    parser.add_argument(
        "--ultrasound-settings-metadata",
        default=str(DEFAULT_ULTRASOUND_SETTINGS_METADATA),
        help=(
            "ultrasound decoding metadata; defaults to the fixed training "
            "configuration copied into <deploy-dir>\\config"
        ),
    )
    parser.add_argument(
        "--safety-trajectory",
        default=None,
        help=(
            "optional demonstrated trajectory used as a spatial safety tube; "
            "omit it to execute direct policy targets with only live safety limits"
        ),
    )
    parser.add_argument("--sensor-max-age", type=float, default=0.5)
    parser.add_argument(
        "--queue-empty-stop",
        type=float,
        default=0.0,
        help=(
            "stop after this many seconds with an empty action queue; "
            "0 disables this timeout. Empty queues never repeat old actions"
        ),
    )
    parser.add_argument("--servo-speed", type=float, default=10.0)
    parser.add_argument("--servo-acceleration", type=float, default=500.0)
    parser.add_argument(
        "--rate-limited-policy",
        action="store_true",
        help=(
            "opt in to continuous local pose steps toward distant absolute policy "
            "targets; direct policy targets remain the default"
        ),
    )
    parser.add_argument(
        "--policy-linear-speed",
        type=float,
        default=2.0,
        help="translation slew rate in mm/s when --rate-limited-policy is enabled",
    )
    parser.add_argument(
        "--policy-angular-speed",
        type=float,
        default=1.0,
        help="RPY vector slew rate in deg/s when --rate-limited-policy is enabled",
    )
    parser.add_argument("--max-force-n", type=float, default=40.0)
    parser.add_argument("--max-torque-nm", type=float, default=5.0)
    parser.add_argument("--allow-zero-ft", action="store_true")
    parser.add_argument("--trajectory-tube-mm", type=float, default=15.0)
    parser.add_argument("--trajectory-tube-deg", type=float, default=8.0)
    parser.add_argument("--max-action-step-mm", type=float, default=3.0)
    parser.add_argument("--max-action-step-deg", type=float, default=2.0)
    parser.add_argument("--max-tracking-error-mm", type=float, default=12.0)
    parser.add_argument("--max-tracking-error-deg", type=float, default=5.0)
    parser.add_argument("--start-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--start-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--max-backtrack-points", type=int, default=20)
    parser.add_argument("--max-forward-jump-points", type=int, default=100)
    parser.add_argument("--log-root", default=str(HERE / "real_runs"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate files, imports and safety configuration; do not connect devices/server",
    )
    args = parser.parse_args()
    if not 0 <= args.chunk_size_threshold <= 1:
        parser.error("--chunk-size-threshold must be in [0,1]")
    if args.fps <= 0 or args.max_seconds <= 0:
        parser.error("--fps and --max-seconds must be positive")
    if args.queue_empty_stop < 0:
        parser.error("--queue-empty-stop must be >= 0")
    if args.policy_linear_speed <= 0 or args.policy_angular_speed <= 0:
        parser.error("--policy-linear-speed and --policy-angular-speed must be positive")
    if args.execute and args.confirm != EXECUTE_CONFIRMATION:
        parser.error(
            f"real execution requires --confirm {EXECUTE_CONFIRMATION}"
        )
    return args


def main():
    args = parse_args()
    transport.check_stubs_work()
    config = LivePolicyConfig(img_size=args.image_size)
    envelope = SafetyEnvelope(
        args.safety_trajectory,
        args.trajectory_tube_mm,
        args.trajectory_tube_deg,
        args.max_action_step_mm,
        args.max_action_step_deg,
        args.max_tracking_error_mm,
        args.max_tracking_error_deg,
        args.start_tolerance_mm,
        args.start_tolerance_deg,
        args.max_backtrack_points,
        args.max_forward_jump_points,
    )
    rate_limiter = PolicyTargetRateLimiter(
        args.rate_limited_policy,
        args.fps,
        args.policy_linear_speed,
        args.policy_angular_speed,
    )
    ultrasound = UltrasoundSource(
        args.ultrasound_live_dir,
        args.ultrasound_settings_metadata,
        args.image_size,
    )
    print("Real xArm remote-policy client")
    print("  mode:", "EXECUTE" if args.execute else "SHADOW (no robot commands)")
    print("  robot:", args.robot_ip)
    print("  server:", args.server)
    print("  image contract:", f"{args.image_size}x{args.image_size} RGB uint8")
    print(
        "  trajectory constraint:",
        envelope.trajectory_path or "DISABLED (direct policy targets)",
    )
    if rate_limiter.enabled:
        print(
            "  policy target mode: RATE LIMITED "
            f"({rate_limiter.linear_speed_mm_s:g} mm/s, "
            f"{rate_limiter.angular_speed_deg_s:g} deg/s)"
        )
    else:
        print("  policy target mode: DIRECT (existing default)")
    print("  PhaseLink live dir:", ultrasound.live_dir)
    print("  max duration:", args.max_seconds, "s")
    if args.validate_only:
        print("Validation OK; no device or server connection was made.")
        return

    run_dir = Path(args.log_root) / time.strftime("real_policy_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "live_policy_records.jsonl"
    camera = RealSenseSource(
        args.realsense_width,
        args.realsense_height,
        args.realsense_fps,
        args.image_size,
        args.realsense_serial,
    )
    robot = XArmInterface(
        args.robot_ip,
        args.robot_report_type,
        args.max_force_n,
        args.max_torque_nm,
        args.allow_zero_ft,
    )
    client = None
    rx_thread = None
    tx_thread = None
    emergency = False
    try:
        camera.start()
        robot.connect()
        # Verify every live observation source before loading the remote model.
        robot.snapshot()
        camera.latest(args.sensor_max_age)
        ultrasound.latest(args.sensor_max_age)
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            client = RealRobotClient(
                args,
                config,
                robot,
                camera,
                ultrasound,
                envelope,
                rate_limiter,
                log_handle,
            )
            print("Connecting remote policy; model loading may take about 105 seconds...")
            if not client.start():
                raise RuntimeError(f"cannot connect remote policy at {args.server}")
            rx_thread = threading.Thread(target=client.receive_actions, daemon=True)
            tx_thread = threading.Thread(target=client.send_observations_loop, daemon=True)
            rx_thread.start()
            tx_thread.start()
            if not client.control_loop_observation(force_must_go=True):
                raise RuntimeError("initial live observation was not queued")
            if not client.first_actions_event.wait(args.initial_action_timeout):
                raise SafetyStop("timed out waiting for first action chunk")
            print("First action chunk ready:", client.action_queue.qsize(), "actions")
            if args.execute:
                print("EXECUTE requested. Robot will NOT be moved to a start pose automatically.")
                print("Checking force sensor, robot errors, and live action limits...")
                robot.prepare_execute(envelope)
                print("Cartesian servo enabled; starting bounded real-robot test.")
            else:
                print("Starting shadow test; predicted actions will not be sent to xArm.")
            client.run_control_loop()
            print("Run complete. actions processed:", client.action_count)
            if client.observation_send_latencies:
                values = client.observation_send_latencies
                print(
                    "Observation RPC latency: avg={:.3f}s max={:.3f}s".format(
                        sum(values) / len(values), max(values)
                    )
                )
    except KeyboardInterrupt:
        emergency = bool(args.execute)
        print("Interrupted by user.")
    except Exception as exc:
        emergency = bool(args.execute)
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        if client is not None:
            client.stop()
        robot.stop_motion(emergency=emergency)
        if rx_thread is not None:
            rx_thread.join(timeout=2.0)
        if tx_thread is not None:
            tx_thread.join(timeout=2.0)
        camera.stop()
        robot.close()
        print("xArm command mode restored/stopped; log:", log_path)


if __name__ == "__main__":
    main()
