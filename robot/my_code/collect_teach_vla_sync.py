#!/usr/bin/env python3
"""
Collect xArm teach-mode state, RealSense RGB-D, and PhaseLink ultrasound.

The three devices are recorded independently at their native rates. Every
record receives a host clock timestamp, and align_vla_episode.py builds the
fixed-rate synchronized samples after capture.

PhaseLink integration:
  Rebuild the modified Qt project once. This script launches Phaselink.exe
  with PHASELINK_VLA_DIR pointing at the current episode directory. The Qt
  client then writes ultrasound_packets.bin + ultrasound_frames.jsonl.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xarm.wrapper import XArmAPI  # noqa: E402
from realsense_sync_recorder import RealSenseRecorder, ensure_dir  # noqa: E402


DEFAULT_ROBOT_IP = "192.168.1.100"
DEFAULT_OUTPUT_ROOT = os.path.join(CURRENT_DIR, "teach_vla_runs")
LEGACY_PHASELINK_EXE = os.path.join(
    WORKSPACE_DIR,
    "超声设备",
    "cs_phaselink",
    "Phaselink",
    "build_debug",
    "debug",
    "Phaselink.exe",
)
# The source tree in this workspace is PAU_SDK - dt/Phaselink. Prefer the
# freshly generated out-of-source Release executable.
DEFAULT_PHASELINK_EXE = os.path.join(
    WORKSPACE_DIR,
    "PAU_SDK - dt",
    "build-Phaselink-Desktop_Qt_5_12_10_MSVC2015_64bit-Release",
    "release",
    "Phaselink.exe",
)
DEFAULT_PHASELINK_SERVER_EXE = os.path.join(
    WORKSPACE_DIR,
    "PAU_SDK - dt",
    "phaselink_server",
    "phaselink_server.exe",
)
TEACH_SENSITIVITY = 5
TCP_OFFSET = [0, 0, 0, 0, 0, 0]
QUEUE_CAPACITY = 4096


def json_line(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def safe_list(value):
    if value is None:
        return None
    try:
        return list(value)
    except (TypeError, ValueError):
        return value


def safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def make_episode_id():
    return time.strftime("teach_vla_%Y%m%d_%H%M%S")


class RobotReportRecorder:
    """Keep the SDK report callback short and write JSONL on another thread."""

    _STOP = object()

    def __init__(self, arm, output_path):
        self.arm = arm
        self.output_path = output_path
        self.records = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.writer_thread = threading.Thread(target=self._writer, daemon=True)
        self.started_event = threading.Event()
        self.record_count = 0
        self.received_count = 0
        self.dropped_count = 0
        self.writer_error = None
        self._callback_registered = False

    def start(self):
        self.writer_thread.start()
        self.started_event.wait(timeout=5)
        if not self.started_event.is_set():
            raise RuntimeError("xArm record writer startup timed out")
        if self.writer_error is not None:
            raise RuntimeError(
                "xArm record writer failed: {}".format(self.writer_error)
            )
        registered = self.arm.register_report_callback(self._on_report)
        if not registered:
            self.records.put(self._STOP)
            self.writer_thread.join(timeout=5)
            raise RuntimeError("xArm report callback registration failed")
        self._callback_registered = True

    def _on_report(self, report):
        # Prefer the timestamp captured in the SDK report parser before its
        # worker-pool callback scheduling. Fall back for an unmodified SDK.
        host_monotonic_ns = report.get("sdk_report_host_monotonic_ns")
        host_unix_ns = report.get("sdk_report_host_unix_ns")
        if host_monotonic_ns is None:
            host_monotonic_ns = time.monotonic_ns()
        if host_unix_ns is None:
            host_unix_ns = time.time_ns()
        report_data = safe_attr(self.arm, "report_data")
        record_index = self.received_count
        self.received_count += 1

        record = {
            "schema": "xarm_report_v2",
            "record_index": record_index,
            "host_unix_ns": host_unix_ns,
            "host_monotonic_ns": host_monotonic_ns,
            "controller_timestamp": safe_attr(report_data, "timestamp"),
            "controller_report_counter": safe_attr(report_data, "counter"),
            "tcp_pose": safe_list(report.get("cartesian")),
            "joint_angles": safe_list(report.get("joints")),
            "joint_velocity": safe_list(
                safe_attr(report_data, "actual_joint_velocity")
            ),
            "tcp_speed": safe_list(safe_attr(report_data, "actual_tcp_speed")),
            "joint_torque": safe_list(report.get("joint_torque")),
            "ft_ext_force": safe_list(report.get("ft_ext_force")),
            "ft_raw_force": safe_list(report.get("ft_raw_force")),
            "realtime_tcp_speed": report.get("realtime_tcp_speed"),
            "realtime_joint_speeds": safe_list(
                report.get("realtime_joint_speeds")
            ),
            "state": report.get("state"),
            "mode": report.get("mode"),
            "cmd_num": report.get("cmdnum"),
            "error_code": report.get("error_code"),
            "warn_code": report.get("warn_code"),
        }
        try:
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped_count += 1

    def _writer(self):
        try:
            with open(self.output_path, "w", encoding="utf-8", buffering=1) as handle:
                self.started_event.set()
                while True:
                    record = self.records.get()
                    if record is self._STOP:
                        return
                    handle.write(json_line(record))
                    self.record_count += 1
        except Exception as exc:
            self.writer_error = exc
            self.started_event.set()

    def stop(self):
        if self._callback_registered:
            self.arm.release_report_callback(self._on_report)
            self._callback_registered = False
        while self.writer_thread.is_alive():
            try:
                self.records.put(self._STOP, timeout=0.5)
                break
            except queue.Full:
                continue
        self.writer_thread.join(timeout=10)
        if self.writer_thread.is_alive():
            raise RuntimeError("xArm record writer did not stop")
        if self.writer_error is not None:
            raise RuntimeError("xArm record writer failed: {}".format(self.writer_error))


def phaselink_server_port_pids():
    """Return PIDs that own both PhaseLink UDP server ports."""
    if os.name != "nt":
        return set()
    result = subprocess.run(
        ["netstat", "-ano", "-p", "udp"],
        capture_output=True,
        text=True,
        check=False,
    )
    ports_by_pid = {}
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) < 4 or columns[0].upper() != "UDP":
            continue
        try:
            port = int(columns[1].rsplit(":", 1)[1])
            pid = int(columns[-1])
        except (IndexError, ValueError):
            continue
        if port in (10001, 20001):
            ports_by_pid.setdefault(pid, set()).add(port)
    return {
        pid for pid, ports in ports_by_pid.items()
        if ports == {10001, 20001}
    }


def ensure_phaselink_server(
    server_executable=DEFAULT_PHASELINK_SERVER_EXE,
    visible_console=True,
):
    """Start the local acquisition server unless both UDP ports are owned."""
    if os.name != "nt":
        return None
    existing_pids = phaselink_server_port_pids()
    if existing_pids:
        print(
            "PhaseLink server already owns UDP 10001/20001; PID(s): {}".format(
                ", ".join(str(pid) for pid in sorted(existing_pids))
            )
        )
        return None
    if not os.path.isfile(server_executable):
        raise FileNotFoundError(
            "PhaseLink server is not running and was not found: {}".format(
                server_executable
            )
        )

    popen_kwargs = {
        "cwd": os.path.dirname(server_executable),
    }
    if visible_console:
        # The operator explicitly needs to see that the hardware server is
        # alive during a contact scan. Give it its own console instead of
        # silently hiding it with CREATE_NO_WINDOW.
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_CONSOLE", 0
        )
    else:
        popen_kwargs.update(
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
            }
        )
    server_process = subprocess.Popen([server_executable], **popen_kwargs)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise RuntimeError(
                "PhaseLink server exited during startup, code={}".format(
                    server_process.returncode
                )
            )
        if server_process.pid in phaselink_server_port_pids():
            print(
                "PhaseLink server ready: PID {}, UDP 10001/20001, console={}".format(
                    server_process.pid, "visible" if visible_console else "hidden"
                )
            )
            return server_process
        time.sleep(0.2)

    server_process.terminate()
    server_process.wait(timeout=5)
    raise RuntimeError(
        "PhaseLink server started but did not bind UDP ports 10001 and 20001"
    )


def launch_phaselink(executable, episode_dir, preset=None):
    if not executable:
        return None
    if not os.path.isfile(executable):
        raise FileNotFoundError(
            "Phaselink.exe not found: {}. Rebuild Phaselink.pro first or pass "
            "--no-launch-phaselink.".format(executable)
        )
    ensure_phaselink_server(visible_console=True)
    exporter_source = os.path.join(
        WORKSPACE_DIR,
        "超声设备",
        "cs_phaselink",
        "Phaselink",
        "vladatapacketrecorder.cpp",
    )
    if (
        os.path.isfile(exporter_source)
        and os.path.getmtime(executable) < os.path.getmtime(exporter_source)
    ):
        raise RuntimeError(
            "Phaselink.exe is older than vladatapacketrecorder.cpp. Rebuild "
            "Phaselink.pro before collecting, otherwise ultrasound packets "
            "will not be exported."
        )
    phaselink_source_dir = os.path.join(
        WORKSPACE_DIR, "PAU_SDK - dt", "Phaselink"
    )
    exporter_sources = [
        os.path.join(phaselink_source_dir, relative)
        for relative in (
            "vladatapacketrecorder.cpp",
            "vladatapacketrecorder.h",
            "mainwindow.cpp",
            "mainwindow.h",
            os.path.join("UI", "scanning.cpp"),
            os.path.join("UI", "scanning.h"),
            "Phaselink.pro",
        )
    ]
    newer_sources = [
        path
        for path in exporter_sources
        if os.path.isfile(path)
        and os.path.getmtime(executable) < os.path.getmtime(path)
    ]
    if newer_sources:
        raise RuntimeError(
            "Phaselink.exe is older than VLA recorder/integration source: {}. "
            "Re-run qmake and rebuild Phaselink.pro before collecting; the "
            "current executable must not be treated as VLA-enabled.".format(
                ", ".join(os.path.basename(path) for path in newer_sources)
            )
        )
    env = os.environ.copy()
    env["PHASELINK_VLA_DIR"] = episode_dir
    env.setdefault("PHASELINK_SERVER_IP", "127.0.0.1")
    if preset:
        env["PHASELINK_VLA_PRESET"] = preset
    runtime_dirs = [
        os.path.dirname(executable),
        os.path.join(
            WORKSPACE_DIR, "PAU_SDK - dt", "PhaselinkNetSDK", "release", "lib"
        ),
        os.path.join(WORKSPACE_DIR, "PAU_SDK - dt", "EXE_PACK"),
        r"D:\ayiSoftware\Qt\5.12.10\msvc2015_64\bin",
    ]
    existing_runtime_dirs = [path for path in runtime_dirs if os.path.isdir(path)]
    env["PATH"] = os.pathsep.join(
        existing_runtime_dirs + [env.get("PATH", "")]
    )
    qt_plugins = r"D:\ayiSoftware\Qt\5.12.10\msvc2015_64\plugins"
    if os.path.isdir(qt_plugins):
        env.setdefault("QT_PLUGIN_PATH", qt_plugins)
    return subprocess.Popen(
        [executable],
        cwd=os.path.dirname(executable),
        env=env,
    )


def prepare_teach_mode(arm, sensitivity):
    if arm.warn_code:
        print("clean_warn:", arm.clean_warn())
    if arm.error_code:
        print("clean_error:", arm.clean_error())
    print("motion_enable:", arm.motion_enable(True))
    print("set_mode 0:", arm.set_mode(0))
    print("set_state 0:", arm.set_state(0))
    time.sleep(0.5)
    print("set_tcp_offset:", arm.set_tcp_offset(TCP_OFFSET, wait=True))
    print("set_mode 2:", arm.set_mode(2))
    print("set_state 0:", arm.set_state(0))
    print("set_teach_sensitivity:", arm.set_teach_sensitivity(sensitivity))
    time.sleep(1.0)
    if arm.mode != 2 or arm.state != 0 or arm.error_code != 0:
        raise RuntimeError(
            "Robot did not enter teach mode: mode={}, state={}, error={}".format(
                arm.mode, arm.state, arm.error_code
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--teach-sensitivity", type=int, default=TEACH_SENSITIVITY)
    parser.add_argument("--robot-report-type", choices=("real", "rich"), default="real",
                        help="'real' is preferred for high-rate motion capture.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Capture seconds; 0 means press Enter to stop.")
    parser.add_argument("--no-realsense", action="store_true")
    parser.add_argument("--no-teach-mode", action="store_true",
                        help="Record without changing the robot mode.")
    parser.add_argument("--phaselink-exe", default=DEFAULT_PHASELINK_EXE)
    parser.add_argument("--no-launch-phaselink", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    episode_id = make_episode_id()
    episode_dir = os.path.abspath(os.path.join(args.output_root, episode_id))
    ensure_dir(episode_dir)

    print("Synchronized teach-mode VLA capture")
    print("  episode:", episode_id)
    print("  output:", episode_dir)
    print("  emergency stop must remain reachable")

    phaselink_process = None
    camera = None
    camera_stop = threading.Event()
    arm = None
    robot_recorder = None
    window_start_unix_ns = None
    window_end_unix_ns = None

    metadata = {
        "schema": "xarm_realsense_phaselink_teach_episode_v1",
        "episode_id": episode_id,
        "created_at_unix_ns": time.time_ns(),
        "robot_ip": args.robot_ip,
        "teach_mode_requested": not args.no_teach_mode,
        "teach_sensitivity": args.teach_sensitivity,
        "robot_report_type": args.robot_report_type,
        "tcp_offset": TCP_OFFSET,
        "clock_policy": {
            "alignment_clock": "host_unix_ns",
            "host_monotonic_recorded": True,
            "xarm_host_timestamp": "sdk_parser_before_callback_dispatch",
            "ultrasound_device_clock_hz": 200000000,
        },
        "records": {
            "robot": "robot_records.jsonl",
            "realsense": None if args.no_realsense else "realsense_frames.jsonl",
            "ultrasound": "ultrasound_frames.jsonl",
            "ultrasound_packets": "ultrasound_packets.bin",
        },
    }
    write_json(os.path.join(episode_dir, "metadata.json"), metadata)

    try:
        if not args.no_launch_phaselink:
            print("Launching PhaseLink with VLA packet export enabled...")
            phaselink_process = launch_phaselink(args.phaselink_exe, episode_dir)
        else:
            print(
                "PhaseLink auto-launch disabled. Its process must have "
                "PHASELINK_VLA_DIR={} set before startup.".format(episode_dir)
            )

        if not args.no_realsense:
            print("Starting RealSense...")
            camera = RealSenseRecorder(episode_dir, camera_stop)
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

        if not args.no_teach_mode:
            prepare_teach_mode(arm, args.teach_sensitivity)

        input(
            "\nIn PhaseLink: connect the device and click Start scan. "
            "Then press Enter here to start the synchronized window..."
        )
        window_start_unix_ns = time.time_ns()
        start_monotonic_ns = time.monotonic_ns()
        print("CAPTURE STARTED - move the robot by hand.")

        if args.duration > 0:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                time.sleep(min(0.2, deadline - time.monotonic()))
        else:
            input("Press Enter to stop capture...\n")

        window_end_unix_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        print("CAPTURE STOPPED")

        metadata["capture_window"] = {
            "start_unix_ns": window_start_unix_ns,
            "end_unix_ns": window_end_unix_ns,
            "start_monotonic_ns": start_monotonic_ns,
            "end_monotonic_ns": end_monotonic_ns,
            "duration_s": (end_monotonic_ns - start_monotonic_ns) / 1e9,
        }
    except KeyboardInterrupt:
        window_end_unix_ns = time.time_ns()
        print("\nCapture interrupted; closing recorders safely.")
    finally:
        if robot_recorder is not None:
            try:
                robot_recorder.stop()
                metadata["robot_record_count"] = robot_recorder.record_count
                metadata["robot_dropped_count"] = robot_recorder.dropped_count
            except Exception as exc:
                metadata["robot_stop_error"] = str(exc)

        if arm is not None:
            try:
                if not args.no_teach_mode:
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
            if camera.error is not None:
                metadata["realsense_error"] = str(camera.error)

        if window_start_unix_ns is not None and window_end_unix_ns is not None:
            metadata.setdefault("capture_window", {
                "start_unix_ns": window_start_unix_ns,
                "end_unix_ns": window_end_unix_ns,
            })
        write_json(os.path.join(episode_dir, "metadata.json"), metadata)

    if phaselink_process is not None and phaselink_process.poll() is None:
        print(
            "\nIn PhaseLink: click Stop scan and close the window so its final "
            "ultrasound metadata is flushed. The script will not force-kill it."
        )

    print("\nRaw capture saved:", episode_dir)
    print(
        "Next: python align_vla_episode.py --episode \"{}\" --fps 15".format(
            episode_dir
        )
    )


if __name__ == "__main__":
    main()
