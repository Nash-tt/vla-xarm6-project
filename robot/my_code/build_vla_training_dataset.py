#!/usr/bin/env python3
"""Build a portable, fixed-rate VLA training folder from one collection episode.

The source episode remains unchanged except that alignment files may be generated
when they are missing or use a different requested rate.  Only aligned samples
whose ``quality.valid`` flag is true are copied into the training dataset.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import cv2
import numpy as np

from export_ultrasound_images import (
    calculate_beam_count,
    decode_s_scan,
    load_phaselink_palette,
    resolve_ultrasound_settings,
    DEFAULT_PALETTE_SOURCE,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ALIGN_SCRIPT = SCRIPT_DIR / "align_vla_episode.py"
REQUIRED_SOURCE_FILES = (
    "metadata.json",
    "trajectory.json",
    "robot_records.jsonl",
    "robot_actions.jsonl",
    "realsense_metadata.json",
    "realsense_frames.jsonl",
    "ultrasound_metadata.json",
    "ultrasound_frames.jsonl",
    "ultrasound_packets.bin",
)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid JSON at {} line {}: {}".format(
                            path, line_number, exc
                        )
                    ) from exc


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def relative_path(*parts):
    return "/".join(str(part).replace("\\", "/").strip("/") for part in parts)


def validate_episode(episode_dir):
    missing = [name for name in REQUIRED_SOURCE_FILES if not (episode_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "episode is missing required files: {}".format(", ".join(missing))
        )


def ensure_alignment(episode_dir, fps, realign):
    alignment_path = episode_dir / "alignment.json"
    samples_path = episode_dir / "aligned_samples.jsonl"
    needs_alignment = realign or not alignment_path.is_file() or not samples_path.is_file()
    if not needs_alignment:
        alignment = read_json(alignment_path)
        needs_alignment = abs(float(alignment.get("fps", -1)) - fps) > 1e-9
    if needs_alignment:
        command = [
            sys.executable,
            str(ALIGN_SCRIPT),
            "--episode",
            str(episode_dir),
            "--fps",
            str(fps),
        ]
        print("Running alignment: {} Hz".format(fps))
        subprocess.run(command, check=True)
    return read_json(alignment_path), samples_path


def finite_vector(value, minimum_length):
    if not isinstance(value, list) or len(value) < minimum_length:
        return None
    result = [float(item) for item in value[:minimum_length]]
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def audit_force(samples):
    ext_values = []
    raw_values = []
    missing = 0
    for sample in samples:
        robot = sample["observation"]["robot"]
        ext = finite_vector(robot.get("ft_ext_force"), 6)
        raw = finite_vector(robot.get("ft_raw_force"), 6)
        if ext is None or raw is None:
            missing += 1
            continue
        ext_values.append(ext)
        raw_values.append(raw)

    if not ext_values or missing:
        return {
            "valid": False,
            "reason": "force channels are missing or non-finite in aligned samples",
            "complete_sample_count": len(ext_values),
            "missing_sample_count": missing,
        }

    ext_array = np.asarray(ext_values, dtype=np.float64)
    raw_array = np.asarray(raw_values, dtype=np.float64)
    maximum = float(max(np.max(np.abs(ext_array)), np.max(np.abs(raw_array))))
    variation = float(max(np.max(np.ptp(ext_array, axis=0)), np.max(np.ptp(raw_array, axis=0))))
    valid = maximum > 1e-6 and variation > 1e-6
    return {
        "valid": valid,
        "reason": (
            "force channels contain finite, non-constant measurements"
            if valid
            else "all recorded force channels are zero or constant; force is omitted from training observations"
        ),
        "complete_sample_count": len(ext_values),
        "missing_sample_count": missing,
        "maximum_absolute_value": maximum,
        "maximum_axis_range": variation,
    }


def wrapped_angle_delta(target_deg, current_deg):
    return (float(target_deg) - float(current_deg) + 180.0) % 360.0 - 180.0


def action_delta(target_pose, current_pose):
    return [
        float(target_pose[index]) - float(current_pose[index])
        for index in range(3)
    ] + [
        wrapped_angle_delta(target_pose[index], current_pose[index])
        for index in range(3, 6)
    ]


def require_source_path(episode_dir, relative_source, label):
    if not relative_source:
        raise ValueError("{} path is absent from aligned sample".format(label))
    source = episode_dir / Path(relative_source)
    if not source.is_file():
        raise FileNotFoundError("{} source file not found: {}".format(label, source))
    return source


def make_output_tree(root):
    for name in ("rgb", "depth", "ultrasound", "metadata"):
        (root / name).mkdir(parents=True, exist_ok=True)


def prepare_ultrasound_training_gray(
    gray,
    square_size,
    preserve_physical_aspect,
    depth_span_mm,
    lateral_span_mm,
):
    """Resize decoded ultrasound for training and return its geometry mode."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("decoded ultrasound must be a non-empty 2-D matrix")
    if square_size <= 0:
        raise ValueError("ultrasound square size must be positive")
    if preserve_physical_aspect:
        depth_mm_per_pixel = depth_span_mm / gray.shape[0]
        display_width = max(
            gray.shape[1],
            int(round(lateral_span_mm / depth_mm_per_pixel)),
        )
        return (
            cv2.resize(
                gray,
                (display_width, gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ),
            "equal_physical_mm_per_pixel",
        )
    return (
        cv2.resize(
            gray,
            (square_size, square_size),
            interpolation=cv2.INTER_AREA,
        ),
        "square_model_input_axis_resampling",
    )


def publish_directory(source, target):
    """Publish a completed directory, with a Windows ACL-safe fallback.

    Some Windows data volumes allow creating files and directories but reject
    renaming a populated directory (WinError 5).  In that case copy the
    already-complete tree into a newly created final directory instead.
    """
    try:
        source.replace(target)
        return "atomic_rename", None
    except PermissionError as rename_error:
        if target.exists():
            raise
        try:
            shutil.copytree(source, target, copy_function=shutil.copy2)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            raise
        cleanup_warning = None
        try:
            shutil.rmtree(source)
        except OSError as cleanup_error:
            cleanup_warning = (
                "final dataset is complete, but temporary directory could not "
                "be removed: {} ({})".format(source, cleanup_error)
            )
        return "copy_fallback_after_{}".format(type(rename_error).__name__), cleanup_warning


def build_dataset(
    episode_dir,
    output_dir,
    fps,
    overwrite,
    realign,
    require_force,
    ultrasound_size=256,
    preserve_ultrasound_physical_aspect=False,
):
    validate_episode(episode_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite: {}".format(output_dir)
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output path exists but is not a directory: {}".format(output_dir))
    alignment, aligned_path = ensure_alignment(episode_dir, fps, realign)
    all_samples = list(read_jsonl(aligned_path))
    valid_samples = [
        sample
        for sample in all_samples
        if sample.get("quality", {}).get("valid") is True
        and sample.get("action") is not None
        and sample.get("observation", {}).get("robot") is not None
        and sample.get("observation", {}).get("realsense") is not None
        and sample.get("observation", {}).get("ultrasound") is not None
    ]
    if not valid_samples:
        raise RuntimeError("alignment contains no complete quality.valid samples")

    force_audit = audit_force(valid_samples)
    if require_force and not force_audit["valid"]:
        raise RuntimeError("--require-force failed: {}".format(force_audit["reason"]))

    episode_meta = read_json(episode_dir / "metadata.json")
    ultrasound_meta = read_json(episode_dir / "ultrasound_metadata.json")
    settings, settings_source = resolve_ultrasound_settings(
        str(episode_dir), ultrasound_meta
    )
    beam_count = calculate_beam_count(settings)
    display_max = float(settings["basic"]["display_max_amplitude_percent"])
    range_start = float(settings["basic"]["range_start_mm"])
    range_end = float(settings["basic"]["range_end_mm"])
    pitch_mm = float(settings["basic"]["probe_pitch_mm"])
    element_step = int(settings["array"]["element_step"])
    if range_end <= range_start:
        raise ValueError("invalid ultrasound depth range")
    palette = load_phaselink_palette(DEFAULT_PALETTE_SOURCE)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=str(output_dir.parent))
    )
    backup_dir = None
    publish_method = None
    publish_warning = None
    try:
        make_output_tree(temporary_dir)
        manifest_path = temporary_dir / "samples.jsonl"
        packet_file = episode_dir / ultrasound_meta.get(
            "packet_file", "ultrasound_packets.bin"
        )
        cached_packet_key = None
        cached_ultrasound = None
        ultrasound_shape = None
        ultrasound_native_shape = None
        ultrasound_geometry_mode = None
        rgb_shape = None
        depth_shape = None
        depth_dtype = None

        with packet_file.open("rb") as packets, manifest_path.open(
            "w", encoding="utf-8", buffering=1
        ) as manifest:
            for training_index, sample in enumerate(valid_samples):
                sample_name = "{:06d}".format(training_index)
                realsense = sample["observation"]["realsense"]
                ultrasound = sample["observation"]["ultrasound"]
                robot = sample["observation"]["robot"]
                action = sample["action"]

                color_source = require_source_path(
                    episode_dir, realsense.get("color_path"), "RGB"
                )
                depth_source = require_source_path(
                    episode_dir, realsense.get("depth_path"), "depth"
                )
                color_destination = temporary_dir / "rgb" / (sample_name + ".jpg")
                depth_destination = temporary_dir / "depth" / (sample_name + ".png")
                shutil.copy2(color_source, color_destination)
                shutil.copy2(depth_source, depth_destination)

                if training_index == 0:
                    rgb_check = cv2.imread(str(color_source), cv2.IMREAD_COLOR)
                    depth_check = cv2.imread(str(depth_source), cv2.IMREAD_UNCHANGED)
                    if rgb_check is None or depth_check is None:
                        raise ValueError("OpenCV could not read the first RGB-D pair")
                    if depth_check.dtype != np.uint16:
                        raise ValueError(
                            "depth PNG must be uint16, got {}".format(depth_check.dtype)
                        )
                    rgb_shape = [int(value) for value in rgb_check.shape]
                    depth_shape = [int(value) for value in depth_check.shape]
                    depth_dtype = str(depth_check.dtype)

                offset = int(ultrasound["binary_offset"])
                size = int(ultrasound["binary_size"])
                packet_key = (offset, size)
                if packet_key != cached_packet_key:
                    packets.seek(offset)
                    packet = packets.read(size)
                    if len(packet) != size:
                        raise EOFError(
                            "short ultrasound packet read at offset {}".format(offset)
                        )
                    gray = decode_s_scan(packet, beam_count, display_max)
                    lateral_span_mm = (beam_count - 1) * pitch_mm * element_step
                    display_gray, ultrasound_geometry_mode = (
                        prepare_ultrasound_training_gray(
                            gray,
                            square_size=ultrasound_size,
                            preserve_physical_aspect=(
                                preserve_ultrasound_physical_aspect
                            ),
                            depth_span_mm=range_end - range_start,
                            lateral_span_mm=lateral_span_mm,
                        )
                    )
                    ultrasound_native_shape = [
                        int(gray.shape[0]), int(gray.shape[1])
                    ]
                    cached_ultrasound = palette[display_gray]
                    cached_packet_key = packet_key
                ultrasound_destination = (
                    temporary_dir / "ultrasound" / (sample_name + ".png")
                )
                if not cv2.imwrite(
                    str(ultrasound_destination),
                    cached_ultrasound,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                ):
                    raise OSError("failed to write {}".format(ultrasound_destination))
                ultrasound_shape = [int(value) for value in cached_ultrasound.shape]

                tcp_pose = finite_vector(robot.get("tcp_pose"), 6)
                target_pose = finite_vector(action.get("target_tcp_pose"), 6)
                if tcp_pose is None or target_pose is None:
                    raise ValueError("robot or action TCP pose is invalid")
                robot_record = {
                    "tcp_pose": tcp_pose,
                    "joint_angles": finite_vector(robot.get("joint_angles"), 6),
                    "joint_torque": finite_vector(robot.get("joint_torque"), 6),
                    "state": robot.get("state"),
                    "mode": robot.get("mode"),
                    "error_code": robot.get("error_code"),
                    "warn_code": robot.get("warn_code"),
                    "force_valid": force_audit["valid"],
                }
                if force_audit["valid"]:
                    robot_record["ft_ext_force"] = finite_vector(
                        robot.get("ft_ext_force"), 6
                    )
                    robot_record["ft_raw_force"] = finite_vector(
                        robot.get("ft_raw_force"), 6
                    )

                training_record = {
                    "schema": "xarm_rgbd_ultrasound_vla_training_sample_v1",
                    "episode_id": episode_meta.get("episode_id", episode_dir.name),
                    "training_index": training_index,
                    "source_sample_index": int(sample["sample_index"]),
                    "timestamp_unix_ns": int(sample["timestamp_unix_ns"]),
                    "timestamp_from_start_s": float(sample["timestamp_from_start_s"]),
                    "instruction": sample.get("instruction") or episode_meta.get("instruction"),
                    "observation": {
                        "rgb_path": relative_path("rgb", sample_name + ".jpg"),
                        "depth_path": relative_path("depth", sample_name + ".png"),
                        "ultrasound_path": relative_path(
                            "ultrasound", sample_name + ".png"
                        ),
                        "robot": robot_record,
                    },
                    "action": {
                        "action_index": int(action["action_index"]),
                        "type": action.get("type"),
                        "phase": action.get("phase"),
                        "trajectory_index": action.get("trajectory_index"),
                        "target_tcp_pose": target_pose,
                        "delta_tcp_pose": action_delta(target_pose, tcp_pose),
                        "speed_mm_s": action.get("speed_mm_s", action.get("speed")),
                        "acc_mm_s2": action.get("acc_mm_s2", action.get("acc")),
                        "servo_period_s": action.get(
                            "servo_period_s", action.get("servo_period")
                        ),
                        "return_code": action.get("return_code"),
                    },
                    "alignment_skew_ms": {
                        "robot": sample["quality"].get("robot_skew_ms"),
                        "realsense": sample["quality"].get("realsense_skew_ms"),
                        "ultrasound": sample["quality"].get("ultrasound_skew_ms"),
                        "action": sample["quality"].get("action_skew_ms"),
                    },
                    "source_indices": {
                        "realsense_frame_index": realsense.get("frame_index"),
                        "ultrasound_packet_index": ultrasound.get("packet_index"),
                        "robot_before_index": robot.get("source_before_index"),
                        "robot_after_index": robot.get("source_after_index"),
                    },
                }
                manifest.write(
                    json.dumps(
                        training_record, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )

        metadata_sources = (
            "metadata.json",
            "trajectory.json",
            "alignment.json",
            "realsense_metadata.json",
            "ultrasound_metadata.json",
        )
        for name in metadata_sources:
            shutil.copy2(episode_dir / name, temporary_dir / "metadata" / name)

        settings_are_actual = settings_source == "ultrasound_metadata.applied_settings"
        dataset_info = {
            "schema": "xarm_rgbd_ultrasound_vla_training_dataset_v1",
            "source_episode": str(episode_dir),
            "episode_id": episode_meta.get("episode_id", episode_dir.name),
            "fixed_rate_hz": fps,
            "sample_count": len(valid_samples),
            "source_aligned_sample_count": len(all_samples),
            "filtered_invalid_or_incomplete_count": len(all_samples) - len(valid_samples),
            "manifest": "samples.jsonl",
            "modalities": {
                "rgb": {"directory": "rgb", "shape_hwc": rgb_shape},
                "depth": {
                    "directory": "depth",
                    "shape_hw": depth_shape,
                    "dtype": depth_dtype,
                    "unit": "camera depth units; see metadata/realsense_metadata.json",
                },
                "ultrasound": {
                    "directory": "ultrasound",
                    "shape_hwc": ultrasound_shape,
                    "native_shape_depth_by_beam": ultrasound_native_shape,
                    "representation": (
                        "PhaseLink amplitude-palette scan resampled to a square model input"
                        if not preserve_ultrasound_physical_aspect
                        else "PhaseLink amplitude-palette scan with equal physical mm/pixel axes"
                    ),
                    "geometry_mode": ultrasound_geometry_mode,
                    "square_size_pixels": (
                        ultrasound_size
                        if not preserve_ultrasound_physical_aspect
                        else None
                    ),
                    "depth_range_mm": [range_start, range_end],
                    "beam_count": beam_count,
                    "settings_source": settings_source,
                    "settings_are_actual_readback": settings_are_actual,
                },
                "robot": {
                    "tcp_pose_units": ["mm", "mm", "mm", "deg", "deg", "deg"],
                    "joint_angle_unit": "deg",
                    "active_joint_count": 6,
                    "force": force_audit,
                },
                "action": {
                    "type": (
                        "actual commanded position_target or "
                        "cartesian_servo_target active at sample time"
                    ),
                    "target_tcp_pose_units": ["mm", "mm", "mm", "deg", "deg", "deg"],
                    "delta_tcp_pose_units": ["mm", "mm", "mm", "deg", "deg", "deg"],
                },
            },
            "alignment": {
                "quality_filter": "quality.valid == true and all required modalities/action present",
                "clock_fits": alignment.get("clock_fits"),
                "tolerances_ms": alignment.get("tolerances_ms"),
                "action_validity_policy": alignment.get(
                    "action_validity_policy"
                ),
            },
            "warnings": [
                "Force channels are excluded from each observation when force.valid is false.",
                "Split train/validation/test by episode, not by neighboring frames from the same episode.",
            ] + (
                [
                    "Ultrasound depth and beam axes are independently resampled to a square image; use metadata for physical dimensions."
                ]
                if not preserve_ultrasound_physical_aspect
                else []
            ) + (
                [
                    "Ultrasound decoding used requested settings because actual PhaseLink readback was unavailable."
                ]
                if not settings_are_actual
                else []
            ),
        }
        write_json(temporary_dir / "dataset_info.json", dataset_info)

        if output_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = output_dir.with_name(output_dir.name + ".backup_" + timestamp)
            if backup_dir.exists():
                raise FileExistsError("backup path already exists: {}".format(backup_dir))
            publish_directory(output_dir, backup_dir)
        publish_method, publish_warning = publish_directory(
            temporary_dir, output_dir
        )
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if (
            backup_dir is not None
            and backup_dir.exists()
            and not output_dir.exists()
        ):
            publish_directory(backup_dir, output_dir)
        raise

    print("Training dataset ready:", output_dir)
    print("Valid samples:", len(valid_samples), "/", len(all_samples))
    print("Force valid:", force_audit["valid"], "-", force_audit["reason"])
    print("Ultrasound settings source:", settings_source)
    print("Output publish method:", publish_method)
    if publish_warning is not None:
        print("WARNING:", publish_warning)
    if backup_dir is not None:
        print("Previous output backup:", backup_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one portable VLA training folder from a collected episode."
    )
    parser.add_argument("--episode", required=True, help="Source episode directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <episode>/vla_training_15hz",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--ultrasound-size",
        type=int,
        default=256,
        help="Square ultrasound output width and height in pixels (default: 256).",
    )
    parser.add_argument(
        "--preserve-ultrasound-physical-aspect",
        action="store_true",
        help=(
            "Use the previous equal-mm-per-pixel rectangular ultrasound "
            "output instead of the default square model input."
        ),
    )
    parser.add_argument(
        "--realign",
        action="store_true",
        help="Regenerate aligned_samples.jsonl even when it already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output after preserving it as a timestamped backup.",
    )
    parser.add_argument(
        "--require-force",
        action="store_true",
        help="Fail instead of building when usable force measurements are absent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be a positive finite number")
    if args.ultrasound_size <= 0:
        raise ValueError("--ultrasound-size must be positive")
    episode_dir = Path(args.episode).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else episode_dir / "vla_training_15hz"
    )
    if output_dir == episode_dir:
        raise ValueError("output directory cannot be the episode directory itself")
    if output_dir == Path(output_dir.anchor):
        raise ValueError("output directory cannot be a filesystem root")
    try:
        episode_dir.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output directory cannot contain the source episode")
    build_dataset(
        episode_dir=episode_dir,
        output_dir=output_dir,
        fps=float(args.fps),
        overwrite=args.overwrite,
        realign=args.realign,
        require_force=args.require_force,
        ultrasound_size=args.ultrasound_size,
        preserve_ultrasound_physical_aspect=(
            args.preserve_ultrasound_physical_aspect
        ),
    )


if __name__ == "__main__":
    main()
