#!/usr/bin/env python3
"""Generate intermediate Cartesian teach-point trajectories without robot I/O.

The input order defines the cross-scan order.  New trajectories are generated
only between each adjacent pair of measured trajectories; this tool never
extrapolates outside the measured set.
"""

import argparse
import bisect
import html
import json
import math
import os
import time
from pathlib import Path


# =============================================================================
# USER CONFIGURATION
#
# Edit this block when you prefer to run the script without command-line paths.
# Input order is important: list measured trajectories in their physical
# cross-scan order (for example left -> center -> right). Use raw strings for
# Windows absolute paths. At least two input files are required.
#
# Example:
# INPUT_TRAJECTORY_FILES = [
#     r"D:\teach_trajectories\scan_left.json",
#     r"D:\teach_trajectories\scan_center.json",
#     r"D:\teach_trajectories\scan_right.json",
# ]
# OUTPUT_DIRECTORY = r"D:\teach_trajectories\generated_set_001"
# =============================================================================
INPUT_TRAJECTORY_FILES = [
      r"<xArm-SDK>\my_code\teach_points_3_1.json",
      r"<xArm-SDK>\my_code\teach_points_3_2.json",
      r"<xArm-SDK>\my_code\teach_points_3_3.json",
]
OUTPUT_DIRECTORY = r"<xArm-SDK>\my_code\generated_set_3"

GENERATED_BETWEEN_EACH_PAIR = 3

# None means: use the largest point count among the input JSON files.
OUTPUT_CONTROL_POINT_COUNT = None
# =============================================================================


POSITION_EPSILON_MM = 1e-9
METADATA_TOLERANCE = 1e-6


def wrap_degrees(value):
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 and value > 0 else wrapped


def shortest_angle_delta_degrees(start, end):
    return (float(end) - float(start) + 180.0) % 360.0 - 180.0


def interpolate_pose(start, end, fraction):
    pose = [
        float(start[index])
        + (float(end[index]) - float(start[index])) * fraction
        for index in range(3)
    ]
    for index in range(3, 6):
        value = float(start[index]) + shortest_angle_delta_degrees(
            start[index], end[index]
        ) * fraction
        pose.append(wrap_degrees(value))
    return pose


def position_distance(first, second):
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def load_teach_trajectory(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("{}: points must contain at least two items".format(path))

    poses = []
    for index, item in enumerate(raw_points):
        pose = item.get("tcp_pose") if isinstance(item, dict) else None
        if (
            not isinstance(pose, list)
            or len(pose) != 6
            or not all(isinstance(value, (int, float)) for value in pose)
        ):
            raise ValueError(
                "{}: points[{}].tcp_pose must contain six numbers".format(
                    path, index
                )
            )
        poses.append([float(value) for value in pose])

    tcp_offset = payload.get("tcp_offset")
    if (
        not isinstance(tcp_offset, list)
        or len(tcp_offset) != 6
        or not all(isinstance(value, (int, float)) for value in tcp_offset)
    ):
        raise ValueError("{}: tcp_offset must contain six numbers".format(path))

    robot_ip = payload.get("robot_ip")
    if not isinstance(robot_ip, str) or not robot_ip.strip():
        raise ValueError("{}: robot_ip is missing".format(path))

    return {
        "path": str(path),
        "robot_ip": robot_ip.strip(),
        "tcp_offset": [float(value) for value in tcp_offset],
        "poses": poses,
        "original_count": len(poses),
    }


def validate_common_metadata(trajectories):
    reference = trajectories[0]
    for current in trajectories[1:]:
        if current["robot_ip"] != reference["robot_ip"]:
            raise ValueError(
                "robot_ip mismatch: {} has {}, expected {}".format(
                    current["path"], current["robot_ip"], reference["robot_ip"]
                )
            )
        if any(
            abs(first - second) > METADATA_TOLERANCE
            for first, second in zip(
                current["tcp_offset"], reference["tcp_offset"]
            )
        ):
            raise ValueError(
                "tcp_offset mismatch between {} and {}".format(
                    reference["path"], current["path"]
                )
            )


def align_direction(reference_poses, candidate_poses):
    same_cost = position_distance(reference_poses[0], candidate_poses[0]) + position_distance(
        reference_poses[-1], candidate_poses[-1]
    )
    reverse_cost = position_distance(reference_poses[0], candidate_poses[-1]) + position_distance(
        reference_poses[-1], candidate_poses[0]
    )
    if reverse_cost < same_cost:
        return list(reversed(candidate_poses)), True, same_cost, reverse_cost
    return list(candidate_poses), False, same_cost, reverse_cost


def cumulative_lengths(poses):
    lengths = [0.0]
    for index in range(1, len(poses)):
        lengths.append(lengths[-1] + position_distance(poses[index - 1], poses[index]))
    return lengths


def resample_by_normalized_arc_length(poses, output_count):
    if output_count < 2:
        raise ValueError("control point count must be at least 2")
    lengths = cumulative_lengths(poses)
    total_length = lengths[-1]
    if total_length <= POSITION_EPSILON_MM:
        raise ValueError("trajectory has zero Cartesian path length")

    result = []
    for output_index in range(output_count):
        target = total_length * output_index / (output_count - 1)
        right = bisect.bisect_right(lengths, target)
        segment = min(max(right - 1, 0), len(poses) - 2)
        segment_length = lengths[segment + 1] - lengths[segment]
        if segment_length <= POSITION_EPSILON_MM:
            fraction = 0.0
        else:
            fraction = (target - lengths[segment]) / segment_length
        result.append(interpolate_pose(poses[segment], poses[segment + 1], fraction))
    result[0] = list(poses[0])
    result[-1] = list(poses[-1])
    return result


def path_metrics(poses):
    steps = [position_distance(poses[index - 1], poses[index]) for index in range(1, len(poses))]
    orientation_steps = []
    for index in range(1, len(poses)):
        orientation_steps.append(
            math.sqrt(
                sum(
                    shortest_angle_delta_degrees(
                        poses[index - 1][axis], poses[index][axis]
                    ) ** 2
                    for axis in range(3, 6)
                )
            )
        )
    return {
        "path_length_mm": sum(steps),
        "maximum_position_step_mm": max(steps) if steps else 0.0,
        "maximum_rpy_step_deg": max(orientation_steps) if orientation_steps else 0.0,
        "xyz_min_mm": [min(pose[axis] for pose in poses) for axis in range(3)],
        "xyz_max_mm": [max(pose[axis] for pose in poses) for axis in range(3)],
    }


def make_output_payload(reference, poses, generation):
    return {
        "robot_ip": reference["robot_ip"],
        "created_at": time.time(),
        "tcp_offset": reference["tcp_offset"],
        "count": len(poses),
        "unit": {"position": "mm", "orientation": "degree"},
        "source": "interpolated_between_measured_teach_trajectories",
        "generation": generation,
        "points": [
            {
                "name": "P{:03d}".format(index),
                "index": index,
                "tcp_pose": [round(value, 6) for value in pose],
            }
            for index, pose in enumerate(poses)
        ],
    }


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def project_point(pose, axes):
    return pose[axes[0]], pose[axes[1]]


def write_preview_svg(path, source_series, generated_series):
    panels = [((0, 1), "X-Y"), ((1, 2), "Y-Z"), ((0, 2), "X-Z")]
    width, height = 1440, 500
    margin, gap = 55, 35
    panel_width = (width - 2 * margin - 2 * gap) / 3
    panel_height = height - 2 * margin
    all_series = source_series + generated_series
    source_colors = ["#111827", "#374151", "#6b7280", "#9ca3af"]
    generated_colors = ["#2563eb", "#0891b2", "#059669", "#ca8a04", "#dc2626", "#9333ea"]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}.axis{stroke:#9ca3af;stroke-width:1}.source{stroke-width:2.5}.generated{stroke-width:1.5;stroke-dasharray:5 3}</style>',
    ]
    for panel_index, (axes, title) in enumerate(panels):
        x0 = margin + panel_index * (panel_width + gap)
        y0 = margin
        projected = [project_point(pose, axes) for _, poses in all_series for pose in poses]
        minimum_x = min(point[0] for point in projected)
        maximum_x = max(point[0] for point in projected)
        minimum_y = min(point[1] for point in projected)
        maximum_y = max(point[1] for point in projected)
        span_x = max(maximum_x - minimum_x, 1.0)
        span_y = max(maximum_y - minimum_y, 1.0)
        pad_x, pad_y = span_x * 0.05, span_y * 0.05
        minimum_x -= pad_x
        maximum_x += pad_x
        minimum_y -= pad_y
        maximum_y += pad_y

        def screen(point):
            x_value = x0 + (point[0] - minimum_x) / (maximum_x - minimum_x) * panel_width
            y_value = y0 + panel_height - (point[1] - minimum_y) / (maximum_y - minimum_y) * panel_height
            return x_value, y_value

        lines.append('<rect class="axis" fill="none" x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}"/>'.format(x0, y0, panel_width, panel_height))
        lines.append('<text x="{:.1f}" y="30" font-size="18" font-weight="600">{}</text>'.format(x0, title))
        for series_index, (_, poses) in enumerate(source_series):
            points = " ".join("{:.2f},{:.2f}".format(*screen(project_point(pose, axes))) for pose in poses)
            color = source_colors[series_index % len(source_colors)]
            lines.append('<polyline class="source" fill="none" stroke="{}" points="{}"/>'.format(color, points))
        for series_index, (_, poses) in enumerate(generated_series):
            points = " ".join("{:.2f},{:.2f}".format(*screen(project_point(pose, axes))) for pose in poses)
            color = generated_colors[series_index % len(generated_colors)]
            lines.append('<polyline class="generated" fill="none" stroke="{}" points="{}"/>'.format(color, points))

    legend_y = height - 15
    legend = "Measured: solid dark lines | Generated: colored dashed lines | input order defines adjacent trajectory pairs"
    lines.append('<text x="{}" y="{}" font-size="14">{}</text>'.format(margin, legend_y, html.escape(legend)))
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_intermediate_trajectories(input_paths, between, output_dir, control_points=None):
    if len(input_paths) < 2:
        raise ValueError("at least two input trajectories are required")
    if between < 1:
        raise ValueError("--between must be at least 1")

    trajectories = [load_teach_trajectory(path) for path in input_paths]
    validate_common_metadata(trajectories)
    if control_points is None:
        control_points = max(item["original_count"] for item in trajectories)
    if control_points < 2:
        raise ValueError("--control-points must be at least 2")

    aligned = []
    direction_records = []
    reference_poses = trajectories[0]["poses"]
    aligned.append(reference_poses)
    direction_records.append({"path": trajectories[0]["path"], "reversed": False})
    for trajectory in trajectories[1:]:
        poses, reversed_order, same_cost, reverse_cost = align_direction(
            reference_poses, trajectory["poses"]
        )
        aligned.append(poses)
        direction_records.append(
            {
                "path": trajectory["path"],
                "reversed": reversed_order,
                "same_direction_endpoint_cost_mm": same_cost,
                "reversed_direction_endpoint_cost_mm": reverse_cost,
            }
        )

    resampled = [
        resample_by_normalized_arc_length(poses, control_points)
        for poses in aligned
    ]

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "output directory is not empty: {}. Use a new or empty directory.".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_records = []
    generated_series = []
    for pair_index in range(len(resampled) - 1):
        first = resampled[pair_index]
        second = resampled[pair_index + 1]
        for intermediate_index in range(1, between + 1):
            alpha = intermediate_index / (between + 1)
            poses = [
                interpolate_pose(first[index], second[index], alpha)
                for index in range(control_points)
            ]
            filename = "intermediate_{:02d}_{:02d}.json".format(
                pair_index + 1,
                intermediate_index,
            )
            generation = {
                "method": "normalized_arc_length_pointwise_interpolation",
                "extrapolated": False,
                "source_a": trajectories[pair_index]["path"],
                "source_b": trajectories[pair_index + 1]["path"],
                "alpha_from_source_a_to_b": alpha,
                "control_point_count": control_points,
                "orientation_interpolation": "shortest_wrapped_RPY_degrees",
                "note": "Synthetic Cartesian targets only; no joint, RGB-D, force, or ultrasound data are fabricated.",
            }
            payload = make_output_payload(trajectories[0], poses, generation)
            write_json(output_dir / filename, payload)
            metrics = path_metrics(poses)
            generated_records.append(
                {
                    "file": filename,
                    "pair_index": pair_index,
                    "alpha": alpha,
                    "metrics": metrics,
                }
            )
            generated_series.append((filename, poses))

    source_series = [
        (Path(trajectory["path"]).name, poses)
        for trajectory, poses in zip(trajectories, resampled)
    ]
    write_preview_svg(
        output_dir / "trajectory_preview.svg", source_series, generated_series
    )
    manifest = {
        "created_at": time.time(),
        "method": "interpolation_only_between_adjacent_measured_trajectories",
        "input_order_is_cross_scan_order": True,
        "input_trajectories": [
            {
                "path": item["path"],
                "original_count": item["original_count"],
                "resampled_count": control_points,
                "metrics": path_metrics(poses),
            }
            for item, poses in zip(trajectories, resampled)
        ],
        "direction_alignment": direction_records,
        "between_each_adjacent_pair": between,
        "generated_count": len(generated_records),
        "robot_ip": trajectories[0]["robot_ip"],
        "tcp_offset": trajectories[0]["tcp_offset"],
        "generated_trajectories": generated_records,
        "safety_limitations": [
            "No inverse-kinematics, collision, singularity, force, or contact check is performed.",
            "Generated files must be reviewed and dry-run before any contact scan.",
            "The tool does not generate sensor observations; each trajectory still requires a real collection run.",
        ],
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate trajectories only between measured teach-point paths; no robot connection."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help=(
            "Two or more teach-point JSON files in cross-scan order. "
            "Overrides INPUT_TRAJECTORY_FILES in the script."
        ),
    )
    parser.add_argument(
        "--between",
        type=int,
        help=(
            "Number generated between each adjacent pair. Overrides "
            "GENERATED_BETWEEN_EACH_PAIR."
        ),
    )
    parser.add_argument(
        "--control-points",
        type=int,
        help=(
            "Points in each output JSON. Overrides OUTPUT_CONTROL_POINT_COUNT; "
            "default is the largest input point count."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Overrides OUTPUT_DIRECTORY in the script.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = args.inputs if args.inputs is not None else INPUT_TRAJECTORY_FILES
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIRECTORY
    between = (
        args.between
        if args.between is not None
        else GENERATED_BETWEEN_EACH_PAIR
    )
    control_points = (
        args.control_points
        if args.control_points is not None
        else OUTPUT_CONTROL_POINT_COUNT
    )
    if len(input_paths) < 2:
        raise SystemExit(
            "Configure at least two files in INPUT_TRAJECTORY_FILES near the "
            "top of this script, or pass --inputs fileA.json fileB.json."
        )
    if not output_dir:
        raise SystemExit(
            "Configure OUTPUT_DIRECTORY near the top of this script, or pass "
            "--output-dir."
        )
    manifest = generate_intermediate_trajectories(
        input_paths=input_paths,
        between=between,
        output_dir=output_dir,
        control_points=control_points,
    )
    print("Generated trajectories:", manifest["generated_count"])
    print("Output:", str(Path(output_dir).resolve()))
    print("Preview: trajectory_preview.svg")
    print("No robot connection or motion was performed.")


if __name__ == "__main__":
    main()
