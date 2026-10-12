import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import align_vla_episode as align


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class AlignVlaEpisodeTest(unittest.TestCase):
    def test_angle_interpolation_wraps(self):
        self.assertAlmostEqual(
            align.interpolate_angle_degrees(179.0, -179.0, 0.5), -180.0
        )

    def test_long_lived_position_action_remains_valid(self):
        action = {"type": "position_target", "return_code": 0}
        self.assertTrue(align.action_is_valid(action, -30_000.0, 80.0))
        servo = {"type": "cartesian_servo_target", "return_code": 0}
        self.assertFalse(align.action_is_valid(servo, -100.0, 80.0))

    def test_synthetic_episode_uses_real_actions(self):
        base_ns = 1_800_000_000_000_000_000
        duration_ns = 1_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            (episode / "metadata.json").write_text(
                json.dumps(
                    {
                        "capture_window": {
                            "start_unix_ns": base_ns,
                            "end_unix_ns": base_ns + duration_ns,
                        }
                    }
                ),
                encoding="utf-8",
            )

            robot = []
            actions = []
            for index in range(51):
                timestamp = base_ns + index * 20_000_000
                robot.append(
                    {
                        "record_index": index,
                        "host_unix_ns": timestamp,
                        "tcp_pose": [index, 0, 0, 179 if index == 0 else -179, 0, 0],
                        "joint_angles": [179 if index == 0 else -179] * 7,
                        "joint_torque": [index] * 7,
                        "ft_ext_force": [1] * 6,
                        "ft_raw_force": [2] * 6,
                        "state": 0,
                        "mode": 1,
                        "error_code": 0,
                        "warn_code": 0,
                    }
                )
                actions.append(
                    {
                        "action_index": index,
                        "host_unix_ns": timestamp,
                        "command_end_unix_ns": timestamp + 500_000,
                        "target_tcp_pose": [index, 0, 0, 0, 0, 0],
                        "return_code": 0,
                    }
                )

            realsense = []
            for index in range(31):
                device_ms = index * (1000.0 / 30.0) + 0.25
                ideal_ns = base_ns + int(round(index * 1e9 / 30.0))
                jitter_ns = ((index % 3) - 1) * 800_000
                realsense.append(
                    {
                        "frame_index": index,
                        "device_timestamp_ms": device_ms,
                        "device_frame_number": index,
                        "host_unix_ns": ideal_ns + jitter_ns,
                        "color_path": "color/{:06d}.jpg".format(index),
                        "depth_path": "depth/{:06d}.png".format(index),
                    }
                )

            ultrasound = []
            for index in range(41):
                ticks = index * 5_000_000
                ideal_ns = base_ns + index * 25_000_000
                jitter_ns = ((index % 5) - 2) * 400_000
                ultrasound.append(
                    {
                        "packet_index": index,
                        "device_timestamp_ticks_200mhz": str(ticks),
                        "device_frame_number": index,
                        "host_unix_ns": str(ideal_ns + jitter_ns),
                        "host_steady_ns": str(index * 25_000_000),
                        "binary_offset": index * 64,
                        "binary_size": 64,
                    }
                )

            # Simulate a PhaseLink UI/control packet emitted well after the
            # capture window. Its tail is not a valid 200 MHz timestamp and
            # must not corrupt the capture-window device-clock fit.
            ultrasound.append(
                {
                    "packet_index": 41,
                    "device_timestamp_ticks_200mhz": str(0x4000100010004),
                    "device_frame_number": 0,
                    "host_unix_ns": str(base_ns + 20_000_000_000),
                    "host_steady_ns": str(20_000_000_000),
                    "binary_offset": 41 * 64,
                    "binary_size": 64,
                }
            )

            write_jsonl(episode / "robot_records.jsonl", robot)
            write_jsonl(episode / "robot_actions.jsonl", actions)
            write_jsonl(episode / "realsense_frames.jsonl", realsense)
            write_jsonl(episode / "ultrasound_frames.jsonl", ultrasound)
            (episode / "ultrasound_packets.bin").write_bytes(bytes(41 * 64))

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(align.__file__)),
                    "--episode",
                    str(episode),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            alignment = json.loads(
                (episode / "alignment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                alignment["action_source"],
                "commanded_position_or_cartesian_servo_target",
            )
            self.assertTrue(alignment["clock_fits"]["realsense"]["used"])
            self.assertTrue(alignment["clock_fits"]["ultrasound"]["used"])
            self.assertEqual(
                alignment["clock_fits"]["ultrasound"]["pair_count"], 41
            )
            self.assertEqual(
                alignment["clock_fits"]["ultrasound"]["total_record_count"], 42
            )

            samples = [
                json.loads(line)
                for line in (episode / "aligned_samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreaterEqual(len(samples), 15)
            self.assertIsNone(samples[0]["action"])
            self.assertFalse(samples[0]["quality"]["valid"])
            valid = next(sample for sample in samples if sample["quality"]["valid"])
            self.assertIn("target_tcp_pose", valid["action"])
            self.assertIn("alignment_skew_ms", valid["action"])
            self.assertIn("alignment_skew_ms", valid["observation"]["robot"])
            self.assertIn("alignment_skew_ms", valid["observation"]["realsense"])
            self.assertIn("alignment_skew_ms", valid["observation"]["ultrasound"])

            dropped = subprocess.run(
                [
                    sys.executable,
                    str(Path(align.__file__)),
                    "--episode",
                    str(episode),
                    "--drop-invalid",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(dropped.returncode, 0, dropped.stderr)
            dropped_samples = [
                json.loads(line)
                for line in (episode / "aligned_samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(item["quality"]["valid"] for item in dropped_samples))
            dropped_alignment = json.loads(
                (episode / "alignment.json").read_text(encoding="utf-8")
            )
            self.assertGreater(dropped_alignment["skipped_invalid_count"], 0)


if __name__ == "__main__":
    unittest.main()
