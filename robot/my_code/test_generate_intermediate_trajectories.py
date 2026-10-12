import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_intermediate_trajectories as generator


def write_teach_file(path, poses, robot_ip="192.168.1.100", tcp_offset=None):
    payload = {
        "robot_ip": robot_ip,
        "tcp_offset": tcp_offset or [0, 0, 0, 0, 0, 0],
        "points": [
            {"name": "P{:03d}".format(index), "index": index, "tcp_pose": pose}
            for index, pose in enumerate(poses)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class GenerateIntermediateTrajectoriesTest(unittest.TestCase):
    def test_reverses_direction_and_generates_midline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.json"
            second = root / "b.json"
            output = root / "output"
            write_teach_file(
                first,
                [[0, 0, 100, 179, 0, 0], [10, 0, 100, 179, 0, 0]],
            )
            write_teach_file(
                second,
                [[10, 10, 100, -179, 0, 0], [0, 10, 100, -179, 0, 0]],
            )

            manifest = generator.generate_intermediate_trajectories(
                [first, second], between=1, output_dir=output, control_points=3
            )

            self.assertEqual(manifest["generated_count"], 1)
            self.assertTrue(manifest["direction_alignment"][1]["reversed"])
            self.assertEqual(
                manifest["generated_trajectories"][0]["file"],
                "intermediate_01_01.json",
            )
            generated_file = output / manifest["generated_trajectories"][0]["file"]
            payload = json.loads(generated_file.read_text(encoding="utf-8"))
            poses = [item["tcp_pose"] for item in payload["points"]]
            self.assertEqual([pose[1] for pose in poses], [5.0, 5.0, 5.0])
            self.assertEqual([pose[0] for pose in poses], [0.0, 5.0, 10.0])
            self.assertTrue(all(abs(abs(pose[3]) - 180.0) < 1e-6 for pose in poses))
            self.assertTrue((output / "trajectory_preview.svg").is_file())
            self.assertTrue((output / "generation_manifest.json").is_file())

    def test_rejects_tcp_offset_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.json"
            second = root / "b.json"
            write_teach_file(first, [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
            write_teach_file(
                second,
                [[0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                tcp_offset=[1, 0, 0, 0, 0, 0],
            )

            with self.assertRaisesRegex(ValueError, "tcp_offset mismatch"):
                generator.generate_intermediate_trajectories(
                    [first, second], between=1, output_dir=root / "output"
                )

    def test_rejects_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.json"
            second = root / "b.json"
            output = root / "output"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            write_teach_file(first, [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
            write_teach_file(second, [[0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                generator.generate_intermediate_trajectories(
                    [first, second], between=1, output_dir=output
                )


if __name__ == "__main__":
    unittest.main()
