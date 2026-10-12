import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import Mock

from real_robot_client import (
    PolicyTargetRateLimiter,
    SafetyEnvelope,
    SafetyStop,
    angle_delta_deg,
    pose_distance,
)


class SafetyEnvelopeTest(unittest.TestCase):
    def make_envelope(self):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "trajectory.json"
        points = [[424.0, float(y), 330.0, 164.0, 15.0, -92.0] for y in range(20)]
        path.write_text(json.dumps({"trajectory": points}), encoding="utf-8")
        envelope = SafetyEnvelope(
            path,
            tube_mm=5,
            tube_deg=3,
            max_step_mm=3,
            max_step_deg=2,
            max_tracking_error_mm=10,
            max_tracking_error_deg=5,
            start_tolerance_mm=2,
            start_tolerance_deg=2,
            max_backtrack_points=2,
            max_forward_jump_points=5,
        )
        return temp, envelope

    def test_angle_wrap(self):
        self.assertAlmostEqual(float(angle_delta_deg(-179, 179)), 2.0)

    def test_accepts_small_forward_action(self):
        temp, envelope = self.make_envelope()
        self.addCleanup(temp.cleanup)
        envelope.require_at_start([424, 0, 330, 164, 15, -92])
        result = envelope.validate(
            [424, 1, 330, 164, 15, -92],
            [424, 0.5, 330, 164, 15, -92],
        )
        self.assertEqual(result["reference_index"], 1)

    def test_rejects_action_outside_tube(self):
        temp, envelope = self.make_envelope()
        self.addCleanup(temp.cleanup)
        envelope.require_at_start([424, 0, 330, 164, 15, -92])
        with self.assertRaises(SafetyStop):
            envelope.validate(
                [450, 1, 330, 164, 15, -92],
                [424, 0.5, 330, 164, 15, -92],
            )

    def test_rejects_large_single_step(self):
        temp, envelope = self.make_envelope()
        self.addCleanup(temp.cleanup)
        envelope.require_at_start([424, 0, 330, 164, 15, -92])
        with self.assertRaises(SafetyStop):
            envelope.validate(
                [424, 4, 330, 164, 15, -92],
                [424, 3.5, 330, 164, 15, -92],
            )

    def test_accepts_project_trajectory_record_format(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "trajectory.json"
        path.write_text(
            json.dumps(
                {
                    "trajectory": [
                        {
                            "trajectory_index": 0,
                            "target_tcp_pose": [424, 0, 330, 164, 15, -92],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        envelope = SafetyEnvelope(
            path, 5, 3, 3, 2, 10, 5, 2, 2, 2, 5
        )
        self.assertEqual(envelope.points.shape, (1, 6))

    def test_direct_policy_mode_skips_reference_trajectory(self):
        envelope = SafetyEnvelope(
            None, 5, 3, 3, 2, 10, 5, 2, 2, 2, 5
        )
        envelope.require_at_start([424, 100, 330, 164, 15, -92])
        result = envelope.validate(
            [424, 101, 330, 164, 15, -92],
            [424, 100.5, 330, 164, 15, -92],
        )
        self.assertIsNone(result["reference_index"])
        self.assertIsNone(result["tube_distance_mm"])

    def test_direct_policy_mode_keeps_step_limit(self):
        envelope = SafetyEnvelope(
            None, 5, 3, 3, 2, 10, 5, 2, 2, 2, 5
        )
        envelope.require_at_start([424, 100, 330, 164, 15, -92])
        with self.assertRaises(SafetyStop):
            envelope.validate(
                [424, 104, 330, 164, 15, -92],
                [424, 103.5, 330, 164, 15, -92],
            )


class XArmStateValidationTest(unittest.TestCase):
    def make_robot(self, state):
        from real_robot_client import XArmInterface

        robot = XArmInterface("192.0.2.1", "real", 40, 5, False)
        robot.arm = Mock(connected=True)
        robot.snapshot = Mock(
            return_value={
                "state": state,
                "error_code": 0,
                "warn_code": 0,
                "pose": np.zeros(6),
                "ft_ext_force": np.ones(6),
            }
        )
        return robot

    def test_state5_allowed_only_during_pre_execution_observation(self):
        robot = self.make_robot(5)
        self.assertEqual(
            robot._require_healthy(require_ft=False, allow_state5=True)["state"],
            5,
        )
        with self.assertRaises(SafetyStop):
            robot._require_healthy(require_ft=False, allow_state5=False)

    def test_state4_remains_rejected(self):
        robot = self.make_robot(4)
        with self.assertRaises(SafetyStop):
            robot._require_healthy(require_ft=False, allow_state5=True)


class PolicyTargetRateLimiterTest(unittest.TestCase):
    def test_default_mode_preserves_raw_target(self):
        limiter = PolicyTargetRateLimiter(False, 15, 2, 1)
        raw = np.array([100, 20, 30, 10, 20, 30], dtype=float)
        command, info = limiter.apply(raw, np.zeros(6))
        np.testing.assert_allclose(command, raw)
        self.assertFalse(info["enabled"])

    def test_limits_far_translation_and_rotation_by_rate(self):
        limiter = PolicyTargetRateLimiter(True, 10, 2, 5)
        command, info = limiter.apply(
            [100, 0, 0, 0, 20, 0],
            [0, 0, 0, 0, 0, 0],
        )
        linear, angular = pose_distance(command, np.zeros(6))
        self.assertAlmostEqual(linear, 0.2)
        self.assertAlmostEqual(angular, 0.5)
        self.assertTrue(info["linear_limited"])
        self.assertTrue(info["angular_limited"])

    def test_uses_shortest_angle_across_wrap(self):
        limiter = PolicyTargetRateLimiter(True, 10, 10, 10)
        command, _ = limiter.apply(
            [0, 0, 0, -179, 0, 0],
            [0, 0, 0, 179, 0, 0],
        )
        self.assertAlmostEqual(
            float(angle_delta_deg(command[3], 179)), 1.0
        )

    def test_chunk_direction_change_remains_continuous(self):
        limiter = PolicyTargetRateLimiter(True, 10, 10, 10)
        first, _ = limiter.apply([100, 0, 0, 0, 0, 0], np.zeros(6))
        second, _ = limiter.apply([-100, 0, 0, 0, 0, 0], np.zeros(6))
        self.assertAlmostEqual(pose_distance(first, np.zeros(6))[0], 1.0)
        self.assertAlmostEqual(pose_distance(second, first)[0], 1.0)

    def test_limited_far_target_passes_existing_live_limits(self):
        limiter = PolicyTargetRateLimiter(True, 15, 2, 1)
        envelope = SafetyEnvelope(
            None, 5, 3, 3, 2, 12, 5, 2, 2, 2, 5
        )
        actual = np.array([424, 100, 330, 164, 15, -92], dtype=float)
        envelope.require_at_start(actual)
        command, _ = limiter.apply(
            [424, -200, 300, 175, 10, -90], actual
        )
        result = envelope.validate(command, actual)
        self.assertLessEqual(result["tracking_error_mm"], 2.0 / 15.0 + 1e-9)
        self.assertLessEqual(result["tracking_angle_deg"], 1.0 / 15.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
