import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_single_curve_from_latest_points_servo_soft_pose as trajectory


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class StateFiveThenReadyArm:
    def __init__(self):
        self.calls = []
        self.states = [5, 2, 2, 2, 2]
        self.state_index = 0
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0

    def set_mode(self, mode):
        self.calls.append(("set_mode", mode))
        return 0

    def set_state(self, state):
        self.calls.append(("set_state", state))
        return 0

    def get_state(self):
        state = self.states[min(self.state_index, len(self.states) - 1)]
        self.state_index += 1
        return 0, state


class StagedApproachTest(unittest.TestCase):
    def test_rearms_state_five_before_first_position_move(self):
        clock = FakeClock()
        arm = StateFiveThenReadyArm()

        status = trajectory.restore_position_mode_ready(
            arm,
            timeout_seconds=1.0,
            stable_seconds=0.1,
            poll_seconds=0.05,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertEqual(arm.calls, [("set_mode", 0), ("set_state", 0)])
        self.assertEqual(status["state"], 2)

    def test_keeps_current_z_then_uses_exact_trajectory_start(self):
        current = [424.3, 108.2, 402.9, 164.3, 14.8, -92.2]
        start = [457.668732, -230.973633, 264.2901, 163.968489, 23.389627, -92.300235]

        above, descent = trajectory.make_staged_approach_poses(current, start)

        self.assertEqual(above, [start[0], start[1], current[2], *start[3:]])
        self.assertEqual(descent, start)
        self.assertEqual(above[2], current[2])
        self.assertEqual(above[:2] + above[3:], descent[:2] + descent[3:])

    def test_rejects_starting_below_trajectory_start(self):
        with self.assertRaisesRegex(RuntimeError, "below trajectory-start Z"):
            trajectory.make_staged_approach_poses(
                [0, 0, 100, 0, 0, 0],
                [1, 2, 200, 3, 4, 5],
            )


if __name__ == "__main__":
    unittest.main()
