import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fitted_trajectory_vla_sync as collector


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeCore:
    def __init__(self):
        self.state_is_ready = False


class ReadyArm:
    def __init__(self, states, motor_enable_states=None):
        self.states = list(states)
        self.state_index = 0
        self.mode = 0
        self.axis = 6
        self.motor_enable_states = (
            [1, 1, 1, 1, 1, 1, 0, 0]
            if motor_enable_states is None
            else list(motor_enable_states)
        )
        self.error_code = 0
        self.warn_code = 0
        self.core = FakeCore()

    def get_state(self):
        state = self.states[min(self.state_index, len(self.states) - 1)]
        self.state_index += 1
        self.core.state_is_ready = state == 2
        return 0, state

    def get_err_warn_code(self):
        return 0, [0, 0]


class OrderedArm:
    def __init__(self):
        self.calls = []
        self.warn_code = 0
        self.error_code = 0

    def motion_enable(self, enabled):
        self.calls.append(("motion_enable", enabled))
        return 0

    def set_mode(self, mode):
        self.calls.append(("set_mode", mode))
        return 0

    def set_tcp_offset(self, offset, wait):
        self.calls.append(("set_tcp_offset", list(offset), wait))
        return 0

    def set_state(self, state):
        self.calls.append(("set_state", state))
        return 0

    def set_position(self, **kwargs):
        self.calls.append(("set_position", kwargs))
        return 0


class MemoryActionRecorder:
    def __init__(self):
        self.actions = []

    def record(self, action):
        self.actions.append(action)


class RobotReadinessTest(unittest.TestCase):
    def test_waits_until_state_two_is_stable(self):
        clock = FakeClock()
        arm = ReadyArm([5, 5, 2, 2, 2, 2])

        status = collector.wait_for_robot_ready(
            arm,
            expected_mode=0,
            timeout=1.0,
            stable_seconds=0.1,
            poll_seconds=0.05,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertEqual(status["state"], 2)
        self.assertTrue(status["motors_ready"])
        self.assertTrue(status["protocol_ready"])

    def test_real_report_unknown_motor_fields_use_protocol_ready(self):
        clock = FakeClock()
        arm = ReadyArm([2], motor_enable_states=[-1] * 8)

        status = collector.wait_for_robot_ready(
            arm,
            expected_mode=0,
            timeout=1.0,
            stable_seconds=0.1,
            poll_seconds=0.05,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertFalse(status["motor_states_available"])
        self.assertFalse(status["motors_ready"])
        self.assertTrue(status["protocol_ready"])

    def test_state_five_times_out_without_bypassing_safety(self):
        clock = FakeClock()
        arm = ReadyArm([5])

        with self.assertRaisesRegex(RuntimeError, "state.*5"):
            collector.wait_for_robot_ready(
                arm,
                expected_mode=0,
                timeout=0.2,
                stable_seconds=0.1,
                poll_seconds=0.05,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

    def test_prepare_configures_robot_without_cartesian_motion(self):
        arm = OrderedArm()
        tcp_offset = [0.0] * 6

        with mock.patch.object(collector, "wait_for_robot_ready") as wait_ready:
            collector.prepare_robot_for_capture(arm, tcp_offset)

        names = [call[0] for call in arm.calls]
        self.assertEqual(
            names,
            [
                "motion_enable",
                "set_mode",
                "set_tcp_offset",
                "set_state",
            ],
        )
        self.assertEqual(
            wait_ready.call_args_list,
            [mock.call(arm, expected_mode=0)],
        )

    def test_restore_ready_after_operator_wait_before_capture(self):
        arm = OrderedArm()
        ready_status = {"state": 2, "mode": 0}

        with mock.patch.object(
            collector, "wait_for_robot_ready", return_value=ready_status
        ) as wait_ready:
            result = collector.restore_position_mode_ready(arm)

        self.assertEqual(
            arm.calls,
            [("set_mode", 0), ("set_state", 0)],
        )
        wait_ready.assert_called_once_with(arm, expected_mode=0)
        self.assertIs(result, ready_status)

    def test_staged_approach_keeps_current_z_then_descends(self):
        arm = OrderedArm()
        recorder = MemoryActionRecorder()
        curve = [[1.0, 2.0, 3.0, 170.0, 20.0, -90.0]]
        current_pose = [10.0, 20.0, 30.0, 160.0, 10.0, -80.0]

        with mock.patch.object(collector, "wait_for_robot_ready") as wait_ready:
            offset = collector.move_to_trajectory_start_staged(
                arm,
                curve,
                current_pose,
                recorder,
                approach_speed_mm_s=10.0,
                descent_speed_mm_s=5.0,
            )

        position_calls = [call[1] for call in arm.calls if call[0] == "set_position"]
        self.assertEqual(offset, 2)
        self.assertEqual(len(position_calls), 2)
        self.assertEqual(
            [position_calls[0][key] for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            [1.0, 2.0, 30.0, 170.0, 20.0, -90.0],
        )
        self.assertEqual(
            [position_calls[1][key] for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            curve[0],
        )
        self.assertEqual(
            [action["type"] for action in recorder.actions],
            ["position_target", "position_target"],
        )
        self.assertEqual(
            [action["phase"] for action in recorder.actions],
            ["constant_z_approach", "vertical_descent"],
        )
        wait_ready.assert_called_once_with(arm, expected_mode=1)


class OutputStorageTest(unittest.TestCase):
    def test_prepares_output_root_and_records_free_space(self):
        usage = mock.Mock(total=100 * 1024 ** 3, used=20 * 1024 ** 3,
                          free=80 * 1024 ** 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = str(Path(temp_dir) / "episodes")
            with mock.patch.object(collector.shutil, "disk_usage", return_value=usage):
                resolved, metadata = collector.prepare_output_root(output_root, 10.0)

        self.assertEqual(resolved, str(Path(output_root).resolve()))
        self.assertEqual(metadata["free_bytes_before_episode"], usage.free)
        self.assertEqual(metadata["minimum_free_gb"], 10.0)

    def test_rejects_low_space_before_episode_creation(self):
        usage = mock.Mock(total=10 * 1024 ** 3, used=9 * 1024 ** 3,
                          free=1 * 1024 ** 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(collector.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(RuntimeError, "insufficient free space"):
                    collector.prepare_output_root(temp_dir, 10.0)


class UltrasoundReadinessTest(unittest.TestCase):
    def test_accepts_nonempty_packet_and_index_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "ultrasound_packets.bin").write_bytes(b"packet")
            Path(temp_dir, "ultrasound_frames.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            result = collector.wait_for_ultrasound_packets(temp_dir, timeout=0.1)

        self.assertEqual(result["packet_file_bytes"], 6)
        self.assertGreater(result["index_file_bytes"], 0)

    def test_rejects_zero_byte_ultrasound_files(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "ultrasound_packets.bin").touch()
            Path(temp_dir, "ultrasound_frames.jsonl").touch()
            with self.assertRaisesRegex(RuntimeError, "not reaching Client::DataPacket"):
                collector.wait_for_ultrasound_packets(
                    temp_dir,
                    timeout=0.2,
                    poll_seconds=0.05,
                    monotonic=clock.monotonic,
                    sleep=clock.sleep,
                )


if __name__ == "__main__":
    unittest.main()
