import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

import remote_client


class FakeBundle:
    def __init__(self, n=200):
        self.n = n
        self.state = np.zeros((n, 6), dtype=np.float32)
        self.action_gt = np.zeros((n, 6), dtype=np.float32)
        self.rgb = np.zeros((n, 2, 2, 3), dtype=np.uint8)
        self.us = np.zeros((n, 2, 2, 3), dtype=np.uint8)
        self.cam_base = "base_0_rgb"
        self.cam_wrist = "left_wrist_0_rgb"
        self.task = "timing test"


class RemoteTimingTest(unittest.TestCase):
    def test_forced_warmup_observation_sets_must_go(self):
        args = SimpleNamespace(
            fps=100,
            actions_per_chunk=50,
            chunk_size_threshold=0.5,
        )
        client = remote_client.LeanRobotClient(args, FakeBundle())

        self.assertTrue(client.control_loop_observation(force_must_go=True))
        obs = client.observation_queue.get_nowait()
        self.assertTrue(obs.must_go)

    def test_slow_observation_rpc_does_not_block_action_clock(self):
        args = SimpleNamespace(
            fps=100,
            actions_per_chunk=50,
            chunk_size_threshold=0.5,
        )
        client = remote_client.LeanRobotClient(args, FakeBundle())
        client.start_barrier = threading.Barrier(1)

        from lerobot.async_inference.helpers import TimedAction

        for timestep in range(50):
            client.action_queue.put(
                TimedAction(
                    timestamp=time.time(),
                    timestep=timestep,
                    action=np.zeros(6, dtype=np.float32),
                )
            )

        rpc_started = threading.Event()

        def slow_send(_obs):
            rpc_started.set()
            time.sleep(0.30)
            return True

        client.send_observation = slow_send
        obs_thread = threading.Thread(target=client.send_observations_loop)
        control_thread = threading.Thread(target=client.control_loop)
        obs_thread.start()
        control_thread.start()

        self.assertTrue(rpc_started.wait(timeout=1.0))
        time.sleep(0.20)
        client.shutdown_event.set()
        control_thread.join(timeout=1.0)
        obs_thread.join(timeout=1.0)

        self.assertFalse(control_thread.is_alive())
        self.assertFalse(obs_thread.is_alive())
        # The old synchronous implementation stopped around action 25 for
        # the entire 300 ms RPC. The decoupled clock keeps consuming actions.
        self.assertGreaterEqual(len(client.performed), 40)


if __name__ == "__main__":
    unittest.main()
