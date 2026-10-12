import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_robot_trajectory_with_realsense_preview import RealSenseRgbPreview


class FakeColorFrame:
    def get_data(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class FakeFrames:
    def get_color_frame(self):
        return FakeColorFrame()


class FakePipeline:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self, config):
        self.started = True

    def wait_for_frames(self, timeout_ms=1000):
        return FakeFrames()

    def stop(self):
        self.stopped = True


class FakeConfig:
    def enable_stream(self, *args):
        pass


class FakeCv2(types.SimpleNamespace):
    WINDOW_NORMAL = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    WND_PROP_VISIBLE = 0

    def namedWindow(self, *args):
        pass

    def resizeWindow(self, *args):
        pass

    def putText(self, *args):
        pass

    def imshow(self, *args):
        pass

    def waitKey(self, delay):
        return ord("q")

    def getWindowProperty(self, *args):
        return 1

    def destroyWindow(self, *args):
        pass


class PreviewOnlyTest(unittest.TestCase):
    def test_user_close_does_not_set_robot_stop_event(self):
        pipeline = FakePipeline()
        fake_rs = types.SimpleNamespace(
            pipeline=lambda: pipeline,
            config=FakeConfig,
            stream=types.SimpleNamespace(color=1),
            format=types.SimpleNamespace(bgr8=1),
        )
        stop_event = threading.Event()
        preview = RealSenseRgbPreview(stop_event)

        with mock.patch.dict(
            sys.modules,
            {"cv2": FakeCv2(), "pyrealsense2": fake_rs},
        ):
            preview.run()

        self.assertTrue(preview.closed_by_user)
        self.assertFalse(stop_event.is_set())
        self.assertIsNone(preview.error)
        self.assertEqual(preview.frame_count, 1)
        self.assertTrue(pipeline.started)
        self.assertTrue(pipeline.stopped)


if __name__ == "__main__":
    unittest.main()
