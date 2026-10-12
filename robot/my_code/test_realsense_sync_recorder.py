import queue
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realsense_sync_recorder import (
    DEPTH_PNG_COMPRESSION,
    FRAME_QUEUE_CAPACITY,
    RealSenseRecorder,
)


class FakeFrame:
    def copy(self):
        return FakeFrame()


class FakeCv2:
    WINDOW_NORMAL = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    WND_PROP_VISIBLE = 0

    def __init__(self):
        self.named = False
        self.shown = False
        self.destroyed = False

    def namedWindow(self, name, flags):
        self.named = True

    def resizeWindow(self, name, width, height):
        pass

    def putText(self, *args, **kwargs):
        pass

    def imshow(self, name, frame):
        self.shown = True

    def waitKey(self, delay):
        return ord("q") if self.shown else -1

    def getWindowProperty(self, name, prop):
        return 1

    def destroyWindow(self, name):
        self.destroyed = True


class RealSensePreviewTest(unittest.TestCase):
    def test_1280x720_defaults_bound_raw_memory_and_use_fast_lossless_depth(self):
        recorder = RealSenseRecorder("episode", threading.Event())

        self.assertEqual(FRAME_QUEUE_CAPACITY, 16)
        self.assertEqual(recorder.queue_capacity, 16)
        self.assertEqual(DEPTH_PNG_COMPRESSION, 1)
        self.assertEqual(recorder.depth_png_compression, 1)

    def test_latest_preview_frame_replaces_stale_frame(self):
        frames = queue.Queue(maxsize=1)
        frames.put("old")
        RealSenseRecorder._offer_latest(frames, "new")
        self.assertEqual(frames.get_nowait(), "new")

    def test_q_closes_preview_without_setting_capture_stop(self):
        recorder = RealSenseRecorder("episode", threading.Event(), preview=True)
        frames = queue.Queue(maxsize=1)
        stop_token = object()
        frames.put(FakeFrame())
        fake_cv2 = FakeCv2()

        recorder._preview(frames, stop_token, fake_cv2)

        self.assertTrue(recorder.preview_started_event.is_set())
        self.assertTrue(recorder.preview_closed_event.is_set())
        self.assertTrue(recorder.preview_closed_by_user)
        self.assertIsNone(recorder.preview_error)
        self.assertFalse(recorder.stop_event.is_set())
        self.assertTrue(fake_cv2.named)
        self.assertTrue(fake_cv2.shown)
        self.assertTrue(fake_cv2.destroyed)


if __name__ == "__main__":
    unittest.main()
