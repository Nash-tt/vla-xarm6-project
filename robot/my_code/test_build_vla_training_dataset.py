import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_vla_training_dataset as dataset


class UltrasoundGeometryTest(unittest.TestCase):
    def test_default_training_geometry_is_square(self):
        gray = np.arange(20, dtype=np.uint8).reshape(5, 4)

        result, mode = dataset.prepare_ultrasound_training_gray(
            gray,
            square_size=8,
            preserve_physical_aspect=False,
            depth_span_mm=50.0,
            lateral_span_mm=30.0,
        )

        self.assertEqual(result.shape, (8, 8))
        self.assertEqual(mode, "square_model_input_axis_resampling")

    def test_can_restore_equal_physical_pixel_scale(self):
        gray = np.zeros((100, 10), dtype=np.uint8)

        result, mode = dataset.prepare_ultrasound_training_gray(
            gray,
            square_size=32,
            preserve_physical_aspect=True,
            depth_span_mm=50.0,
            lateral_span_mm=30.0,
        )

        self.assertEqual(result.shape, (100, 60))
        self.assertEqual(mode, "equal_physical_mm_per_pixel")

    def test_rejects_invalid_square_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            dataset.prepare_ultrasound_training_gray(
                np.zeros((5, 4), dtype=np.uint8),
                square_size=0,
                preserve_physical_aspect=False,
                depth_span_mm=50.0,
                lateral_span_mm=30.0,
            )


if __name__ == "__main__":
    unittest.main()
