import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import ml_dtypes
from safetensors.numpy import load_file, save_file

from train import save_checkpoint


class StridedModel:
    def __init__(self, tensor):
        self.tensor = tensor

    def flat_state_dict(self):
        return {"matrix": self.tensor}


class CheckpointLayoutTest(unittest.TestCase):
    def test_fortran_strided_tensor_round_trips(self):
        for dtype in (np.float32, ml_dtypes.bfloat16):
            with self.subTest(dtype=dtype):
                tensor = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6).astype(dtype))
                self.assertTrue(tensor.flags.f_contiguous)
                self.assertFalse(tensor.flags.c_contiguous)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "checkpoint.safetensors"
                    save_checkpoint(path, StridedModel(tensor))
                    saved = load_file(path)["matrix"]
                np.testing.assert_array_equal(saved, tensor)

    def test_rejects_corrupt_write_before_replacing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.safetensors"
            original = np.array([1.0, 2.0], dtype=np.float32)
            save_file({"matrix": original}, path)

            def write_wrong_values(_state, destination):
                save_file({"matrix": np.array([2.0, 1.0], dtype=np.float32)}, destination)

            with patch("train.save_file", side_effect=write_wrong_values):
                with self.assertRaisesRegex(ValueError, "read-back differs for matrix"):
                    save_checkpoint(path, StridedModel(original))

            np.testing.assert_array_equal(load_file(path)["matrix"], original)
            self.assertFalse(path.with_suffix(".safetensors.tmp").exists())


if __name__ == "__main__":
    unittest.main()
