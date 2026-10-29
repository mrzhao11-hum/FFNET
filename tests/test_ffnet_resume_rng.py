"""Regression test for resuming an FFNet checkpoint on CUDA."""

import unittest

import torch

from vehicle_ffnet.train import capture_rng_state, restore_rng_state


class ResumeRngTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_restore_rng_state_after_cuda_checkpoint_load(self):
        original = capture_rng_state()
        try:
            # torch.load(..., map_location="cuda:0") maps these tensors too.
            mapped = {
                **original,
                "torch": original["torch"].to("cuda:0"),
                "cuda": [rng.to("cuda:0") for rng in original["cuda"]],
            }
            restore_rng_state(mapped)
            self.assertTrue(torch.equal(torch.get_rng_state(), original["torch"]))
            self.assertTrue(all(
                torch.equal(actual, expected)
                for actual, expected in zip(torch.cuda.get_rng_state_all(), original["cuda"])
            ))
        finally:
            restore_rng_state(original)


if __name__ == "__main__":
    unittest.main()
