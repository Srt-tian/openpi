"""CPU-only regressions for the four-plugin entrypoint."""

import itertools
import unittest

import numpy as np

from train_four_plugins import DeterministicBatchSampler, numpy_collate


class RunnerTests(unittest.TestCase):
    def test_collates_numpy_boolean_masks(self):
        sample = {"image_mask": {"camera": np.True_}, "state": np.zeros(8, np.float32)}
        batch = numpy_collate([sample, sample])
        self.assertEqual(batch["image_mask"]["camera"].dtype, np.bool_)
        self.assertEqual(batch["image_mask"]["camera"].tolist(), [True, True])
        self.assertEqual(batch["state"].shape, (2, 8))

    def test_resume_matches_across_epoch_boundary(self):
        full = DeterministicBatchSampler(19, 4, 42)
        resumed = DeterministicBatchSampler(19, 4, 42, start_batch=7)
        expected = list(itertools.islice(full, 7, 12))
        self.assertEqual(expected, list(itertools.islice(resumed, 5)))

    def test_epoch_has_no_duplicate_indices(self):
        batches = list(itertools.islice(DeterministicBatchSampler(19, 4, 42), 4))
        indices = list(itertools.chain.from_iterable(batches))
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertTrue(all(0 <= i < 19 for i in indices))


if __name__ == "__main__":
    unittest.main()
