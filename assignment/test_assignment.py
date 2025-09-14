import logging
import unittest
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight # type:ignore
from cse587Autils.SequenceObjects.SequenceModel import SequenceModel
from cse587Autils.configure_logging import configure_logging

# Handle both VS Code (relative import) and autograder (absolute import) contexts
try:
    from .assignment import site_sample  # VS Code context
except ImportError:
    from assignment import site_sample  # Autograder context

configure_logging(logging.WARN)


class Testsite_sample(unittest.TestCase):

    @weight(4)
    def test_1(self):
        """Testing that the dimensions of the sample are correct"""
        sequence_model = SequenceModel(0.5,
                               [[1/4, 1/4, 1/4, 1/4],
                                [1/4, 1/4, 1/4, 1/4],
                                [1/4, 1/4, 1/4, 1/4]],
                               [1/5, 1/5, 1/5, 2/5])

        actual = site_sample(sequence_model, 7)

        self.assertEqual(actual.shape, (7, 3))

    @ weight(8)
    def test_2_and_3(self):
        """Testing that the sampled values lie in the correct range and that 0
            probabilities are handled correctly"""
        site_base_probs = [[1/3, 1/3, 1/3, 0] for _ in range(5)]
        background_base_probs = [0, 1/5, 2/5, 2/5]

        sm1 = SequenceModel(1, site_base_probs, background_base_probs)
        sm2 = SequenceModel(0, site_base_probs, background_base_probs)

        actual_1 = site_sample(sm1, 40)
        actual_2 = site_sample(sm2, 40)

        # In actual_1, a 0 (A) can occur (and does given the default seed)
        # but a 3 (T) cannot occur since the background_prior is 0 and
        # the site_base_probs are all 0 for T.
        self.assertEqual(np.min(actual_1), 0)
        self.assertEqual(np.max(actual_1), 2)

        # In actual_2, a 3 (T) can occur (and does given the default seed)
        # but a 0 (A) cannot occur since the site_prior is 0 and
        # the background_prob of a 0 (A) is 0.
        self.assertEqual(np.min(actual_2), 1)
        self.assertEqual(np.max(actual_2), 3)

    @ weight(4)
    def test_4(self):
        """Testing that the relative probabilities are in the right
            ballpark in large samples"""
        site_base_probs = [[1/3, 1/3, 1/3, 0] for _ in range(5)]
        background_base_probs = [0, 1/5, 2/5, 2/5]

        sm1 = SequenceModel(0, site_base_probs, background_base_probs)

        sample = site_sample(sm1, 200)

        # Flatten the sample and count occurrences of each value
        flat_sample = sample.flatten()
        bin_counts = np.bincount(flat_sample, minlength=4)

        # Normalize counts to get an approximate distribution
        normalized_counts = bin_counts / bin_counts.sum()

        # Calculate the deviation from expected probabilities
        expected_probs = np.array(background_base_probs)
        deviations = np.round(np.abs(normalized_counts - expected_probs), 5)

        # Check that deviations are in an acceptable range
        # (this threshold can be adjusted)
        self.assertTrue(np.all(deviations < 0.1))

    @ weight(4)
    def test_5(self):
        """Testing that two calls to site_sample produce different values"""
        site_base_probs = [[1/3, 1/3, 1/3, 0] for _ in range(5)]
        background_base_probs = [0, 1/5, 2/5, 2/5]

        sm1 = SequenceModel(0, site_base_probs, background_base_probs)

        sample1 = site_sample(sm1, 200, seed=314)
        sample2 = site_sample(sm1, 200, seed=413)

        self.assertFalse(np.array_equal(sample1, sample2))
