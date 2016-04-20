"""Test the partitioning of the data."""

# Python imports.
import unittest

# 3rd party imports.
import numpy as np

# User imports.
from CodeMiningPython import partition_dataset


class CompletitionTest(unittest.TestCase):
    """Tests where the partitioning succeeds."""

    def test_simple(self):
        Y = np.array([0, 0, 1, 2, 0, 1, 1, 0, 2, 2, 2, 0])
        partition = partition_dataset.main(Y, indicesToUse=None, numPartitions=2, isStratified=False)
        self.assertEqual(np.unique(partition).tolist(), [0, 1])
        print(partition)
        partition = partition_dataset.main(Y, indicesToUse=None, numPartitions=2, isStratified=True)
        self.assertEqual(np.unique(partition).tolist(), [0, 1])
        print(partition)