"""Test the partitioning of the data.

To run this unittest run the command "python -m unittest TestPython.test_partitioning" from the Code directory.

"""

# Python imports.
import unittest

# 3rd party imports.
import numpy as np

# User imports.
from CodeMiningPython import partition_dataset


class CompletionTest(unittest.TestCase):
    """Tests whether the partitioning succeeds."""

    def test_simple(self):
        for i in range(2, 11):
            Y = np.random.randint(i, size=100 * i)
            classesInY = np.unique(Y)
            for j in range(2, 11):
                partition = partition_dataset.main(Y, indicesToUse=None, numPartitions=j, isStratified=False)
                self.assertEqual(np.unique(partition).tolist(), list(range(j)))
                partition = partition_dataset.main(Y, indicesToUse=None, numPartitions=j, isStratified=True)
                self.assertEqual(np.unique(partition).tolist(), list(range(j)))

                # Ensure that the difference in the number of examples in each partition is within the
                # expected bounds, i.e. that the partition with the most examples has no more than K examples more
                # than the partition with the least, where K is the number of classes.
                examplesInEachPartition = []
                for k in range(j):
                    examplesInEachPartition.append(sum(partition == k))
                self.assertTrue((max(examplesInEachPartition) - min(examplesInEachPartition)) <= classesInY.shape[0])

                # Ensure that the number of examples of each class in the original Y array is the same
                # as the sum of the number in each partition.
                for k in classesInY:
                    numObservationsOfClass = sum(Y == k)
                    partitionsClassIn = np.unique(partition[Y == k])
                    numInEachPartition = {i : sum(partition[Y == k] == i) for i in partitionsClassIn}
                    totalExamplesPartitioned = sum([numInEachPartition[i] for i in numInEachPartition])
                    self.assertEqual(numObservationsOfClass, totalExamplesPartitioned)