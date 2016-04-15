"""Partition a dataset based on its associated target array."""

# Python imports.
import collections
import numbers
import random
import sys

# 3rd party imports.
import numpy as np


def main(Y, numPartitions=2, isStratified=False):
    """Partition a dataset into CV folds.

    Observations with all their target values in Y equal to np.nan are treated as if the observation has no class.
    In the case of a 1 dimensional Y, this mean the single entry for the observation is np.nan. In the case of 2 or
    greater dimensional Y, this means that all entries on the row corresponding to the observation are np.nan.
    This can be useful if some observations are to be ignored, as you can simply set their targets to
    np.nan rather than removing them from the dataset.

    The partitioning will work when Y contains more than one column (i.e. more than one target variable).

    When multiple columns are present, stratified fold generation takes each column to represent a different class.
    Each row in the matrix should therefore have a single non-zero value. The column with the non-zero value
    is treated as the class to which the example belongs.

    Stratified partitioning is intended  for situations where Y contains class labels.
    It will work for real valued Ys, but only when there are enough entries in Y with the same value.
    However, it will not function correctly for real valued Ys when there is more than one column in Y.

    :param Y:               An NxM array of targets to partition.
    :type Y:                numpy array or matrix (or other numpy-like object with similar properties)
    :param numPartitions:   The number of folds to partition the dataset into. Must be at least 2.
    :type numPartitions:    int
    :param isStratified:    Whether the partitioning should be stratified.
    :type isStratified:     bool
    :return :               A list containing the number of the partition that each example in Y belongs to.
    :rtype :                list

    """

    # Get the dimensions of Y.
    if hasattr(Y, "shape"):
        try:
            # If Y is a 1 dimensional array, then shape will be a tuple of value.
            numExamples, numTargets = Y.shape
        except ValueError:
            # Shape had too few elements, so the num of targets must be 1.
            numExamples = Y.shape[0]
            numTargets = 1
    else:
        print("Y must be a numpy object with a shape attribute.")
        sys.exit()

    # Make sure the number of partitions is valid.
    if not isinstance(numPartitions, numbers.Integral):
        print("The number of partitions must be an integer.")
        sys.exit()
    elif numPartitions < 2:
        print("{0:d} partitions were request, while the minimum is 2.".format(numPartitions))

    # Generate the folds.
    if isStratified:
        # Generate stratified folds.

        try:
            # Try to turn the input target into a numpy array (if it is a matrix).
            tempY = Y.A
        except AttributeError:
            # Y has no attribute A, so is likely a numpy array.
            tempY = Y

        # Determine the classes present.
        if numTargets == 1:
            # Y is a column vector, so determine classes from the unique values of Y.
            differentClasses = np.unique(Y[~np.isnan(Y)])  # Get the classes.

            # Check if there are enough classes to perform stratified fold generation.
            if differentClasses.shape[0] == 1:
                print("Stratified CV was requested, but only one class was found.")
                sys.exit()

            # Assign an index to each class starting at 0, and determine the class index of each example.
            classMembership = Y.tolist()  # Create the list to hold the class index for each observation.
            classMapping = dict([(i, ind) for ind, i in enumerate(differentClasses)])  # Map values in Y to class index.
            classMembership = [(np.nan if np.isnan(i) else classMapping[i]) for i in classMembership]
        else:
            # Multiple target variables are present. Each column in Y is taken to be a separate response variable.
            nonzeroResponse = np.nonzero(tempY)
            rowsWithValues = nonzeroResponse[0]
            if len(set(rowsWithValues)) != len(rowsWithValues):
                # There is at least one row with a value for multiple response variables.
                # Stratified CV can therefore not be performed.
                print("Stratified CV was requested, but there are rows in Y with multiple non-zero entries.")
                sys.exit()

            # Determine the index of the class each observation belongs to.
            classMembership = nonzeroResponse[1].tolist()

        # Generate a warning if any class has too few observations to be in all folds.
        for i in set(classMembership):
            occurrences = classMembership.count(i)
            if occurrences < numPartitions:
                print("WARNING: class {0:d} occurs {1:d} times, and will not appear in each of "
                      "the {2:d} folds.".format(i, occurrences, numPartitions))

        # Determine the partitions.
        # Start with a list of class memberships -> [0, 1, 2, 0, 0, 1, 0, 2, 1, 0, 2, 0, 0, 1, 2, 1, 0, 0]
        # Create dictionary mapping class to indices ->
        #   {0 : [0, 3, 4, 6, 9, 11, 12, 16, 17], 1 : [1, 5, 8, 13, 15], 2 : [2, 7, 10, 14]}
        # Randomise each classes indices ->
        #   {0 : [4, 6, 9, 3, 17, 16, 0, 12, 11], 1 : [13, 1, 5, 15, 8], 2 : [10, 14, 7, 2]}
        # Split each class into cvFolds partitions ->
        #   {0 : [[0, 6, 12], [3, 9, 16], [4, 11, 17]], 1 : [[1, 13], [5, 15], [8]], 2 : [[2, 14], [7], [10]]}
        #   The overall partitions can be determined from this. For example, the first partition will be
        #   [0, 6, 12, 1, 13, 2, 4].
        # Assign partition groupings according to original indices ->
        #   [0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 1, 1, 2]
        #
        # Folds are only guaranteed to have a bound on the difference in the number of observations they contain.
        # The largest possible difference in size between the fold with the most observations and the one with the
        # least is equal to the number of classes.
        #     Classes -> {0 : [[1, 2], [3]], 1 : [[4, 5], [6]], 2 : [[7, 8], [9]]}
        #     Partition -> [0, 0, 1, 0, 0, 1, 0, 0, 1]
        classIndices = collections.defaultdict(list)  # Indices of the observations belonging to each class.
        for ind, i in enumerate(classMembership):
            classIndices[i].append(ind)
        partition = [0] * numExamples  # Create the list to hold the partition number for each observation.
        for i in classIndices:
            random.shuffle(classIndices[i])  # Randomise the list of observations belonging to each class.
            classIndices[i] = [classIndices[i][j::numPartitions] for j in range(numPartitions)]  # Partition each class.
            for ind, j in enumerate(classIndices[i]):
                for k in j:
                    partition[k] = ind
    else:
        # Create random partitions where each partition has an (almost) equal number of examples.
        # Start with a list of the indices of the examples ->  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Randomise the list -> [7, 4, 8, 1, 5, 6, 9, 2, 3, 0]
        # Partition the indices -> [[7, 1, 9, 0], [4, 5, 2], [8, 6, 3]] (if cvFolds == 3)
        # Assign partition groupings according to original indices -> [0, 0, 1, 2, 1, 1, 2, 0, 2, 0]

        # Create the list containing the indices of the non NaN observations.
        if numTargets == 1:
            validIndices = [i for i in range(numExamples) if ~np.isnan(Y[i])]
        else:
            # An observation is treated as a NaN observation if all target values are NaN.
            allIndices = np.array(list(range(numExamples)))  # List containing the index of each example.
            validIndices = np.all(np.isnan(Y), axis=1)  # Observations where all their targets are not NaN.
            validIndices = allIndices[validIndices]

        # Create the partitions.
        random.shuffle(validIndices)  # Randomise the order of the indices.
        partitionedIndices = [validIndices[i::numPartitions] for i in range(numPartitions)]
        partition = [0] * numExamples  # Create the list to hold the partition number for each example.
        for ind, i in enumerate(partitionedIndices):
            for j in i:
                partition[j] = ind

    return partition