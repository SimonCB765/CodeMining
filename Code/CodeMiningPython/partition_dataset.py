"""Partition a dataset based on its associated target array."""

# Python imports.
import numbers
import random
import sys

# 3rd party imports.
import numpy as np


def main(Y, indicesToUse=None, numPartitions=2, isStratified=False):
    """Partition a dataset into CV folds.

    return partition values that are np.nan are ones not in partition

    The partitioning will work when Y contains more than one column (i.e. more than one target variable).
    When multiple columns are present, stratified generation takes each column to represent a different class.
    Each row in the matrix should therefore have a single non-zero value. The column with the non-zero value
    is treated as the class to which the example belongs.

    Stratified partitioning is intended for situations where Y contains class labels.
    It will work for real valued Ys, but only when there are enough entries in Y with the same value.
    However, it will not function correctly for real valued Ys when there is more than one column in Y.

    Examples with all their target values in Y equal to np.nan are treated as if the example has no class.
    In the case of a 1 dimensional Y, this mean the single entry for the example is np.nan. In the case of 2 or
    greater dimensional Y, this means that all entries on the row corresponding to the example are np.nan.
    This can be useful if some examples are to be ignored, as you can simply set their target values to
    np.nan rather than removing them from the dataset.

    :param Y:               An NxM array of targets to partition.
    :type Y:                numpy array or matrix (or other numpy-like object with similar properties)
    :param indicesToUse:    A 1 dimensional array containing the indices of the examples in Y to consider when
                                generating the partition. For example, if Y is 1x10 and indicesToUse == [0, 3, 4, 5],
                                then only Y[0], Y[3], Y[4] and Y[5] will be considered.
    :type indicesToUse:     list or numpy array
    :param numPartitions:   The number of folds to partition the dataset into. Must be at least 2.
    :type numPartitions:    int
    :param isStratified:    Whether the partitioning should be stratified.
    :type isStratified:     bool
    :return :               A 1 dimensional array containing the partition to which each example in Y belongs.
                                If Y[i] == np.nan, then the ith entry in the return value will also be np.nan.
                                If i is not in indicesToUse, then Y[i] == np.nan.
                                Otherwise Y[i] will contain an integer recording the partition it is in.
    :rtype :                numpy array

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

    # Determine indices to partition.
    indicesToPartition = np.array(indicesToUse) if indicesToUse is not None else np.array([i for i in range(numExamples)])

    # Generate the partition.
    if isStratified:
        # Generate stratified folds.

        try:
            # Try to turn the input target into a numpy array (if it is a numpy matrix).
            Y = Y.A
        except AttributeError:
            # Y has no attribute A, so is likely a numpy array.
            pass

        # Determine the subset of Y to look at.
        Y = Y[indicesToPartition]

        # Setup the class membership array. This will record the integer value of the class of each example in
        # indicesToPartition. If the example has no class, then np.nan will be used instead of an integer.
        classMembership = np.empty(indicesToPartition.shape[0])
        classMembership.fill(np.nan)

        # Determine the classes present.
        if numTargets == 1:
            # Y is a column vector, so determine classes from the unique values of Y.
            differentClasses = np.unique(Y[~np.isnan(Y)])  # Get the classes.

            # Check if there are enough classes to perform stratified fold generation.
            if differentClasses.shape[0] == 1:
                print("Stratified CV was requested, but only one class was found.")
                sys.exit()

            # Assign an integer (starting at 0) to each class, and determine the (integer) class of each example
            # in the set of examples that are being partitioned.
            for ind, i in enumerate(differentClasses):
                classMembership[Y == i] = ind
        else:
            # Multiple target variables are present. Each column in Y is taken to be a separate response variable.
            Y[np.isnan(Y)] = 0  # Set all NaN values to 0 to treat those entries as not being the class of the example.
            nonzeroResponse = np.nonzero(Y)
            rowsWithValues = nonzeroResponse[0]
            if len(set(rowsWithValues)) != len(rowsWithValues):
                # There is at least one row with a value for multiple response variables.
                # Stratified CV can not be performed.
                print("Stratified CV was requested, but there are rows in Y with multiple non-zero entries.")
                sys.exit()

            # Set the integer representation of the class of each example that does not have all NaN values for
            # the response variables. As numpy arrays are 0 based, this will automatically set class membership
            # integers to be 0 based.
            classMembership[nonzeroResponse[0]] = nonzeroResponse[1]

        # Determine the (non-NaN) classes in the data subset.
        classesInData = np.unique(classMembership[~np.isnan(classMembership)])

        # Generate a warning if any class has too few observations to be in all folds.
        for i in classesInData:
            occurrences = sum(classMembership == i)
            if occurrences < numPartitions:
                print("WARNING: class {0:d} occurs {1:d} times, and will not appear in each of "
                      "the {2:d} folds.".format(int(i), occurrences, numPartitions))

        # Determine the partitions.
        # Assume the subset of examples used is ->
        #   [0     , 4, 5, 8,     10, 12, 13, 14, 20, 21, 24, 26, 27    , 30, 32, 33, 34, 35, 37, 40    , 45, 48]
        # Start with a list of class memberships ->
        #   [np.nan, 0, 1, 2, np.nan, 0 , 1 , 0 , 0 , 2 , 1 , 0 , np.nan, 1 , 0 , 0 , 1 , 2 , 2 , np.nan, 0 , 0]
        # Go through each (non-NaN) class and determine its example indices based on the class memberships ->
        #   0 -> [1, 5, 7, 8, 11, 14, 15, 20, 21]
        #   1 -> [2, 6, 10, 13, 16]
        #   2 -> [3, 9, 17, 18]
        # Shuffle the example indices ->
        #   0 -> [1, 5, 21, 14, 7, 11, 15, 8, 20]
        #   1 -> [13, 16, 2, 6, 10]
        #   2 -> [9, 17, 3, 18]
        # Partition the example indices (with 2 folds here) ->
        #   0 -> [[1, 21, 7, 15, 20], [5, 14, 11, 8]]
        #   1 -> [[13, 2, 10], [16, 6]]
        #   2 -> [[9, 3], [17, 18]]
        # Convert to original indices ->
        #   0 -> [[4, 48, 14, 33, 45], [12, 32, 26, 20]]
        #   1 -> [[30, 5, 24], [34, 13]]
        #   2 -> [[21, 8], [35, 37]]
        # Finally, create the partition array ->
        #   [np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 1, 1, 0, np.nan,
        #    np.nan, np.nan, np.nan, np.nan, 1, 0, np.nan, np.nan, 0, ...
        #
        # Partitions are only guaranteed to have a bound on the difference in the number of examples they contain.
        # The largest possible difference in size between the fold with the most examples and the one with the
        # least is equal to the number of classes.
        #     Classes -> {0 : [[1, 2], [3]], 1 : [[4, 5], [6]], 2 : [[7, 8], [9]]}
        #     Partition -> [0, 0, 1, 0, 0, 1, 0, 0, 1]
        partition = np.empty(numExamples)  # The array to hold the partition number for each example.
        partition.fill(np.nan)
        for i in classesInData:
            # Determine the indices of the examples in the subset (based off of indicesToPartition) that belong to
            # class i.
            examplesInClass = np.nonzero(classMembership == i)[0]
            random.shuffle(examplesInClass)  # Randomise the list of examples belonging to the class.
            classPartitions = [examplesInClass[j::numPartitions] for j in range(numPartitions)]  # Partition the class.
            for ind, i in enumerate(classPartitions):
                originalIndices = indicesToPartition[i]
                partition[originalIndices] = ind
    else:
        # Create random partitions where each partition has an (almost) equal number of examples.
        # Start with a list of the indices of the examples to partition ->  [3, 4, 6, 8, 11, 12, 13, 14, 18, 19]
        # Shuffle the list -> [19, 3, 8, 6, 11, 13, 14, 12, 4, 20]
        # Partition the indices -> [[18, 6, 14, 19], [3, 11, 12], [8, 13, 4]] (if cvFolds == 3)
        # Assign partition groupings according to original indices ->
        #   [np.nan, np.nan, np.nan, 1     , 2     , np.nan, 0     , np.nan, 2     , np.nan,
        #    np.nan, 1     , 1     , 2     , 0     , np.nan, np.nan, np.nan, 0     , 0]

        # Create the list containing the indices of the non NaN observations.
        if numTargets == 1:
            validIndices = [i for i in indicesToPartition if ~np.isnan(Y[i])]
        else:
            # An observation is treated as a NaN observation if all target values are NaN.
            allIndices = np.array(indicesToPartition)  # List containing the index of examples to partition.
            validIndices = np.all(np.isnan(Y), axis=1)  # Examples where all target values are not NaN.
            validIndices = allIndices[validIndices]

        # Create the partitions.
        random.shuffle(validIndices)  # Randomise the order of the indices.
        partitionedIndices = [validIndices[i::numPartitions] for i in range(numPartitions)]
        partition = np.empty(numExamples)  # The array to hold the partition number for each example.
        partition.fill(np.nan)
        for ind, i in enumerate(partitionedIndices):
            for j in i:
                partition[j] = ind

    return partition