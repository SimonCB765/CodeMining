"""Partition patients based on the cases."""

# Python imports.
import logging
import random

# Globals.
LOGGER = logging.getLogger(__name__)


def main(cases, numFolds):
    """Generate stratified folds to use for cross validation.

    :param cases:           A mapping from case names to the IDs of the patients meeting the case definition.
    :type cases:            dict
    :param numFolds:        The number of cross validation folds to generate.
    :type numFolds:         int
    :return:                A mapping from fold indices (0, 1, 2, ..., numFolds - 1) to a subset of the cases
                                dictionary that represents the fold. For example, if two folds were being generated
                                and the cases were:
                                    "CaseA": {4, 12, 14, 20, 26, 32, 33, 45, 48}
                                    "CaseB": {5, 13, 24, 30, 34}
                                    "CaseC": {8, 21, 35, 37}
                                 then, the output may be something like:
                                    0: {"CaseA": {4, 48, 14, 33, 45}, "CaseB": {30, 5, 24}, "CaseC": {21, 8}}
                                    1: {"CaseA": {12, 32, 26, 20}, "CaseB": {34, 13}, "CaseC": {35, 37}}
    :rtype:                 dict

    """

    # Generate a warning if any case has too few observations to be in all folds.
    for i, j in cases.items():
        if len(j) < numFolds:
            LOGGER.warning(
                "WARNING: case {:s} occurs {:d} times, and will not appear in each of the {:d} folds.".format(
                    i, len(j), numFolds
                )
            )

    # Determine the partitions.
    # Assume the indices of the patients meeting each case definition is something like:
    #   "CaseA": {4, 12, 14, 20, 26, 32, 33, 45, 48}
    #   "CaseB": {5, 13, 24, 30, 34}
    #   "CaseC": {8, 21, 35, 37}
    # Shuffle the indices:
    #   "CaseA": [4, 48, 14, 33, 45, 12, 32, 26, 20]
    #   "CaseB": [30, 5, 24, 34, 13]
    #   "CaseC": [21, 8, 35, 37]
    # Partition (in this case into two folds):
    #   "CaseA": [[4, 48, 14, 33, 45], [12, 32, 26, 20]]
    #   "CaseB": [[30, 5, 24], [34, 13]]
    #   "CaseC": [[21, 8], [35, 37]]
    # Create folds:
    #   0: {"CaseA": {4, 48, 14, 33, 45}, "CaseB": {30, 5, 24}, "CaseC": {21, 8}}
    #   1: {"CaseA": {12, 32, 26, 20}, "CaseB": {34, 13}, "CaseC": {35, 37}}
    #
    # Partitions are only guaranteed to have a bound on the difference in the number of examples they contain.
    # The largest possible difference in size between the fold with the most examples and the one with the
    # least is equal to the number of classes. For example, the partition:
    #     {0 : [[1, 2], [3]], 1 : [[4, 5], [6]], 2 : [[7, 8], [9]]}
    # Will give six examples in the first fold and three in the second.
    shuffledCases = {i: list(j) for i, j in cases.items()}
    for i, j in shuffledCases.items():
        random.shuffle(j)
    partitions = {i: [j[k::numFolds] for k in range(numFolds)] for i, j in shuffledCases.items()}
    folds = {i: {j: set(k[i]) for j, k in partitions.items()} for i in range(numFolds)}

    return folds
