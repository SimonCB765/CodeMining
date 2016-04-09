"""Functions to check the class definitions in the user's parameter file and find patients in each class."""

# Python imports.
import itertools

# 3rd party imports.
import numpy as np

# User imports.
import extract_child_codes


def find_patients(dataMatrix, classData, mapCodeToIndex):
    """Determine the indices of the patients in each class.

    Patients are deemed to be in a class if they have a nonzero value for a code belonging to the class.
    E.g. if class A consists of codes C10E and C10F, then any example with a nonzero value for one
    of these codes will belong to class A.

    Ambiguous examples are those belonging to more than one class.

    :param dataMatrix:      Matrix containing the dataset.
    :type dataMatrix:       scipy sparse matrix
    :param classData:       Information about how to determine the classes.
    :type classData:        dict
    :param mapCodeToIndex:  A mapping from codes to their indices in the dataset.
    :type mapCodeToIndex:   dict
    :return :               The lists of which examples belong to which class, and which examples are ambiguous.
    :rtype :                dict

    """

    # For each class determine the indices of the codes that define the class, and the patients in the class.
    # Any codes in the parameter list that do not appear in the dataset will be ignored.
    classExamples = {}
    for i in classData:
        # Get variable indices.
        classCodeIndices = [j for j in classData[i] if j[-1] != '.']
        getChildren = [j[:-1] for j in classData[i] if j[-1] == '.']  # Need to get children for any code ending in '.'.
        getChildren = extract_child_codes.main(getChildren, mapCodeToIndex.keys())
        classCodeIndices.extend(getChildren)
        classCodeIndices = [mapCodeToIndex.get(j, None) for j in classCodeIndices]
        classCodeIndices = [j for j in classCodeIndices if j]  # Remove all None values from the list of indices.

        # Determine patients. For a dense matrix np.where(dataMatrix > 0) could be used, but np.where
        # doesn't work for sparse matrices.
        classSubset = dataMatrix[:, classCodeIndices] > 0  # Subset of the dataset containing the class's examples.
        exampleIndices = set(np.nonzero(classSubset)[0])  # The indices of the patients in the class.
        classExamples[i] = exampleIndices

    # Determine the ambiguous examples.
    ambiguousExamples = set([])
    for i, j in itertools.combinations(classExamples.keys(), 2):
        ambiguousExamples |= classExamples[i] & classExamples[j]

    # Remove the ambiguous examples from all classes.
    for i in classExamples:
        classExamples[i] -= ambiguousExamples
        classExamples[i] = sorted(classExamples[i])
    classExamples["Ambiguous"] = sorted(ambiguousExamples)

    return classExamples


def check_validity(classDict):
    """Check whether the class data is valid.

    :param classDict:   The JSON object containing the class information.
    :type classDict:    dict
    :return :           Whether the class data is valid, and a message to go along with it.
    :rtype :            list

    """

    classesWithoutCodes = [i for i in classDict if not classDict[i]]  # Names of all classes without supplied codes.
    outputMessage = "The maximum number of classes without provided codes is 1. Classes without supplied codes " \
                   "were: {0:s}".format(','.join(classesWithoutCodes))
    isValid = len(classesWithoutCodes) <= 1
    return [isValid, "Class definition is valid." if isValid else outputMessage]