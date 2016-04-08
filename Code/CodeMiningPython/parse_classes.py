"""Utility functions for parsing and checking the class definitions in the user's parameter file."""

# Python imports.

# User imports.


def parse_classes():
    """

    :return:

    """


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