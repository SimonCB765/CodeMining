"""Calculate performance metrics from predictions."""

# 3rd party imports.
import numpy as np


def calc_g_mean(predictions, targets):
    """Calculate the G(eometric) mean of a set of class predictions.

    The arrays of predictions and targets must be ordered the same. For example, predictions[i] is the predicted
    class of the ith example, and targets[i] is the true class of the ith example.

    The G mean can be calculated with any number of classes in predictions and targets.
    However, the calculation is made off of the number of different classes in the targets array.

    :param predictions:     The predictions made by a classifier.
    :type predictions:      numpy array (1 dimensional)
    :param targets:         The target classes for the examples.
    :type targets:          numpy array (1 dimensional)
    :return :               The G mean.
    :rtype :                float

    """

    # Determine the target classes.
    classes = np.unique(targets)

    # Determine sensitivity for each class.
    sensitivities = []
    for i in classes:
        # Determine examples predicted to belong to class and examples actually in class.
        predictedAsClass = predictions == i
        examplesInClass = targets == i
        numClassExamples = sum(examplesInClass)

        # Calculate sensitivity.
        truePositives = sum(predictedAsClass & examplesInClass)
        numClassExamples = float(numClassExamples)  # Cast to float makes sensitivity a float in Python 2.x and 3.x.
        sensitivities.append(truePositives / numClassExamples)

    # Calculate G mean.
    return np.power(np.prod(sensitivities), 1. / len(classes))  # Cast 1 to float to ensure Python 2.x and 3.x work.
