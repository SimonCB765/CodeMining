"""Methods for training models for the code mining."""

# Python imports.
import math
import random

# 3rd party imports.
import numpy as np

# User imports.
import calc_metrics


def mini_batch_e_net(classifier, trainingMatrix, targetClasses, classesUsed, testingMatrix=None, testingClasses=None,
                     batchSize=500, numIterations=5):
    """Run mini batch training for elastic net regularised logistic regression.

    :param classifier:          The classifier to train.
    :type classifier:           SGDClassifier
    :param trainingMatrix:      The matrix to use for training.
    :type trainingMatrix:       numpy array
    :param targetClasses:       The array of target classes for the examples in targetMatrix.
    :type targetClasses:        numpy array
    :param classesUsed:         The unique classes across the whole dataset.
    :type classesUsed:          list
    :param testingMatrix:       The matrix of examples to use for testing.
    :type testingMatrix:        numpy array
    :param testingClasses:      The target classes for the examples in testingMatrix.
    :type testingClasses:       numpy array
    :param batchSize:           The size of the mini batches to use.
    :type batchSize:            int
    :param numIterations:       The number of mini batch iterations to run.
    :type numIterations@        int
    :return :                   The record of the gradient descent for each mini batch used.
    :rtype :                    list

    """

    descent = []  # The list recording the gradient descent.
    numTrainingExamples = trainingMatrix.shape[0]  # The number of training examples in the dataset.
    permutedIndices = list(range(numTrainingExamples))  # List containing the indices of all training examples.

    # Run the desired number of training iterations.
    for i in range(numIterations):
        # Shuffle the training matrix and class array for this iteration. All examples and codes in
        # trainingMatrix will still be used, just the order of the examples changes.
        random.shuffle(permutedIndices)
        permTrainingMatrix = trainingMatrix[permutedIndices, :]
        permTrainingClasses = targetClasses[permutedIndices]

        # Run through the batches.
        for j in range(int(math.ceil(numTrainingExamples / batchSize))):
            # Determine the indices to access the batch. Sparse matrices throw errors if you try
            # to index beyond the maximum index, so prevent this by truncating stopIndex.
            startIndex = j * batchSize
            stopIndex = min((j + 1) * batchSize, numTrainingExamples - 1)

            # Generate the training matrix and class array for the batch. This will contain a subset of the examples,
            # but all the codes in the training matrix.
            batchTrainingMatrix = permTrainingMatrix[startIndex:stopIndex, :]
            batchTrainingClasses = permTrainingClasses[startIndex:stopIndex]

            # Train the model on the batch.
            classifier.partial_fit(batchTrainingMatrix, batchTrainingClasses, classes=classesUsed)

            # Record the descent by predicting on the entire training matrix (not just the permuted
            # mini batch) if there is no test set. Otherwise predict on the test set.
            if testingMatrix is None:
                trainingPredictions = classifier.predict(trainingMatrix)
                gMean = calc_metrics.calc_g_mean(trainingPredictions, targetClasses)
                descent.append(gMean)
            else:
                testPredictions = classifier.predict(testingMatrix)
                gMean = calc_metrics.calc_g_mean(testPredictions, testingClasses)
                descent.append(gMean)

    return descent