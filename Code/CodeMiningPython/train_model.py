"""Methods for training models for the code mining."""

# Python imports.
import math

# 3rd party imports.
import numpy as np

# User imports.
import calc_metrics


def mini_batch_e_net(classifier, trainingMatrix, targetClasses, classesUsed, permutations, testingMatrix=None, testingClasses=None,
                     batchSize=500, numIterations=5):
    """

    :return :

    """

    descent = []  # The list recording the gradient descent.
    numTrainingExamples = trainingMatrix.shape[0]  # The number of training examples in the dataset.

    # Run the desired number of training iterations.
    for i in range(numIterations):
        # Shuffle the training matrix and class array for this iteration. All examples and codes in
        # trainingMatrix will still be used, just the order of the examples changes.
        permTrainingMatrix = trainingMatrix[permutations[i], :]
        permTrainingClasses = targetClasses[permutations[i]]

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


def mini_batch_e_net_cv(classifier, dataMatrix, targetClasses, patientIndicesToUse, folds, classesUsed, permutations,
                        batchSize=500, numIterations=5, cvFolds=2):
    """

    :param trainingMatrix:
    :param targetClasses:
    :param classesUsed:
    :param permutations:
    :param lambdaVal:
    :param elasticNetRatio:
    :param batchSize:
    :param numIterations:
    :return :

    """

    performanceOfEachFold = []  # Create a list to record the performance for each fold.
    descent = []  # Create a list to record the gradient descent for each fold.
    predictions = np.empty(dataMatrix.shape[0])  # Create the array to hold the predictions of all examples.
    predictions.fill(np.nan)

    for i in range(cvFolds):
        # Determine training and testing example masks for this fold.
        trainingExamples = (folds != i) & patientIndicesToUse
        testingExamples = (folds == i) & patientIndicesToUse

        # Create training and testing matrices and target class arrays for this fold.
        trainingMatrix = dataMatrix[trainingExamples, :]
        trainingClasses = targetClasses[trainingExamples]
        testingMatrix = dataMatrix[testingExamples, :]
        testingClasses = targetClasses[testingExamples]

        # Train the model and record the descent on this fold.
        foldDescent = mini_batch_e_net(classifier, trainingMatrix, trainingClasses, classesUsed, permutations[i],
                                                   testingMatrix, testingClasses, batchSize, numIterations)
        descent.append(foldDescent)

        # Record the model's performance on this fold. If only one fold is being used, then this
        # will be the performance on the holdout portion.
        testPredictions = classifier.predict(testingMatrix)
        predictions[testingExamples] = testPredictions
        performanceOfEachFold.append(calc_metrics.calc_g_mean(testPredictions, testingClasses))

    return descent, performanceOfEachFold, predictions