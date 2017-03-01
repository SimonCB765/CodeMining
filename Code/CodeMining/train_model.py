"""Functions for training models for the code mining."""

# Python imports.
import datetime
import logging
import math
import os
import random

# User imports.
from . import calc_metrics
from . import generate_CV_folds

# 3rd party imports.
import numpy as np
from sklearn.linear_model import SGDClassifier

# Globals.
LOGGER = logging.getLogger(__name__)


def main(dataMatrix, dataClasses, dirResults, patientMask, codeMask, cases, config):
    """Train models to perform the code mining.

    :param dataMatrix:      The sparse matrix containing the data to use for training/testing.
    :type dataMatrix:       scipy.sparse.csr_matrix
    :param dataClasses:     The class (integer case) that each patient belongs to.
    :type dataClasses:      np.array
    :param dirResults:      The location to store the results of the training/testing.
    :type dirResults:       str
    :param patientMask:     A boolean mask indicating which patients are permissible to use for training/testing.
    :type patientMask:      np.array
    :param codeMask:        A boolean mask indicating which codes are permissible to use for training/testing.
    :type codeMask:         np.array
    :param cases:           A mapping between the case names and the patients that meet the case definitions.
    :type cases:            dict
    :param config:          The JSON-like object containing the configuration parameters to use.
    :type config:           JsonschemaManipulation.Configuration
    :return:                The trained classifier.
    :rtype:                 sklearn.linear_model.SGDClassifier

    """

    # Create all combinations of parameters that will be used.
    epochs = config.get_param(["Epoch"])[1]
    batchSizes = config.get_param(["BatchSize"])[1]
    lambdaVals = config.get_param(["Lambda"])[1]
    elasticNetMixing = config.get_param(["ElasticNetMixing"])[1]
    paramCombos = [[i, j, k, l] for i in epochs for j in batchSizes for k in lambdaVals for l in elasticNetMixing]

    # Setup training.
    cvFoldsToUse = config.get_param(["CrossValFolds"])[1]
    if len(cvFoldsToUse) == 1:
        # Perform non-nested cross validation.

        # Generate folds.
        LOGGER.info("Now generating folds for non-nested cross validation.")
        folds = generate_CV_folds.main(cases, cvFoldsToUse[0])

        # Perform training.
        LOGGER.info("Now performing CV.")
        filePerformance = os.path.join(dirResults, "Performance_Folds.tsv")
        classifier, bestParamCombination, bestPerformance = perform_training(
            dataMatrix, dataClasses, folds, paramCombos, patientMask, codeMask, filePerformance
        )

        # Train a final model with the best parameter combination found.
        LOGGER.info("Now training final model.")
        fileFinalPerformance = os.path.join(dirResults, "Performance_Final_Classifier.tsv")
        classifier, bestParamCombination, bestPerformance = perform_training(
            dataMatrix, dataClasses, {0: cases}, [bestParamCombination], patientMask, codeMask, fileFinalPerformance
        )
    else:
        # Perform nested cross validation.

        # Generate outer folds.
        LOGGER.info("Now generating outer folds for nested cross validation.")
        outerFolds = generate_CV_folds.main(cases, cvFoldsToUse[0])

        # Perform inner cross validation.
        for ind, (i, j) in enumerate(outerFolds.items()):
            LOGGER.info("Now generating inner folds for outer fold {:d}.".format(ind))
            innerFolds = generate_CV_folds.main(j, cvFoldsToUse[1])

            # Perform training.
            LOGGER.info("Now performing CV.")
            fileOutput = os.path.join(dirResults, "Performance_Fold_{:d}.tsv".format(ind))
            classifier, bestParamCombination, bestPerformance = perform_training(
                dataMatrix, dataClasses, innerFolds, paramCombos, patientMask, codeMask, fileOutput
            )

    return classifier


def perform_training(dataMatrix, dataClasses, folds, paramCombos, patientMask, codeMask, filePerformance):
    """Perform the cross validated training and testing.

    :param dataMatrix:      The sparse matrix containing the data to use for training/testing.
    :type dataMatrix:       scipy.sparse.csr_matrix
    :param dataClasses:     The class (integer case) that each patient belongs to.
    :type dataClasses:      np.array
    :param folds:           The folds to use in the training. Formatted as:
                                0: {"CaseA": {4, 48, 14, 33, 45}, "CaseB": {30, 5, 24}, "CaseC": {21, 8}}
                                1: {"CaseA": {12, 32, 26, 20}, "CaseB": {34, 13}, "CaseC": {35, 37}}
    :type folds:            dict
    :param paramCombos:     A list of parameter combinations to use. Each entry is a list of
                                number of epochs, batch size, lambda value, elastic net mixing
    :type paramCombos:      list
    :param patientMask:     A boolean mask indicating which patients are permissible to use for training/testing.
    :type patientMask:      np.array
    :param codeMask:        A boolean mask indicating which codes are permissible to use for training/testing.
    :type codeMask:         np.array
    :param filePerformance: The location to store the results of the training/testing.
    :type filePerformance:  str
    :return:                The classifier, the parameter combination that gave the best performance and the performance
                                obtained using that parameter combination.
    :rtype:                 sklearn.linear_model.SGDClassifier, list, float

    """

    # Cut the data matrix down to a matrix containing only the codes to be used.
    dataMatrixSubset = dataMatrix[:, codeMask]

    # Determine the classes used.
    classesUsed = np.unique(dataClasses[~np.isnan(dataClasses)])

    with open(filePerformance, 'w') as fidPerformance:
        # Go through all parameter combinations.
        for params in paramCombos:
            # Define the parameters for this run.
            numEpochs, batchSize, lambdaVal, elasticNetRatio = params

            # Write out the parameters used.
            LOGGER.info(
                "Epochs={:d}\tBatch Size={:d}\tLambda={:1.5f}\tENet={:1.5f}\tTime={:s}".format(
                    numEpochs, batchSize, lambdaVal, elasticNetRatio,
                    datetime.datetime.strftime(datetime.datetime.now(), "%x %X")
                )
            )
            fidPerformance.write(
                "Epochs={:d}\tBatch Size={:d}\tLambda={:1.5f}\tENet={:1.5f}\tTime={:s}\n".format(
                    numEpochs, batchSize, lambdaVal, elasticNetRatio,
                    datetime.datetime.strftime(datetime.datetime.now(), "%x %X")
                )
            )

            # Create a record of the performance for each fold, record of the average performance of the models trained
            # with each parameter combination and one for the prediction for each example.
            performanceOfEachFold = []
            paramComboPerformance = []  # The performance of the model trained with each combination of parameters.
            predictions = np.empty(dataMatrix.shape[0])  # Holds predictions for all examples.
            predictions.fill(np.nan)

            # Perform the cross validation.
            for foldNum, fold in folds.items():
                # Create the model.
                classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                           l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                           learning_rate="optimal", class_weight=None)

                # Determine the training and testing examples.
                patientsInFold = [j for i in fold.values() for j in i]
                trainingExamples = np.copy(patientMask)
                trainingExamples[patientsInFold] = 0
                testingExamples = np.zeros(dataMatrix.shape[0], dtype=bool)
                testingExamples[patientsInFold] = 1

                # If only one fold is being used their will be no training examples as the current fold is used for
                # testing and all others for training (and there are no other folds if there is only one fold). We
                # therefore handle this special case by setting the training and testing sets to be identical.
                if np.count_nonzero(trainingExamples) == 0:
                    trainingExamples = testingExamples

                # Create training and testing matrices and target class arrays for this fold by cutting down the data
                # matrix and class vector to contain only those examples that are intended for training/testing.
                trainingMatrix = dataMatrixSubset[trainingExamples, :]
                trainingClasses = dataClasses[trainingExamples]
                testingMatrix = dataMatrixSubset[testingExamples, :]
                testingClasses = dataClasses[testingExamples]

                # Train the model on this fold and record performance.
                trainingDescent, testingDescent = mini_batch_e_net(
                    classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix, testingClasses,
                    batchSize, numEpochs
                )
                fidPerformance.write("\t{:s}\n".format(",".join(["{:1.4f}".format(i) for i in testingDescent])))

                # Record the model's performance on this fold.
                testPredictions = classifier.predict(testingMatrix)
                predictions[testingExamples] = testPredictions
                performanceOfEachFold.append(calc_metrics.calc_g_mean(testPredictions, testingClasses))

            # Record G mean of predictions across all folds using this parameter combination.
            # If the predicted class of any example is NaN, then the example did not actually have its class
            # predicted.
            examplesUsed = ~np.isnan(predictions)
            gMean = calc_metrics.calc_g_mean(predictions[examplesUsed], dataClasses[examplesUsed])
            fidPerformance.write("\tOverall G mean\t{:1.4f}".format(gMean))
            paramComboPerformance.append(gMean)

        # Determine best parameter combination. If there are two combinations
        # of parameters that give the best performance, then take the one that comes first.
        indexOfBestPerformance = paramComboPerformance.index(max(paramComboPerformance))
        bestParamCombination = paramCombos[indexOfBestPerformance]

        return classifier, bestParamCombination, max(paramComboPerformance)


def mini_batch_e_net(classifier, trainingMatrix, targetClasses, classesUsed, testingMatrix, testingClasses,
                     batchSize=500, numEpochs=5):
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
    :param numEpochs:           The number of mini batch iterations to run.
    :type numEpochs@            int
    :return:                    The record of the gradient descent on the training and testing sets for each mini batch
                                    in each epoch.
    :rtype:                     list

    """

    trainingDescent = []  # The list recording the gradient descent on the training set.
    testingDescent = []  # The list recording the gradient descent on the testing set.
    numTrainingExamples = trainingMatrix.shape[0]  # The number of training examples in the dataset.
    permutedIndices = list(range(numTrainingExamples))  # List containing the indices of all training examples.

    # Run the desired number of training iterations.
    for i in range(numEpochs):
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
            # mini batch) and on the test set.
            trainingPredictions = classifier.predict(trainingMatrix)
            gMean = calc_metrics.calc_g_mean(trainingPredictions, targetClasses)
            trainingDescent.append(gMean)
            testPredictions = classifier.predict(testingMatrix)
            gMean = calc_metrics.calc_g_mean(testPredictions, testingClasses)
            testingDescent.append(gMean)

    return trainingDescent, testingDescent
