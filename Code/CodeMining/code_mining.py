"""Perform the code mining."""

# Python imports.
import logging
import sys

# User imports.
from . import generate_dataset

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileDataset, fileCodeMapping, dirResults, config):
    """Perform the clinical code mining.

    :param fileDataset:         File location containing the data to use.
    :type fileDataset:          str
    :param fileCodeMapping:     File containing the mapping between codes and their descriptions.
    :type fileCodeMapping:      str
    :param dirResults:          Location of the directory in which the results will be saved.
    :type dirResults:           str
    :param config:              The JSON-like object containing the configuration parameters to use.
    :type config:               JsonschemaManipulation.Configuration

    """

    # Generate the mapping from codes to their descriptions. Each line in the file should contain two entries (a code
    # and its description) separated by a tab.
    mapCodeToDescr = {}
    with open(fileCodeMapping, 'r') as fidCodes:
        for line in fidCodes:
            chunks = (line.strip()).split('\t')
            mapCodeToDescr[chunks[0]] = chunks[1]

    # Generate the data matrix, two index mappings and the case mapping.
    # The patient index map is a bidirectional mapping between patients and their row indices in the data matrix.
    # The code index map is a bidirectional mapping between codes and their column indices in the data matrix.
    # The case mapping records which patients meet which case definition. Ambiguous patients are added to a separate
    #   Ambiguous case.
    sparseMatrix, mapPatientIndices, mapCodeIndices, cases = generate_dataset.main(
        fileDataset, dirResults, mapCodeToDescr, config
    )

    # Check whether there are any cases with no patients (the ambiguous patient case does not need accounting for as
    # it is only in the case definition when there are ambiguous patients).
    noExampleCases = [i for i, j in cases.items() if len(j) == 0]
    if noExampleCases:
        LOGGER.error("The following cases have no unambiguous patients: {:s}".format(
            ','.join(noExampleCases))
        )
        print("\nErrors were encountered following case identification. Please see the log file for details.\n")
        sys.exit()



    sys.exit()




    # Determine the examples in each class.
    # classExamples is a dictionary with an entry for each class (and one for "Ambiguous" examples) that contains
    # a list of the examples belonging to that class.
    # classCodeIndices is a list of the indices of the codes used to determine class membership.
    classExamples, classCodeIndices = parse_classes.find_patients(
        dataMatrix, classData, mapCodeToInd, isCodesRemoved=True)

    # Determine the class of each example, and map the classes from their names to an integer representation.
    # A class of NaN is used to indicate that an example does not belong to any class, and is therefore
    # not used in the training.
    allExampleClasses = np.empty(dataMatrix.shape[0])  # The integer class code for each example.
    allExampleClasses.fill(np.nan)
    mapIntRepToClass = {}  # Mapping from the integer code used to reference each class to the class it references.
    currentCode = 0
    for i in classExamples:
        if i != "Ambiguous":
            mapIntRepToClass[currentCode] = i
            allExampleClasses[classExamples[i]] = currentCode
            currentCode += 1
    classesUsed = [i for i in mapIntRepToClass]  # List containing the integer code for every class in the dataset.

    # Calculate masks for the patients and the codes. These will be used to select only those patients and codes
    # that are to be used for training/testing.
    patientIndicesToUse = np.ones(dataMatrix.shape[0], dtype=bool)
    patientIndicesToUse[np.isnan(allExampleClasses)] = 0  # Mask out the patients that have no class.
    codeIndicesToUse = np.ones(dataMatrix.shape[1], dtype=bool)
    codeIndicesToUse[classCodeIndices] = 0  # Mask out the codes used to calculate class membership.

    # Create all combinations of parameters that will be used.
    paramCombos = [[i, j, k, l] for i in numIters for j in batchSizes for k in lambdaVals for l in elasticNetMixing]

    if len(cvFolds) == 0:
        # Training if no testing is needed.

        with open(dirResults + "/BothModelDescentResults.tsv", 'w') as fidPerformance:

            # Write the header for the output files.
            fidPerformance.write("Model\tNumIterations\tBatchSize\tLambda\tENetRatio\tTrainingGMean\tDescentGMean\n")

            # Determine the parameters to use. Only use the first combination in the list.
            numIterations, batchSize, lambdaVal, elasticNetRatio = paramCombos[0]

            # Create the training matrix and target class array.
            # Generate the training matrix in two steps as scipy sparse matrices can only be sliced along
            # both axes in the same operation with arrays of the same length.
            firstTrainingMatrix = dataMatrix[patientIndicesToUse, :]
            firstTrainingMatrix = firstTrainingMatrix[:, codeIndicesToUse]
            firstTrainingClasses = allExampleClasses[patientIndicesToUse]

            # Create the first model.
            firstClassifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                            l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                            learning_rate="optimal", class_weight=None)

            # Train the first model.
            descent = train_model.mini_batch_e_net(
                firstClassifier, firstTrainingMatrix, firstTrainingClasses, classesUsed, batchSize=batchSize,
                numIterations=numIterations)

            # Record the first model's performance and descent.
            firstPredictions = firstClassifier.predict(firstTrainingMatrix)
            firstPosteriors = firstClassifier.predict_proba(firstTrainingMatrix)
            gMean = calc_metrics.calc_g_mean(firstPredictions, firstTrainingClasses)

            # Record information about the model performance.
            fidPerformance.write("FirstModel\t{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}\t{4:1.4f}\t{5:s}\n"
                                 .format(numIterations, batchSize, lambdaVal, elasticNetRatio, gMean,
                                         ','.join(["{0:1.4f}".format(i) for i in descent])))

            # Convert the target array into a matrix with the same number of columns as there are classes.
            # Each column will correspond to a single class. Given array of target classes, trainingClasses, the
            # target matrix, trainingClassesMatrix, will be organised so that
            # trainingClassesMatrix[i, j] == 1 if trainingClasses[i] is equal to j. Otherwise
            # trainingClassesMatrix[i, j] == 0. For example:
            #   trainingClasses == [1 2 3 2 1]
            #                            [1 0 0]
            #                            [0 1 0]
            #   trainingClassesMatrix == [0 0 1]
            #                            [0 1 0]
            #                            [1 0 0]
            trainingClassesMatrix = np.zeros((firstTrainingMatrix.shape[0], len(classesUsed)))
            for i in range(len(firstTrainingClasses)):
                trainingClassesMatrix[i, firstTrainingClasses[i]] = 1

            # Remove examples with posterior probability below the discard threshold.
            # The matrix of posteriors contains one column per class, and each example has a posterior for each
            # class. Examples are considered to be good examples if the posterior of their actual class
            # is greater than the discard threshold provided.
            largeEnoughPost = firstPosteriors > discardThreshold  # Posteriors larger than the discard threshold.
            goodExamples = np.any(trainingClassesMatrix * largeEnoughPost, axis=1)

            # Determine the training set for the second model.
            secondTrainingMatrix = firstTrainingMatrix[goodExamples, :]
            secondTrainingClasses = firstTrainingClasses[goodExamples]

            if len(np.unique(secondTrainingClasses)) < len(classesUsed):
                # The prediction errors of the first model have caused all examples of (at least) one class to be
                # removed from the dataset. Skip training the second model.
                print('WARNING: All examples of one class have been removed for having poor predictions.')
                fidPerformance.write("SecondModel\t{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}\t-\t-\n"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))
            else:
                # Create the second model.
                secondClassifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                                 l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                                 learning_rate="optimal", class_weight=None)

                # Train the second model.
                descent = train_model.mini_batch_e_net(
                    secondClassifier, secondTrainingMatrix, secondTrainingClasses, classesUsed,
                    batchSize=batchSize, numIterations=numIterations)

                # Record the second model's performance and descent.
                secondPredictions = secondClassifier.predict(secondTrainingMatrix)
                secondPosteriors = secondClassifier.predict_proba(secondTrainingMatrix)
                gMean = calc_metrics.calc_g_mean(secondPredictions, secondTrainingClasses)

                # Record information about the model performance.
                fidPerformance.write("SecondModel\t{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}\t{4:1.4f}\t{5:s}\n"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio, gMean,
                                             ','.join(["{0:1.4f}".format(i) for i in descent])))

                # Record the posteriors and predictions of the models.
                with open(dirResults + "/BothModelPredictions.tsv", 'w') as fidPredictions:
                    # Write the headers.
                    fidPredictions.write("PatientID\tClass\tFirstModelClass\tSecondModelClass\t{0:s}\t{1:s}\n".format(
                        '\t'.join(["FirstModel_{0:s}".format(mapIntRepToClass[i]) for i in classesUsed]),
                        '\t'.join(["SecondModel_{0:s}".format(mapIntRepToClass[i]) for i in classesUsed])))

                    # Write out the results
                    indicesOfPatientsUsed = np.array(range(dataMatrix.shape[0]))
                    firstClassifierPatientIndices = indicesOfPatientsUsed[patientIndicesToUse]
                    secondClassifierPatientIndices = firstClassifierPatientIndices[goodExamples]
                    for ind, i in enumerate(firstClassifierPatientIndices):
                        patientID = mapIndToPatient[i]
                        realClass = mapIntRepToClass[allExampleClasses[i]]
                        firstModelClass = mapIntRepToClass[firstPredictions[ind]]
                        firstModelPosteriors = firstPosteriors[ind, :]

                        # As the second model is likely to have only a subset of the examples used to train the first
                        # model, finding the predictions for the second model is more involved.
                        # First we need to determine whether the current example i was a 'good' example and so
                        # was used to train the second model.
                        if goodExamples[ind]:
                            # If the example was a 'good' example and was used for training the second model, then
                            # there will be a prediction for it.
                            secondModelExampleIndex = np.where(secondClassifierPatientIndices == i)[0]
                            secondModelExampleIndex = secondModelExampleIndex.astype("int")[0]  # Extract the index.
                            secondModelClass = mapIntRepToClass[secondPredictions[secondModelExampleIndex]]
                            secondModelPosteriors = secondPosteriors[secondModelExampleIndex, :]
                            fidPredictions.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\t{4:s}\t{5:s}\n".format(
                                patientID, realClass, firstModelClass, secondModelClass,
                                '\t'.join(["{0:1.4f}".format(firstModelPosteriors[j]) for j in classesUsed]),
                                '\t'.join(["{0:1.4f}".format(secondModelPosteriors[j]) for j in classesUsed])))
                        else:
                            # The example was not a 'good' example, and therefore has no predicted class as it wasn't
                            # used to train the second model.
                            fidPredictions.write("{0:s}\t{1:s}\t{2:s}\t-\t{3:s}\t{4:s}\n".format(
                                patientID, realClass, firstModelClass,
                                '\t'.join(["{0:1.4f}".format(firstModelPosteriors[j]) for j in classesUsed]),
                                '\t'.join(['-' for _ in classesUsed])))

                # Record the coefficients of the models.
                indicesOfCodesUsed = np.array(range(dataMatrix.shape[1]))  # The indices of the codes used for training.
                indicesOfCodesUsed = indicesOfCodesUsed[codeIndicesToUse]
                results_recording.record_coefficients(
                    firstClassifier, secondClassifier, mapIntRepToClass, indicesOfCodesUsed, mapCodeToInd,
                    mapCodeToDescr, dirResults + "/Coefficients.tsv")

                # Record the predictions on the ambiguous examples.
                ambiguousExamples = classExamples["Ambiguous"]  # Indices of the ambiguous examples.
                ambiguousMatrix = dataMatrix[ambiguousExamples, :]
                ambiguousMatrix = ambiguousMatrix[:, codeIndicesToUse]
                results_recording.record_ambiguous(
                    firstClassifier, secondClassifier, ambiguousExamples, ambiguousMatrix, mapIntRepToClass,
                    mapIndToPatient, dirResults + "/Ambiguous.tsv")
    elif cvFolds[0] == 0:
        # Perform the parameter optimisation.

        # Cut the data matrix down to a matrix containing only the codes to be used.
        dataMatrixSubset = dataMatrix[:, codeIndicesToUse]

        # Generate the stratified cross validation folds.
        foldsToGenerate = cvFolds[1] if len(cvFolds) > 1 else 10
        stratifiedFolds = np.array(partition_dataset.main(
            allExampleClasses, numPartitions=foldsToGenerate, isStratified=True))

        # Setup the record of the performance of each parameter combination.
        paramComboPerformance = []

        with open(dirResults + "/ParameterOptimisation.tsv", 'w') as fidParamOpt:
            # Write the header for the output file.
            fidParamOpt.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTotalGMean\t{0:s}\n".format(
                '\t'.join(["Fold_{0:d}_GMean".format(i) for i in range(foldsToGenerate)])))

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations, batchSize, lambdaVal, elasticNetRatio = params

                # Display a status update and record current round.
                print("\tIter={0:d}  Batch={1:d}  Lam={2:1.5f}  ENet={3:1.5f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidParamOpt.write("{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

                performanceOfEachFold = []  # Create a list to record the performance for each fold.
                predictions = np.empty(dataMatrix.shape[0])  # Holds predictions for all examples.
                predictions.fill(np.nan)

                # Perform the cross validation for this parameter combination.
                for fold in range(foldsToGenerate):
                    # Create the model.
                    classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                               l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                               learning_rate="optimal", class_weight=None)

                    # Determine training and testing example masks for this fold. Training examples are those
                    # examples not in this fold. Testing examples are examples in the fold.
                    trainingExamples = (stratifiedFolds != fold) & patientIndicesToUse
                    testingExamples = (stratifiedFolds == fold) & patientIndicesToUse

                    # Create training and testing matrices and target class arrays for this fold.
                    trainingMatrix = dataMatrixSubset[trainingExamples, :]
                    trainingClasses = allExampleClasses[trainingExamples]
                    testingMatrix = dataMatrixSubset[testingExamples, :]
                    testingClasses = allExampleClasses[testingExamples]

                    # Train the model on this fold.
                    _ = train_model.mini_batch_e_net(
                        classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix, testingClasses,
                        batchSize, numIterations)

                    # Record the model's performance on this fold.
                    testPredictions = classifier.predict(testingMatrix)
                    predictions[testingExamples] = testPredictions
                    performanceOfEachFold.append(calc_metrics.calc_g_mean(testPredictions,
                                                                                         testingClasses))

                # Record G mean of predictions across all folds using this parameter combination.
                # If the predicted class of any example is NaN, then the example did not actually have its class
                # predicted.
                examplesUsed = ~np.isnan(predictions)
                gMean = calc_metrics.calc_g_mean(predictions[examplesUsed],
                                                                allExampleClasses[examplesUsed])
                fidParamOpt.write("\t{0:1.4f}".format(gMean))
                paramComboPerformance.append(gMean)

                # Write out the performance of the model for each fold.
                fidParamOpt.write("\t{0:s}\n".format('\t'.join(["{0:1.4F}".format(i) for i in performanceOfEachFold])))

        # Determine best parameter combination. If there are two combinations
        # of parameters that give the best performance, then take the one that comes first.
        indexOfBestPerformance = paramComboPerformance.index(max(paramComboPerformance))
        bestInternalParams = paramCombos[indexOfBestPerformance]
        fidBestParams = open(dirResults + "/BestParameters.tsv", 'w')
        fidBestParams.write("Iterations\t{0:d}\nBatch Size\t{1:d}\nLambda Value\t{2:1.6f}\nENet Ratio\t{3:1.6f}\n"
                            .format(*bestInternalParams))
        fidBestParams.close()
    elif cvFolds[0] == 1:
        # Perform a non-nested hold out testing of the performance.
        # Train on a random half of the data and test on the remainder.

        # Cut the data matrix down to a matrix containing only the codes to be used.
        dataMatrixSubset = dataMatrix[:, codeIndicesToUse]

        # Generate the stratified cross validation folds.
        foldsToGenerate = 2
        stratifiedFolds = np.array(partition_dataset.main(
            allExampleClasses, numPartitions=foldsToGenerate, isStratified=True))

        with open(dirResults + "/HoldOutPerformance.tsv", 'w') as fidPerformance:
            # Write the header for the output file. Record the descent and the test performance.
            fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations, batchSize, lambdaVal, elasticNetRatio = params

                # Display a status update and record current round.
                print("Now - Iters={0:d}  Batch={1:d}  Lambda={2:1.5f}  ENet={3:1.2f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidPerformance.write("{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

                # Create the model.
                classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                           l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                           learning_rate="optimal", class_weight=None)

                # Determine training and testing example masks. Training examples are the examples in the first
                # fold, while testing examples are those in the second.
                trainingExamples = (stratifiedFolds == 0)
                testingExamples = (stratifiedFolds == 1)

                # Create training and testing matrices and target class arrays for this fold.
                trainingMatrix = dataMatrixSubset[trainingExamples, :]
                trainingClasses = allExampleClasses[trainingExamples]
                testingMatrix = dataMatrixSubset[testingExamples, :]
                testingClasses = allExampleClasses[testingExamples]

                # Train the model on this fold.
                descent = train_model.mini_batch_e_net(
                    classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix, testingClasses,
                    batchSize, numIterations)

                # Record the model's performance.
                testPredictions = classifier.predict(testingMatrix)
                testPerformance = calc_metrics.calc_g_mean(testPredictions, testingClasses)

                # Write out the performance of the model over all folds and the descent.
                fidPerformance.write("\t{0:1.4f}\t{1:s}\n"
                                     .format(testPerformance, ','.join(["{0:1.4f}".format(i) for i in descent])))
    else:
        # Training if nested cross validation is to be performed.

        # Cut the data matrix down to a matrix containing only the codes to be used.
        dataMatrixSubset = dataMatrix[:, codeIndicesToUse]

        # Determine the number of external and internal folds to use.
        numberExternalFolds= cvFolds[0]
        numberInternalFolds = cvFolds[0] if (len(cvFolds) < 2) else cvFolds[1]

        # Generate the external CV folds using stratified partitioning of the data.
        externalFolds = partition_dataset.main(
            allExampleClasses, numPartitions=numberExternalFolds, isStratified=True)

        # Create the arrays to hold the predictions and posteriors for all examples when using the external
        # CV folds for testing.
        externalPredictions = np.empty(dataMatrix.shape[0])
        externalPredictions.fill(np.nan)
        externalPosteriors = np.empty((dataMatrix.shape[0], len(classesUsed)))
        externalPosteriors.fill(np.nan)

        # Create the file to record the external CV results.
        fidExternalPerformance = open(dirResults + "/ExternalFold_FoldResults.tsv", 'w')
        fidExternalPerformance.write("ExternalFold\tNumIterations\tBatchSize\tLambda\tENetRatio\tGMean\n")

        # Perform external CV.
        for eCV in range(numberExternalFolds):
            # Display a status update.
            print("Starting external CV fold {0:d} at {1:s}."
                  .format(eCV, datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))

            # Generate the integral folds for this external fold. The examples in the fold are those that do not
            # have a class of np.nan (as they aren't being used at all) and have their partition equal to the
            # external fold being used currently.
            examplesInExternalFold = (~np.isnan(externalFolds)) & (externalFolds == eCV)
            examplesNotInExternalFold = (~np.isnan(externalFolds)) & (externalFolds != eCV)
            nonExternalFoldExampleIndices = np.nonzero(examplesNotInExternalFold)[0]
            internalFolds = partition_dataset.main(
                allExampleClasses, indicesToUse=nonExternalFoldExampleIndices, numPartitions=numberInternalFolds,
                isStratified=True)

            # Setup the record of the performance of each parameter combination.
            paramComboPerformance = []

            # Perform a grid search over all parameter combinations.
            with open(dirResults + "/ExternalFold_{0:d}.tsv".format(eCV), 'w') as fidPerformance:
                # Write the header.
                fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTotalGMean\t{0:s}\n".format(
                    '\t'.join(["Fold_{0:d}_GMean".format(i) for i in range(numberInternalFolds)])))

                for params in paramCombos:
                    # Define the parameters for this run.
                    numIterations, batchSize, lambdaVal, elasticNetRatio = params

                    # Display a status update and record current round.
                    print("\tIter={0:d}  Batch={1:d}  Lam={2:1.5f}  ENet={3:1.5f}  Time={4:s}"
                          .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                                  datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                    fidPerformance.write("{0:d}\t{1:d}\t{2:1.5f}\t{3:1.2f}"
                                         .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

                    performanceOfEachFold = []  # Create a list to record the performance for each internal fold.
                    predictions = np.empty(dataMatrix.shape[0])  # Holds predictions for all examples in external fold.
                    predictions.fill(np.nan)

                    # Perform the internal cross validation.
                    for iCV in range(numberInternalFolds):

                        # Create the model.
                        classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                                   l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                                   learning_rate="optimal", class_weight=None)

                        # Determine training and testing example masks for this fold. Training examples are those
                        # examples in this external CV fold that are not in the internal CV fold. Testing examples
                        # are examples in the external fold that are in this internal fold.
                        trainingExamples = (internalFolds != iCV) & examplesNotInExternalFold
                        testingExamples = (internalFolds == iCV) & examplesNotInExternalFold

                        # Create training and testing matrices and target class arrays for this fold.
                        trainingMatrix = dataMatrixSubset[trainingExamples, :]
                        trainingClasses = allExampleClasses[trainingExamples]
                        testingMatrix = dataMatrixSubset[testingExamples, :]
                        testingClasses = allExampleClasses[testingExamples]

                        # Train the model on this fold.
                        _ = train_model.mini_batch_e_net(
                            classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix, testingClasses,
                            batchSize, numIterations)

                        # Record the model's performance on this fold.
                        testPredictions = classifier.predict(testingMatrix)
                        predictions[testingExamples] = testPredictions
                        performanceOfEachFold.append(calc_metrics.calc_g_mean(testPredictions,
                                                                                             testingClasses))

                    # Record G mean of predictions across all folds using this parameter combination.
                    # If the predicted class of any example is NaN, then the example did not actually have its class
                    # predicted.
                    examplesUsed = ~np.isnan(predictions)
                    gMean = calc_metrics.calc_g_mean(predictions[examplesUsed],
                                                                    allExampleClasses[examplesUsed])
                    fidPerformance.write("\t{0:1.4f}".format(gMean))
                    paramComboPerformance.append(gMean)

                    # Write out the performance of the model for each fold.
                    fidPerformance.write("\t{0:s}\n"
                                         .format('\t'.join(["{0:1.4F}".format(i) for i in performanceOfEachFold])))

            # Determine best parameter combo from the internal folds. If there are two combinations
            # of parameters that give the best performance, then take the one that comes first.
            indexOfBestPerformance = paramComboPerformance.index(max(paramComboPerformance))
            bestInternalParams = paramCombos[indexOfBestPerformance]

            # Determine training and testing example masks for this external fold. Training examples are all examples
            # not in this external fold, testing examples are the ones in this external fold.
            trainingExamples = examplesNotInExternalFold
            testingExamples = examplesInExternalFold

            # Create training and testing matrices and target class arrays for this fold.
            trainingMatrix = dataMatrixSubset[trainingExamples, :]
            trainingClasses = allExampleClasses[trainingExamples]
            testingMatrix = dataMatrixSubset[testingExamples, :]
            testingClasses = allExampleClasses[testingExamples]

            # Train a model on this external CV fold using the optimal parameters from the internal folds.
            optimalIterations, optimalBatchSize, optimalLambda, optimalENetRatio = bestInternalParams
            classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=optimalLambda,
                                       l1_ratio=optimalENetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                       learning_rate="optimal", class_weight=None)
            _ = train_model.mini_batch_e_net(
                classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix, testingClasses,
                optimalBatchSize, optimalIterations)

            # Record the model's performance on this external fold.
            testPredictions = classifier.predict(testingMatrix)
            externalPredictions[testingExamples] = testPredictions
            testPosteriors = classifier.predict_proba(testingMatrix)
            externalPosteriors[testingExamples, :] = testPosteriors
            gMean = calc_metrics.calc_g_mean(testPredictions, testingClasses)
            fidExternalPerformance.write("{0:d}\t{1:d}\t{2:d}\t{3:1.5f}\t{4:1.2f}\t{5:1.4f}\n".format(
                eCV, optimalIterations, optimalBatchSize, optimalLambda, optimalENetRatio, gMean))

        # Close external CV results file.
        fidExternalPerformance.close()

        # Record G mean, AUC and ROC for the external folds.
        with open(dirResults + "/ExternalFold_OverallPerformance.tsv", 'w') as fidOverallPerformance:
            # Calculate G mean of the entire nested procedure.
            finalGMean = calc_metrics.calc_g_mean(externalPredictions[~np.isnan(externalPredictions)],
                                                                 allExampleClasses[~np.isnan(allExampleClasses)])
            fidOverallPerformance.write("G mean over all external folds - {0:1.4f}\n".format(finalGMean))

            # Calculate ROC curve for the final external CV predictions.
            for i in classesUsed:
                # Create arrays containing the real class of each example and the posterior probability of the
                # positive class (class i).
                trueClasses = allExampleClasses[patientIndicesToUse]
                posPosteriors = externalPosteriors[patientIndicesToUse, i]

                # Calculate ROC curve values.
                falsePosRates, truePosRates, thresholds = metrics.roc_curve(trueClasses, posPosteriors, pos_label=i)

                # Calculate the AUC.
                binaryIndicator = (trueClasses == i) * 1
                auc = metrics.roc_auc_score(binaryIndicator, posPosteriors)

                # Write out the ROC results for the class.
                fidOverallPerformance.write("{0:s} AUC\t{1:1.4f}\n{0:s} FPR\t{2:s}\n{0:s} TPR\t{3:s}\n"
                                            "{0:s} thresholds\t{4:s}\n"
                    .format(mapIntRepToClass[i], auc,
                            ','.join(["{0:1.4f}".format(j) for j in falsePosRates]),
                            ','.join(["{0:1.4f}".format(j) for j in truePosRates]),
                            ','.join(["{0:1.4f}".format(j) for j in thresholds])))