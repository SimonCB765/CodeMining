"""Perform the code mining."""

# Python imports.
import datetime
import os
import sys

# 3rd party imports.
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# User imports.
import calc_metrics
import generate_code_mapping
import generate_dataset
import parse_classes
import partition_dataset
import train_model


def main(fileDataset, fileCodeMapping, dirResults, classData, lambdaVals=(0.01,), elasticNetMixing=(0.15,),
         batchSizes=(500,), numIters=(5,), codeOccurrences=0, patientOccurrences=0, cvFolds=(0,),
         dataNormVal=0, discardThreshold=0.0):
    """Perform the clinical code mining.

    :param fileDataset:         File location containing the data to use.
    :type fileDataset:          str
    :param fileCodeMapping:     File containing the mapping between codes and their descriptions.
    :type fileCodeMapping:      str
    :param dirResults:          Location of the directory in which the results will be saved.
    :type dirResults:           str
    :param classData:           Class definitions.
    :type classData:            JSON style dict
    :param lambdaVals:          The regularisation parameter values to use.
    :type lambdaVals:           list-like
    :param elasticNetMixing:    The elastic net mixing values to use.
    :type elasticNetMixing:     list-like
    :param batchSizes:          The batch sizes to use.
    :type batchSizes:           list-like
    :param numIters:            The values to use for the number of iterations to run the gradient descent for.
    :type numIters:             list-like
    :param codeOccurrences:     The minimum number of different patients a code must co-occur with to be used.
    :type codeOccurrences:      int
    :param patientOccurrences:  The minimum number of different codes a patient must co-occur with to be used.
    :type patientOccurrences:   int
    :param cvFolds:             The number of cross validation folds to use. Can be interpreted in one of four ways:
                                    If cvFolds == [], then the sequential model training is performed.
                                    If cvFolds[0] == 0, then the best parameter combination is determined using
                                        K fold cross validation, where K = cvFolds[1] (default to 10).
                                    If cvFolds[0] == 1, then holdout testing of the parameter combinations is performed.
                                    If cvFolds[0] > 1, the nested cross validation is performed. cvFolds[0] is the
                                        number of external folds to use, and cvFolds[1] the number of internal folds.
    :type cvFolds:              list-like
    :param dataNormVal:         The data normalisation code to use.
    :type dataNormVal:          int
    :param discardThreshold:    The posterior probability that an example must have to be included in the final model.
    :type discardThreshold:     float

    """

    # Create the results directory if necessary.
    if not os.path.exists(dirResults):
        try:
            os.makedirs(dirResults)
        except Exception as err:
            print("Error creating results directory: {0}".format(err))
            sys.exit()

    # Generate the mapping from codes to their descriptions.
    mapCodeToDescr = generate_code_mapping.main(fileCodeMapping)

    # Generate the data matrix and two index mappings.
    # The code to index mapping maps codes to their column indices in the data matrix.
    # The index to patient mapping maps row indices for the data matrix to the patient they correspond to.
    dataMatrix, mapIndToPatient, mapCodeToInd = generate_dataset.main(fileDataset, dirResults, mapCodeToDescr,
                                                                      dataNormVal)

    # Determine the examples in each class.
    # classExamples is a dictionary with an entry for each class (and one for "Ambiguous" examples) that contains
    # a list of the examples belonging to that class.
    # classCodeIndices is a list of the indices of the codes used to determine class membership.
    classExamples, classCodeIndices = parse_classes.find_patients(dataMatrix, classData,
                                                                  mapCodeToInd, isCodesRemoved=True)

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
    # TODO: Remove the codes and patient that don't occur frequently enough.
    # TODO: Repeatedly remove patients and codes from data matrix until not patients
    # TODO: or code have too few connections.
    # TODO: basically repeatedly remove codes with < codeOccurrence patients they occur in
    # TODO: and patients wih < patientOccurrences codes they occur in

    # Check whether there are any classes (besides the ambiguous class) with no examples.
    noExampleClasses = [i for i in classExamples if (len(classExamples[i]) == 0 and i != "Ambiguous")]
    haveExamples = [i for i in classExamples if (len(classExamples[i]) > 0 and i != "Ambiguous")]
    if len(haveExamples) == 1:
        print("Only class {0:s} has any examples.".format(haveExamples[0]))
        sys.exit()
    elif noExampleClasses:
        print("The following classes have no unambiguous examples and will be unused: {0:s}"
            .format(','.join(noExampleClasses)))

    # Create all combinations of parameters that will be used.
    paramCombos = [[i, j, k, l] for i in numIters for j in batchSizes for k in lambdaVals for l in elasticNetMixing]

    if cvFolds[0] == 0:
        # Training if no testing is needed.

        with open(dirResults + "/Performance_First.tsv", 'w') as fidPerformanceFirst, \
                open(dirResults + "/Performance_Second.tsv", 'w') as fidPerformanceSecond:

            # Write the header for the output files.
            fidPerformanceFirst.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")
            fidPerformanceSecond.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations, batchSize, lambdaVal, elasticNetRatio = params

                # Display a status update and record current round.
                print("Now - Iters={0:d}  Batch={1:d}  Lambda={2:1.4f}  ENet={3:1.4f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidPerformanceFirst.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))
                fidPerformanceSecond.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

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
                fidPerformanceFirst.write("\t{0:1.4f}".format(gMean))
                fidPerformanceFirst.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent])))

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
                    fidPerformanceSecond.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}\t-\t-\n"
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
                    secondPredictions = secondClassifier.predict(firstTrainingMatrix)
                    secondPosteriors = secondClassifier.predict_proba(firstTrainingMatrix)
                    gMean = calc_metrics.calc_g_mean(secondPredictions, firstTrainingClasses)
                    fidPerformanceSecond.write("\t{0:1.4f}".format(gMean))
                    fidPerformanceSecond.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent])))

                    # Record the posteriors and predictions of the models.
                    with open(dirResults + "/Predictions.tsv", 'w') as fidPredictions, \
                            open(dirResults + "/Posteriors.tsv", 'w') as fidPosteriors:
                        # Write the headers.
                        fidPredictions.write("PatientID\tClass\tFirstModelClass\tSecondModelClass\n")
                        fidPosteriors.write("PatientID\t{0:s}\t{1:s}\n".format(
                            '\t'.join(["FirstModel_{0:s}".format(mapIntRepToClass[i]) for i in classesUsed]),
                            '\t'.join(["SecondModel_{0:s}".format(mapIntRepToClass[i]) for i in classesUsed])))

                        # Write out the results
                        indicesOfUsedPatients = np.array(range(dataMatrix.shape[0]))
                        indicesOfUsedPatients = indicesOfUsedPatients[patientIndicesToUse]
                        for ind, i in enumerate(indicesOfUsedPatients):
                            patientID = mapIndToPatient[i]
                            realClass = mapIntRepToClass[allExampleClasses[i]]
                            firstModelClass = mapIntRepToClass[firstPredictions[ind]]
                            firstModelPosteriors = firstPosteriors[ind, :]
                            secondModelClass = mapIntRepToClass[secondPredictions[ind]]
                            secondModelPosteriors = secondPosteriors[ind, :]

                            fidPredictions.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\n".format(
                                patientID, realClass, firstModelClass, secondModelClass))
                            fidPosteriors.write("{0:s}\t{1:s}\t{2:s}\n".format(patientID,
                                '\t'.join(["{0:1.4f}".format(firstModelPosteriors[i]) for i in classesUsed]),
                                '\t'.join(["{0:1.4f}".format(secondModelPosteriors[i]) for i in classesUsed])))

                    # Record the coefficients of the models.
                    with open(dirResults + "/Coefficients.tsv", 'w') as fidCoefs:
                        # Write out the header.
                        if len(classesUsed) == 2:
                            # If there are two classes, then there is only one coefficient per code.
                            fidCoefs.write("Code\tDescription\tFirstModel\tSecondModel\n")
                        else:
                            # If there are more than two classes, then there is one coefficient per class per code.
                            fidCoefs.write("Code\tDescription\t{0:s}\t{1:s}\n".format(
                                '\t'.join(["FirstModel_{0:s}_Coef".format(mapIntRepToClass[i]) for i in classesUsed]),
                                '\t'.join(["SecondModel_{0:s}_Coef".format(mapIntRepToClass[i]) for i in classesUsed])))

                        # Reverse the code to index mapping.
                        mapIndToCode = {v : k for k, v in mapCodeToInd.items()}

                        # Write out the coefficients.
                        indicesOfUsedCodes = np.array(range(dataMatrix.shape[1]))
                        indicesOfUsedCodes = indicesOfUsedCodes[codeIndicesToUse]
                        for ind, i in enumerate(indicesOfUsedCodes):
                            code = mapIndToCode[i]
                            firstModelCoefs = firstClassifier.coef_[:, ind]
                            secondModelCoefs = secondClassifier.coef_[:, ind]
                            if len(classesUsed) == 2:
                                fidCoefs.write("{0:s}\t{1:s}\t{2:1.4f}\t{3:1.4f}\n".format(
                                    code, mapCodeToDescr.get(code, "Unknown Code"), firstModelCoefs[0],
                                    secondModelCoefs[0]))
                            else:
                                fidCoefs.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\n".format(
                                    code, mapCodeToDescr.get(code, "Unknown Code"),
                                    '\t'.join(["{0:1.4f}".format(firstModelCoefs[i]) for i in classesUsed]),
                                    '\t'.join(["{0:1.4f}".format(secondModelCoefs[i]) for i in classesUsed])))

                    # Record the predictions on the ambiguous examples.
                    ambiguousExamples = classExamples["Ambiguous"]
                    ambiguousMatrix = dataMatrix[ambiguousExamples, :]
                    ambiguousMatrix = ambiguousMatrix[:, codeIndicesToUse]
                    with open(dirResults + "/Ambiguous.tsv", 'w') as fidAmbig:
                        # Write out the header.
                        fidAmbig.write("PatientID\tFirstModelClass\t{0:s}\tSecondModelClass\t{1:s}\n".format(
                            '\t'.join(["FirstModel_{0:s}_Post".format(mapIntRepToClass[i]) for i in classesUsed]),
                            '\t'.join(["SecondModel_{0:s}_Post".format(mapIntRepToClass[i]) for i in classesUsed])))

                        # Generate the predictions.
                        firstModelAmbigPred = firstClassifier.predict(ambiguousMatrix)
                        firstModelAmigPosts = firstClassifier.predict_proba(ambiguousMatrix)
                        secondModelAmbigPred = secondClassifier.predict(ambiguousMatrix)
                        secondModelAmigPosts = secondClassifier.predict_proba(ambiguousMatrix)

                        # Write out the predictions.
                        for ind, i in enumerate(ambiguousExamples):
                            patientID = mapIndToPatient[i]
                            fidAmbig.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\t{4:s}\n".format(
                                patientID, mapIntRepToClass[firstModelAmbigPred[ind]],
                                '\t'.join(["{0:1.4f}".format(firstModelAmigPosts[ind, i]) for i in classesUsed]),
                                mapIntRepToClass[secondModelAmbigPred[ind]],
                                '\t'.join(["{0:1.4f}".format(secondModelAmigPosts[ind, i]) for i in classesUsed])))
    elif cvFolds[0] == 1:
        # Perform a non-nested hold out testing of the performance.
        # Train on a random half of the data and test on the remainder.

        # Cut the data matrix down to a matrix containing only the codes to be used.
        dataMatrixSubset = dataMatrix[:, codeIndicesToUse]

        # Generate the stratified cross validation folds.
        foldsToGenerate = 2
        stratifiedFolds = np.array(partition_dataset.main(allExampleClasses, numPartitions=foldsToGenerate,
                                                          isStratified=True))

        with open(dirResults + "/HoldOutPerformance.tsv", 'w') as fidPerformance:
            # Write the header for the output file. Record the descent and the test performance.
            fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations, batchSize, lambdaVal, elasticNetRatio = params

                # Display a status update and record current round.
                print("Now - Iters={0:d}  Batch={1:d}  Lambda={2:1.4f}  ENet={3:1.4f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidPerformance.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}"
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
                descent = train_model.mini_batch_e_net(classifier, trainingMatrix, trainingClasses, classesUsed,
                                                       testingMatrix, testingClasses, batchSize, numIterations)

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
        externalFolds = partition_dataset.main(allExampleClasses, numPartitions=numberExternalFolds, isStratified=True)

        # Create the arrays to hold the predictions and posteriors for all examples when using the external
        # CV folds for testing.
        externalPredictions = np.empty(dataMatrix.shape[0])
        externalPredictions.fill(np.nan)
        externalPosteriors = np.empty((dataMatrix.shape[0], len(classesUsed)))
        externalPosteriors.fill(np.nan)

        # Create the file to record the external CV results.
        fidExternalPerformance = open(dirResults + "/ExternalFoldResults.tsv", 'w')
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
            internalFolds = partition_dataset.main(allExampleClasses, indicesToUse=nonExternalFoldExampleIndices,
                                                   numPartitions=numberInternalFolds, isStratified=True)

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
                    fidPerformance.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}"
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
                        _ = train_model.mini_batch_e_net(classifier, trainingMatrix, trainingClasses, classesUsed,
                                                         testingMatrix, testingClasses, batchSize, numIterations)

                        # Record the model's performance on this fold.
                        testPredictions = classifier.predict(testingMatrix)
                        predictions[testingExamples] = testPredictions
                        performanceOfEachFold.append(calc_metrics.calc_g_mean(testPredictions, testingClasses))

                    # Record G mean of predictions across all folds using this parameter combination.
                    # If the predicted class of any example is NaN, then the example did not actually have its class
                    # predicted.
                    examplesUsed = ~np.isnan(predictions)
                    gMean = calc_metrics.calc_g_mean(predictions[examplesUsed], allExampleClasses[examplesUsed])
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
            _ = train_model.mini_batch_e_net(classifier, trainingMatrix, trainingClasses, classesUsed, testingMatrix,
                                             testingClasses, optimalBatchSize, optimalIterations)

            # Record the model's performance on this external fold.
            testPredictions = classifier.predict(testingMatrix)
            externalPredictions[testingExamples] = testPredictions
            testPosteriors = classifier.predict_proba(testingMatrix)
            externalPosteriors[testingExamples, :] = testPosteriors
            gMean = calc_metrics.calc_g_mean(testPredictions, testingClasses)
            fidExternalPerformance.write("{0:d}\t{1:d}\t{2:d}\t{3:1.4f}\t{4:1.4f}\t{5:1.4f}\n".format(
                eCV, optimalIterations, optimalBatchSize, optimalLambda, optimalENetRatio, gMean))

        # Close external CV results file.
        fidExternalPerformance.close()

        # Record G mean, AUC and ROC for the external folds.
        with open(dirResults + "/FinalPerformance.tsv", 'w') as fidFinalPerformance:
            # Calculate G mean of the entire nested procedure.
            finalGMean = calc_metrics.calc_g_mean(externalPredictions[~np.isnan(externalPredictions)],
                                                  allExampleClasses[~np.isnan(allExampleClasses)])
            fidFinalPerformance.write("G mean over all external folds - {0:1.4f}\n".format(finalGMean))

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
                fidFinalPerformance.write("{0:s} AUC\t{1:1.4f}\n{2:s} FPR\t{3:s}\n{4:s} TPR\t{5:s}\n"
                                          "{6:s} thresholds\t{7:s}\n"
                    .format(mapIntRepToClass[i], auc,
                            mapIntRepToClass[i], ','.join(["{0:1.4f}".format(i) for i in falsePosRates]),
                            mapIntRepToClass[i], ','.join(["{0:1.4f}".format(i) for i in truePosRates]),
                            mapIntRepToClass[i], ','.join(["{0:1.4f}".format(i) for i in thresholds])))