"""Perform the code mining."""

# Python imports.
import datetime
import os
import sys

# 3rd party imports.
import numpy as np
from sklearn.linear_model import SGDClassifier

# User imports.
import calc_metrics
import generate_code_mapping
import generate_dataset
import parse_classes
import partition_dataset
import train_model


def main(fileDataset, fileCodeMapping, dirResults, classData, lambdaVals=(0.01,), elasticNetMixing=(0.15,),
         batchSizes=(500,), numIters=(5,), codeOccurrences=0, patientOccurrences=0, cvFolds=0,
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
    :param cvFolds:             The number of cross validation folds to use.
    :type cvFolds:              int
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
    # The code to index mapping maps codes to their indices in the data matrix.
    # The patient to index mapping maps patients to their indices in the data matrix.
    dataMatrix, mapPatientToInd, mapCodeToInd = generate_dataset.main(fileDataset, dirResults, mapCodeToDescr,
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
    mapClassToIntRep = {}  # Mapping from the integer code used to reference each class to the class it references.
    currentCode = 0
    for i in classExamples:
        if i != "Ambiguous":
            mapClassToIntRep[currentCode] = i
            allExampleClasses[classExamples[i]] = currentCode
            currentCode += 1
    classesUsed = [i for i in mapClassToIntRep]  # List containing the integer code for every class in the dataset.

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

    if cvFolds == 0:
        # Training if no testing is needed.

        with open(dirResults + "/Performance_First.tsv", 'w') as fidPerformanceFirst, \
                open(dirResults + "/Performance_Second.tsv", 'w') as fidPerformanceSecond:

            # Write the header for the output files.
            fidPerformanceFirst.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")
            fidPerformanceSecond.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations = params[0]
                batchSize = params[1]
                lambdaVal = params[2]
                elasticNetRatio = params[3]

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
                firstPosteriors = firstClassifier.predict_proba(firstTrainingMatrix)
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
                    secondPredictions = secondClassifier.predict(secondTrainingMatrix)
                    gMean = calc_metrics.calc_g_mean(secondPredictions, secondTrainingClasses)
                    fidPerformanceSecond.write("\t{0:1.4f}".format(gMean))
                    fidPerformanceSecond.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent])))
    else:
        # Training if cross validation is to be performed.

        # Never generate fewer than 2 folds. If one was requested, then generate 2 and use one for training and
        # the other for testing.
        foldsToGenerate = cvFolds if cvFolds > 1 else 2

        # Generate the stratified cross validation folds.
        stratifiedFolds = np.array(partition_dataset.main(allExampleClasses, foldsToGenerate, True))

        with open(dirResults + "/CVPerformance.tsv", 'w') as fidPerformance:
            # Write the header for the output file.
            if cvFolds == 1:
                # If there is only one fold, then record the descent and the test performance.
                fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\tTestGMean\tDescentGMean\n")
            else:
                # If there is more than one fold, then only record the performance on each test fold.
                fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\t{0:s}\tTotalGMean\n"
                                     .format('\t'.join(["Fold_{0:d}_GMean".format(i) for i in range(cvFolds)])))

            for params in paramCombos:
                # Define the parameters for this run.
                numIterations = params[0]
                batchSize = params[1]
                lambdaVal = params[2]
                elasticNetRatio = params[3]

                # Display a status update and record current round.
                print("Now - Iters={0:d}  Batch={1:d}  Lambda={2:1.4f}  ENet={3:1.4f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidPerformance.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

                # Cut the data matrix down to a matrix containing only the codes to be used.
                dataMatrixSubset = dataMatrix[:, codeIndicesToUse]

                # Create the model.
                classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                           l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                           learning_rate="optimal", class_weight=None)

                # Train and test on each fold if cvFolds > 1. Otherwise, train on one fold and test on the other.
                descent, performanceOfEachFold, predictions = train_model.mini_batch_e_net_cv(
                    classifier, dataMatrixSubset, allExampleClasses, patientIndicesToUse, stratifiedFolds, classesUsed,
                    batchSize, numIterations, cvFolds)

                # Write out the performance of the model over all folds.
                fidPerformance.write("\t{0:s}".format('\t'.join(["{0:1.4F}".format(i) for i in performanceOfEachFold])))

                if cvFolds == 1:
                    # If only one fold, then record the descent.
                    fidPerformance.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent[0]])))
                else:
                    # Record G mean of predictions across all folds using this parameter combination.
                    # If the predicted class of any example is NaN, then the example did not actually have its class
                    # predicted.
                    examplesUsed = ~np.isnan(predictions)
                    gMean = calc_metrics.calc_g_mean(predictions[examplesUsed], allExampleClasses[examplesUsed])
                    fidPerformance.write("\t{0:1.4f}\n".format(gMean))