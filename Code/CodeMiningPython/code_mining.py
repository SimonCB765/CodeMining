"""Perform the code mining."""

# Python imports.
import datetime
import os
import sys

# 3rd party imports.
import numpy as np
from sklearn.linear_model import SGDClassifier

# User imports.
import generate_code_mapping
import generate_dataset
import parse_classes
import partition_dataset


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

    # Generate the data matrix, initia array of classes and two index mappings.
    # The code to index mapping maps codes to their indices in the data matrix.
    # The patient to index mapping maps patients to their indices in the data matrix.
    dataMatrix, mapPatientToInd, mapCodeToInd = generate_dataset.main(fileDataset, dirResults, mapCodeToDescr)
    allExampleClasses = np.zeros(dataMatrix.shape[0])

    # TODO: Remove the codes and patient that don't occur frequently enough.
    # TODO: Repeatedly remove patients and codes from data matrix until not patients
    # TODO: or code have too few connections.
    # TODO: basically repeatedly remove codes with < codeOccurrence patients they occur in
    # TODO: and patients wih < patientOccurrences codes they occur in

    # Determine the examples in each class along with the class of each example.
    # classExamples is a dictionary with an entry for each class (and one for "Ambiguous" examples) that contains
    # a list of the examples belonging to that class.
    dataMatrix, classExamples = parse_classes.find_patients(dataMatrix, classData, mapCodeToInd, isCodesRemoved=True)
    classCode = {}  # Mapping from the integer code used to reference each class to the class it references.
    currentCode = 1
    for i in classExamples:
        if i != "Ambiguous":
            classCode[currentCode] = i
            allExampleClasses[classExamples[i]] = currentCode
            currentCode += 1

    # Determine a mask for the examples that will be used for training. Any example with a class of 0 is not
    # used for training as the real classes begin at 1.
    trainingExampleMask = allExampleClasses != 0

    # Check whether there are any classes (besides the ambiguous class) with no examples.
    noExampleClasses = [i for i in classExamples if (len(classExamples[i]) == 0 and i != "Ambiguous")]
    haveExamples = [i for i in classExamples if (len(classExamples[i]) > 0 and i != "Ambiguous")]
    if len(haveExamples) == 1:
        print("Only class {0:s} has any examples.".format(haveExamples[0]))
        sys.exit()
    elif noExampleClasses:
        print("The following classes have no unambiguous examples and will be unused: {0:s}".format(
            ','.join(noExampleClasses)))

    # Create all combinations of parameters that will be used.
    paramCombos = [[i, j, k, l] for i in numIters for j in batchSizes for k in lambdaVals for l in elasticNetMixing]

    if cvFolds == 0:
        # Training if no testing is needed.
        # TODO : add he complicated stuff to train the initial and the second model
        pass
    else:
        # Training if cross validation is to be performed.

        # Never generate fewer than 2 folds. If one was requested, then generate 2 and use one for training and
        # the other for testing.
        partitionsToGenerate = cvFolds if cvFolds > 1 else 2

        with open(dirResults + "/CVPerformance.tsv", 'w') as fidPerformance:
            # Write the header for the output file.
            fidPerformance.write("NumIterations\tBatchSize\tLambda\tENetRatio\t{0:s}\tTotalError\n"
                                 .format('\t'.join(["Fold_{0:d}_Error".format(i) for i in range(cvFolds)])))

            for params in paramCombos[:1]:
                # Define the parameters for this run.
                numIterations = params[0]
                batchSize = params[1]
                lambdaVal = params[2]
                elasticNetRatio = params[3]

                # Display a status update and record current round.
                print("Now - Iters={0:d}  Batch={1:d}  Lambda={2:1.4f}  ENet={3:1.4f}  Time={4:s}"
                      .format(numIterations, batchSize, lambdaVal, elasticNetRatio,
                              datetime.datetime.strftime(datetime.datetime.now(), "%x %X")))
                fidPerformance.write("{0:d}\t{1:d}\t{2:1.4f}\t{3:1.4f}\t"
                                     .format(numIterations, batchSize, lambdaVal, elasticNetRatio))

                # Run the desired number of training iterations.
                for _ in range(numIterations):
                    # Generate the stratified cross validation folds.
                    partitions = np.array(partition_dataset.main(allExampleClasses, partitionsToGenerate, True))

                    # Generate the array to hold the predictions.
                    predictions = np.zeros(dataMatrix.shape[0])

                    # Train and test on each fold if cFolds > 1. Otherwise, train on one fold and test on the other.
                    for i in range(cvFolds):
                        # Determine training and testing example masks.
                        trainingExamples = (partitions != i) & trainingExampleMask
                        testingExamples = (partitions == i) & trainingExampleMask

                        # Create training and testing matrices and class arrays.
                        trainingMatrix = dataMatrix[trainingExamples, :]
                        trainingClasses = allExampleClasses[trainingExamples]
                        testingMatrix = dataMatrix[testingExamples, :]
                        testingClasses = allExampleClasses[testingExamples]

                        # Train the model.
                        classifier = SGDClassifier(loss="log", penalty="elasticnet", alpha=lambdaVal,
                                                   l1_ratio=elasticNetRatio, fit_intercept=True, n_iter=1, n_jobs=1,
                                                   learning_rate="optimal", class_weight=None)
                        classifier.fit(trainingMatrix, trainingClasses)

                        # Record the model's predictions on this fold.
                        testPredictions = classifier.predict(testingMatrix)
                        predictions[testingExamples] = testPredictions
                        predictionError = 1 - (sum(testPredictions == testingClasses) / testingClasses.shape[0])
                        fidPerformance.write("\t{0:1.4f}".format(predictionError))

                    # Record error of predictions.
                    examplesUsed = predictions != 0
                    totalPredictionError = predictions[examplesUsed] == allExampleClasses[examplesUsed]
                    totalPredictionError = 1 - (sum(totalPredictionError) / sum(examplesUsed))
                    fidPerformance.write("\t{0:1.4f}\n".format(totalPredictionError))