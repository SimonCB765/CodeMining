"""Perform the code mining."""

# Python imports.
import datetime
import os
import random
import sys

# 3rd party imports.
import numpy as np

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
    dataMatrix, mapPatientToInd, mapCodeToInd = generate_dataset.main(fileDataset, dirResults, mapCodeToDescr)

    # Determine the examples in each class.
    # classExamples is a dictionary with an entry for each class (and one for "Ambiguous" examples) that contains
    # a list of the examples belonging to that class.
    # classCodeIndices is a list of the indices of the codes used to determine class membership.
    classExamples, classCodeIndices = parse_classes.find_patients(dataMatrix, classData,
                                                                  mapCodeToInd, isCodesRemoved=True)

    # Determine the class of each example, and map the classes from their names to an integer representation.
    # A class integer of 0 is used to indicate that an example does not belong to any class, and is therefore
    # not used in the training.
    allExampleClasses = np.zeros(dataMatrix.shape[0])  # The integer class code for each example.
    mapClassToIntRep = {}  # Mapping from the integer code used to reference each class to the class it references.
    currentCode = 1
    for i in classExamples:
        if i != "Ambiguous":
            mapClassToIntRep[currentCode] = i
            allExampleClasses[classExamples[i]] = currentCode
            currentCode += 1
    classesUsed = [i for i in mapClassToIntRep]  # List of containing the integer code for every class in the dataset.

    # Calculate masks for the patients and the codes. These will be used to select only those patients and codes
    # that are to be used for training/testing.
    patientIndicesToUse = np.ones(dataMatrix.shape[0], dtype=bool)
    patientIndicesToUse[allExampleClasses == 0] = 0  # Mask out the patients that have no class.
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
        print("The following classes have no unambiguous examples and will be unused: {0:s}".format(
            ','.join(noExampleClasses)))

    # Create all combinations of parameters that will be used.
    paramCombos = [[i, j, k, l] for i in numIters for j in batchSizes for k in lambdaVals for l in elasticNetMixing]

    if cvFolds == 0:
        # Training if no testing is needed.

        # Generate the permutations of the data to use. The permutations will define the shuffling of the training
        # data that we will use. Calculating these upfront will cause each combination of parameters
        # to use the same permutations.
        numTrainingExamples = sum(patientIndicesToUse)  # The number of training examples we're using.
        maxNumIterations = max(numIters)  # Maximum number of iterations used.
        permutations = [list(range(numTrainingExamples)) for _ in range(maxNumIterations)]
        for i in permutations:
            random.shuffle(i)

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

                # Create the training matrix and class array.
                # Generate the training matrix in two steps as scipy sparse matrices can not be sliced
                # in the same operation with different length arrays.
                firstTrainingMatrix = dataMatrix[patientIndicesToUse, :]
                firstTrainingMatrix = firstTrainingMatrix[:, codeIndicesToUse]
                firstTrainingClasses = allExampleClasses[patientIndicesToUse]

                # Train the first model.
                firstClassifier, descent = train_model.mini_batch_e_net(
                    firstTrainingMatrix, firstTrainingClasses, classesUsed, permutations, lambdaVal, elasticNetRatio,
                    batchSize, numIterations)

                # Record the first model's performance and descent.
                trainingPredictions = firstClassifier.predict(firstTrainingMatrix)
                gMean = calc_metrics.calc_g_mean(trainingPredictions, firstTrainingClasses)
                fidPerformanceFirst.write("\t{0:1.4f}".format(gMean))
                fidPerformanceFirst.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent])))
    else:
        # Training if cross validation is to be performed.

        # Never generate fewer than 2 folds. If one was requested, then generate 2 and use one for training and
        # the other for testing.
        partitionsToGenerate = cvFolds if cvFolds > 1 else 2

        # Generate the stratified cross validation folds.
        partitions = np.array(partition_dataset.main(allExampleClasses, partitionsToGenerate, True))

        # Generate the permutations of each partition to use. The permutations will define the shuffling of the
        # training data that we will use. Calculating these upfront will cause each combination of parameters
        # to use the same permutations.
        permutations = {}
        maxNumIterations = max(numIters)  # Maximum number of iterations used.
        for i in range(cvFolds):
            # Determine the number of training examples used for this fold. The number is the total number of
            # training examples minus the number used in this fold (as these will be used for testing).
            trainingExamples = (partitions != i) & patientIndicesToUse
            trainingInPart = sum(trainingExamples)

            # Generate a list containing copies of the indices for the training examples in this fold.
            # If there are 10 training examples for this fold and the maximum number of iterations is 5, then
            # indexLists will contain 5 copies of the integers 0..9.
            indexLists = [list(range(trainingInPart)) for _ in range(maxNumIterations)]

            # Finally, shuffle the index lists to generate the permutations.
            for j in range(maxNumIterations):
                random.shuffle(indexLists[j])
            permutations[i] = indexLists

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

                # Train and test on each fold if cvFolds > 1. Otherwise, train on one fold and test on the other.
                classifier, descent, performanceOverFolds, predictions = train_model.mini_batch_e_net_cv(
                    dataMatrixSubset, allExampleClasses, patientIndicesToUse, partitions, classesUsed, permutations,
                        lambdaVal, elasticNetRatio, batchSize, numIterations, cvFolds)

                # Write out the performance of the model over all folds.
                fidPerformance.write("\t{0:s}".format('\t'.join(["{0:1.4F}".format(i) for i in performanceOverFolds])))

                if cvFolds == 1:
                    # If only one fold, then record the descent.
                    fidPerformance.write("\t{0:s}\n".format(','.join(["{0:1.4f}".format(i) for i in descent[0]])))
                else:
                    # Record G mean of predictions across all folds using this parameter combination.
                    # If the predicted class of any example is 0, then the example did not actually have its class predicted.
                    # This is due to the fact that real class integer representations begin at 1.
                    examplesUsed = predictions != 0
                    gMean = calc_metrics.calc_g_mean(predictions[examplesUsed], allExampleClasses[examplesUsed])
                    fidPerformance.write("\t{0:1.4f}\n".format(gMean))