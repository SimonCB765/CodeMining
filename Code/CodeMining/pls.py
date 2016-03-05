import sys

def pls():
    """"""

    # Determine dimensions of inputs.
    [numObservations, numPredictors] = X.shape
    [numObservations, numResponses] = Y.shape

    # Create a mapping from patient indices to the class of the patient.
    mapPatientsToClass = {}  # A mapping from the patient indices to the class they belong.
    mapClassToNumber = {}  # A mapping from each class to the integer used to represent it.
    for ind, i in enumerate(classExamples):
        mapClassToNumber[i] = ind
        for j in classExamples[i]:
            mapPatientsToClass[j] = i

    # Define the initial formats for the training and ambiguous matrices.
    # The "Patients" and "Codes" lists will contain paired patient and code indices to use when indexing the data matrices, and will be the same length.
    # For example, (dataMatrix["Patients"][i], dataMatrix["Codes"][i]) will be an index into the data matrix indicating that the patient with index
    # dataMatrix["Patients"][i] has code with index dataMatrix["Codes"][i] in their record.
    # The lists will be used to construct sparse matrices for the training and predictions.
    dataMatrix = {"Patients" : [], "Codes" : []}
    ambiguousMatrix = {"Patients" : [], "Codes" : []}

    # Convert the data matrix to an appropriate sparse matrix format.
    dataMatrix = sparse.coo_matrix((numpy.ones(len(dataMatrix["Patients"])), (dataMatrix["Patients"], dataMatrix["Codes"])))
    dataMatrix = sparse.csr_matrix(dataMatrix)  # CSR is more efficient for computations and has sorted row indices.

    # Convert the ambiguous example matrix to an appropriate sparse matrix format (if there are any ambiguous examples).
    isAmbiguousExamples = len(ambiguousMatrix["Patients"]) != 0
    if isAmbiguousExamples:
        ambiguousMatrix = sparse.coo_matrix((numpy.ones(len(ambiguousMatrix["Patients"])), (ambiguousMatrix["Patients"], ambiguousMatrix["Codes"])))
        ambiguousMatrix = sparse.csc_matrix(ambiguousMatrix)  # CSC is more efficient for computations and has sorted column indices.

    #=================================#
    # Train the initial PLS-DA model. #
    #=================================#
    # Create the target vector.
    targetVector = []
    for i in range(dataMatrix.shape[0]):
        targetVector.append(mapClassToNumber[mapPatientsToClass[i]])
    targetVector = numpy.array(targetVector)

    # Determine which codes should be suppressed (if any).
    trainingCodeIndices = set(codeToIndexMap.values())  # The indices of the codes to use for training.
    if isOneClassCodesIgnored:
        # Codes should not be used for training if they appear in only one class.
        for i in classes:
            # Find codes that are in class i and one of the other classes
            examplesInClass = (targetVector == mapClassToNumber[i])  # Boolean array indicating whether an example is in class i.
            examplesNotInClass = numpy.logical_not(examplesInClass)  # Boolean array indicating whether an example is NOT in class i.
            codesInClass = set(sparse.find(dataMatrix[examplesInClass, :])[1])  # Indices of codes that are in an example of class i.
            codesNotInClass = set(sparse.find(dataMatrix[examplesNotInClass, :])[1])  # Indices of codes that are in an example of class that is NOT i.
            singleClassCodes = codesInClass - codesNotInClass  # The codes only present in examples of class i.
            trainingCodeIndices -= singleClassCodes  # Remove the codes that are only present in class i from the list of codes to use for training.
    trainingCodeIndices = list(trainingCodeIndices)


    sys.exit()

    import time
    from sklearn.linear_model import ElasticNetCV
    start = time.process_time()
    print(start)
    enet = ElasticNetCV(cv=4).fit(dataMatrix, targetVector)
    end = time.process_time()
    print(sorted(enet.coef_[enet.coef_ != 0.0])[:10])
    print(enet.alpha_)
    print(enet.intercept_)
    print(start, end, end-start)

    # Create the target matrix for the initial model.
    # This will be a matrix with one row per patient being used, and one column per class.
    # targetMatrix[i, j] == 1 if patient i belongs to class j, else targetMatrix[i, j] == 0.
    targetMatrix = numpy.zeros((dataMatrix.shape[0], len(classExamples)))
    for i in range(dataMatrix.shape[0]):
        targetMatrix[i, mapClassToNumber[mapPatientsToClass[i]]] = 1
    print(targetMatrix)
    print(sum(targetMatrix))
    print(mapClassToNumber)
    pls2 = PLSRegression(n_components=2)
    pls2.fit(dataMatrix, targetMatrix)

if __name__ == '__main__':
    # Imports only needed if running from the command line.
    import extract_child_codes

    #=========================#
    # Parse the user's input. #
    #=========================#
    parser = argparse.ArgumentParser(
        description="===============PLS used for code mining==================",
        epilog="===============Stuff about looking at the README and yadda yadda==========\nAssumptions on data========")
    parser.add_argument("dataset", help="Location of the dataset to train the algorithm on.")
    parser.add_argument("codeMap", help="Location of the file containing the mapping between codes and their descriptions.")
    parser.add_argument("classes", help="Location of the JSON format file containing the class definitions.")
    parser.add_argument("-f", "--folds", type=int, default=10, help="Number of folds to use in the cross validation process that optimises the model.")
    parser.add_argument("-d", "--discard", type=float, default=0.0, help="=================================")
    parser.add_argument("-m", "--comp", type=int, default=10, help="Maximum number of PLS components to test.")
    parser.add_argument("-o", "--outDir", default="Results", help="Directory where the results should be recorded.")
    parser.add_argument("-s", "--suppress", action='store_true', default=False, help="Set to prevent codes that only appear in one class from being used for training.")
    args = parser.parse_args()

    fileDataset = args.dataset
    fileCodeMap = args.codeMap
    fileClasses = args.classes
    foldsToUse = args.folds
    discardThreshold = args.discard
    maxComponents = args.comp
    dirResults = args.outDir
    isOneClassCodesIgnored = args.suppress

    #========================================#
    # Process and validate the user's input. #
    #========================================#
    errorsFound = []  # List recording all error messages to display.

    if not os.path.isfile(fileDataset):
        # The input dataset must exist to continue.
        errorsFound.append("The dataset file does not exist.")
    if not os.path.isfile(fileCodeMap):
        # The mapping from codes to their descriptions must exist.
        errorsFound.append("The code mappings file does not exist.")
    if foldsToUse < 2:
        # Cross validation can only take place with at least 2 folds.
        errorsFound.append("A minimum of 2 cross validation folds is needed.")
    if (discardThreshold < 0.0) or (discardThreshold > 1.0):
        # It makes no sense to have a discard threshold outside the range [0.0, 1.0].
        errorsFound.append("The discard threshold must be between 0.0 and 1.0.")
    if maxComponents < 1:
        # PLS needs at least one component.
        errorsFound.append("The maximum number of components must be at least 1.")
    if not os.path.exists(dirResults):
        # If the results directory does not exist, then it must be able to be created.
        try:
            os.mkdir(dirResults)
        except OSError as err:
            errorsFound.append("Error creating results directory: {0}".format(err))

    # Process the class information provided.
    if not os.path.isfile(fileClasses):
        # The mapping from codes to their descriptions must exist.
        errorsFound.append("The file of classes does not exist.")
    else:
        readClasses = open(fileClasses, 'r')
        jsonClasses = json.load(readClasses)
        readClasses.close()
        classesWithoutCodes = [i for i in jsonClasses if not jsonClasses[i]]  # Names of all classes without codes.

        # Determine whether there are any errors in the class file itself.
        isCollectorClass = False  # Whether there is a class with no codes that will be used to collect all examples not belonging to the other classes.
        if len(classesWithoutCodes) == 1:
            # There is one class that will contain all examples that do not belong to another class.
            isCollectorClass = True
            collectorClass = classesWithoutCodes[0]
        elif len(classesWithoutCodes) > 1:
            # There can be at most one class without supplied codes.
            errorsFound.append("The maximum number of classes without provided codes is 1. Classes without supplied codes were: {0:s}".format(','.join(classesWithoutCodes)))
        if len(set(jsonClasses)) < len(jsonClasses):
            # Not all class names are unique.
            nonUniqueNames = [i for i in classNameOccurences if classNameOccurences[i] > 1]
            errorsFound.append("Not all class names are unique. Classes without unique names were: {0:s}".format(','.join(nonUniqueNames)))

    # Exit if errors were found.
    if errorsFound:
        print("\n\nThe following errors were encountered while parsing the input parameters:\n")
        print('\n'.join(errorsFound))
        sys.exit()

    # Extract the code mapping.
    codeMapping = {}  # The mapping from codes to their descriptions.
    with open(fileCodeMap, 'r') as readCodeMap:
        for line in readCodeMap:
            lineChunks = (line.strip()).split('\t')
            codeMapping[lineChunks[0]] = lineChunks[1]