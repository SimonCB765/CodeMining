import argparse
import collections
import functools
import itertools
import json
import numpy
import os
import re
from scipy import sparse
from sklearn.cross_decomposition import PLSRegression
import sys


def main(args):
    """Runs the code mining.

    :param args:    The command line arguments.
    :type args:     list

    """

    #=========================#
    # Parse the user's input. #
    #=========================#
    parser = argparse.ArgumentParser(
        description="===============PLS used for code mining==================",
        epilog="===============Stuff about looking at the README and yadda yadda==========\nAssumptions on data - PAtientID\tCode pair only occurs once in file\nAll of a patients record is consecutive========")
    parser.add_argument("dataset", help="Location of the dataset to train the algorithm on.")
    parser.add_argument("codeMap", help="Location of the file containing the mapping between codes and their descriptions.")
    parser.add_argument("classes", help="Location of the JSON format file containing the class definitions.")
    parser.add_argument("-f", "--folds", type=int, default=10, help="Number of folds to use in the cross validation process that optimises the model.")
    parser.add_argument("-d", "--discard", type=float, default=0.0, help="=================================")
    parser.add_argument("-m", "--comp", type=int, default=10, help="Maximum number of PLS components to test.")
    parser.add_argument("-o", "--outDir", default="Results", help="Directory where the results should be recorded.")
    parser.add_argument("-i", "--infrequent", type=int, default=50, help="Minimum number of patients that a code must appear in before it will be used.")
    parser.add_argument("-c", "--codeDensity", type=int, default=0, help="Minimum number of unique FREQUENTLY COCCURING codes that a patient must have in their record to be used.")
    parser.add_argument("-s", "--suppress", action='store_true', default=False, help="Whether codes that only appear in one class should not be used for training.")
    args = parser.parse_args()

    fileDataset = args.dataset
    fileCodeMap = args.codeMap
    fileClasses = args.classes
    foldsToUse = args.folds
    discardThreshold = args.discard
    maxComponents = args.comp
    dirResults = args.outDir
    minPatientsPerCode = args.infrequent
    codeDensity = args.codeDensity
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

    # Setup directory to record the indexed dataset.
    dirIndexedData = dirResults + "/IndexedData"  # Directory containing the indexed subset of the input dataset.
    if not os.path.isdir(dirIndexedData):
        os.mkdir(dirIndexedData)
    fileIndexedDataset = dirIndexedData + "/IndexedPatientData.tsv"  # The indexed subset of the input dataset.
    fileCodeIndices = dirIndexedData + "/CodeIndices.tsv"  # File containing the mapping from each code to its index in the indexed dataset.
    filePatientIndices = dirIndexedData + "/PatientIndices.tsv"  # File containing the mapping from each patient ID to its index in the indexed dataset.
    fileAmbigDataset = dirIndexedData + "/IndexedAmbigData.tsv"  # File containing the indexed subset of the input dataset containing ambiguous patients.
    fileAmbigIndices = dirIndexedData + "/AmbiguousPatientClasses.tsv"  # File containing ambiguous patients, their indices and their classes.

    #=======================================================#
    # Determine the codes in the dataset that will be used. #
    #=======================================================#
    # Determine the number of different patients that each code occurs with.
    codeFrequencies = collections.defaultdict(int)  # Mapping recording the number of patients each code occurs with.
    with open(fileDataset, 'r') as readDataset:
        for line in readDataset:
            # As each line records all occurrences of a patient and a specific code (i.e. the code/patient pair should appear nowhere else in the file),
            # you can simply add one to the record of the number of patients the code appears with.
            lineChunks = line.split('\t')
            codeFrequencies[lineChunks[1]] += 1

    # Record the number of patients per code.
    fileCodeFrequencies = dirResults + "/PatientsPerCode.tsv"  # File storing the number of patients each code is found in.
    with open(fileCodeFrequencies, 'w') as writeCodeFrequencies:
        for i in sorted(codeFrequencies):
            writeCodeFrequencies.write("{0:s}\t{1:d}\n".format(i, codeFrequencies[i]))

    # Determine the codes that occur frequently enough to be used.
    codesToUse = set([i for i in codeFrequencies if codeFrequencies[i] >= minPatientsPerCode])

    # Create and record the mapping from codes that are being used to their indices.
    codeToIndexMap = dict([(x, ind) for ind, x in enumerate(codesToUse)])
    with open(fileCodeIndices, 'w') as writeCodeIndices:
        for i in sorted(codeToIndexMap):
            writeCodeIndices.write("{0:s}\t{1:d}\n".format(i, codeToIndexMap[i]))

    # Get the codes that are to be used to indicate patients belonging to each class.
    classes = {}  # A mapping from class names to the codes that define the class.
    mapCodesToClass = {}  # A mapping from the codes to the class they are used to determine.
    for i in jsonClasses:
        classes[i] = []
        for j in jsonClasses[i]:
            if j.get("GetChildren") and (j["GetChildren"].lower() == 'y'):
                # Get child codes of the current code if the class definition requests it.
                matchingCodes = extract_child_codes([j["Code"]], codesToUse)  # Get all frequent codes that are children of the current code.
                matchingCodes = [codeToIndexMap[k] for k in matchingCodes]  # Convert codes to code indices.
            else:
                # No need to get child codes, just use the parent.
                matchingCodes = [codeToIndexMap[j["Code"]]]
            classes[i].extend(matchingCodes)  # Add the codes to the list of codes defining class i.
            mapCodesToClass.update(dict([(k, i) for k in matchingCodes]))  # Update with the class of the child codes found.

    #=============================#
    # Create the indexed dataset. #
    #=============================#
    # The indexed dataset will convert the original PatientID\tCode\tCount format of the input dataset to a
    # PatientIndex\tCodeIndex\tCount format. In addition, the only lines from the original dataset that get added
    # to the indexed dataset will be those that contain a code that occurs frequently enough to be used AND are
    # for a patient belonging to one of the classes. Ambiguous patients (those that belong to more than one class)
    # will be entered into a separate dataset.

    # Define the initial formats for the training and ambiguous matrices.
    # The "Patients" and "Codes" lists will contain paired patient and code indices to use when indexing the data matrices, and will be the same length.
    # For example, (dataMatrix["Patients"][i], dataMatrix["Codes"][i]) will be an index into the data matrix indicating that the patient with index
    # dataMatrix["Patients"][i] has code with index dataMatrix["Codes"][i] in their record.
    # The lists will be used to construct sparse matrices for the training and predictions.
    dataMatrix = {"Patients" : [], "Codes" : []}
    ambiguousMatrix = {"Patients" : [], "Codes" : []}

    nextPatientIndex = 0  # The index to be associated with the next patient to be recorded in the indexed dataset.
    nextAmbigIndex = 0  # The index to be associated with the next ambiguous patient to be recorded in the ambiguous dataset.
    currentPatientID = ''  # The ID of the current patient having their record examined.
    currentPatientRecord = []  # A list containing all rows of the current patient's record with a code in codesToUse (i.e. a frequently occurring code).
    classesOfPatient = set([])  # The classes that the current patient being looked at belongs to.
    classExamples = dict([(i, set([])) for i in classes])  # The indices of the patients that belong to each class.
    with open(fileDataset, 'r') as readDataset, open(fileIndexedDataset, 'w') as writeIndexedDataset, open(fileAmbigDataset, 'w') as writeAmbigDataset, \
         open(filePatientIndices, 'w') as writePatientIndices, open(fileAmbigIndices, 'w') as writeAmbigIndices:
        for line in readDataset:
            lineChunks = (line.strip()).split('\t')
            patientID = lineChunks[0]
            codeIndex = codeToIndexMap.get(lineChunks[1])

            # Only write out a patient's record once a new patient is encountered.
            # The old patient record will only be written out if the patient belongs to a class and
            # their record contains a code in codesToUse (i.e. a frequently occurring code).
            if currentPatientID != patientID:
                # A new patient has been encountered.
                if len(currentPatientRecord) >= codeDensity:
                    # The patient has at least codeDensity entries in their record with a frequently occurring code (as currentPatientRecord only
                    # contains lines from the patient's record that contain a code in codesToUse).
                    if len(classesOfPatient) == 1:
                        # The patient belongs to exactly ONE class.
                        classExamples[classesOfPatient.pop()].add(nextPatientIndex)
                        writePatientIndices.write("{0:s}\t{1:d}\n".format(patientID, nextPatientIndex))
                        dataMatrix["Patients"].extend([nextPatientIndex] * len(currentPatientRecord))
                        dataMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                        for i in currentPatientRecord:
                            writeIndexedDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextPatientIndex, i["Code"], i["Count"]))
                        nextPatientIndex += 1
                    elif len(classesOfPatient) > 1:
                        # The patient is ambiguous as they belong to multiple classes.
                        writeAmbigIndices.write("{0:s}\t{1:d}\t{2:s}\n".format(patientID, nextAmbigIndex, ','.join(classesOfPatient)))
                        ambiguousMatrix["Patients"].extend([nextAmbigIndex] * len(currentPatientRecord))
                        ambiguousMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                        for i in currentPatientRecord:
                            writeAmbigDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextAmbigIndex, i["Code"], i["Count"]))
                        nextAmbigIndex += 1
                    elif isCollectorClass:
                        # The patient did not belong to any classes, but there is a collector class.
                        classExamples[collectorClass].add(nextPatientIndex)
                        writePatientIndices.write("{0:s}\t{1:d}\n".format(patientID, nextPatientIndex))
                        dataMatrix["Patients"].extend([nextPatientIndex] * len(currentPatientRecord))
                        dataMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                        for i in currentPatientRecord:
                            writeIndexedDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextPatientIndex, i["Code"], i["Count"]))
                        nextPatientIndex += 1
                # Reset values for the next patient.
                currentPatientID = patientID
                currentPatientRecord = []
                classesOfPatient = set([])

            # Check whether the current line should be included in the current patient's record.
            if codeIndex:
                # The code exists in the codeToIndexMap mapping, is in codesToUse and therefore frequently occurring.
                # We are therefore interested in this line, and it should be included in the patient's record.
                currentPatientRecord.append({"Code" : codeIndex, "Count" : lineChunks[2]})

                # Check whether the patient belongs to a class based on the code on this line of their record.
                patientClass = mapCodesToClass.get(codeIndex)  # The class of the patient (or None if the code does not indicate class membership).
                if patientClass:
                    classesOfPatient.add(patientClass)

        # Handle the last patient (who can't get checked within the loop).
        if len(currentPatientRecord) >= codeDensity:
            # The patient has at least codeDensity entries in their record with a frequently occurring code (as currentPatientRecord only
            # contains lines from the patient's record that contain a code in codesToUse).
            if len(classesOfPatient) == 1:
                # The patient belongs to exactly ONE class.
                classExamples[classesOfPatient.pop()].add(nextPatientIndex)
                writePatientIndices.write("{0:s}\t{1:d}\n".format(patientID, nextPatientIndex))
                dataMatrix["Patients"].extend([nextPatientIndex] * len(currentPatientRecord))
                dataMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                for i in currentPatientRecord:
                    writeIndexedDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextPatientIndex, i["Code"], i["Count"]))
                nextPatientIndex += 1
            elif len(classesOfPatient) > 1:
                # The patient is ambiguous as they belong to multiple classes.
                writeAmbigIndices.write("{0:s}\t{1:d}\t{2:s}\n".format(patientID, nextAmbigIndex, ','.join(classesOfPatient)))
                ambiguousMatrix["Patients"].extend([nextAmbigIndex] * len(currentPatientRecord))
                ambiguousMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                for i in currentPatientRecord:
                    writeAmbigDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextAmbigIndex, i["Code"], i["Count"]))
                nextAmbigIndex += 1
            elif isCollectorClass:
                # The patient did not belong to any classes, but there is a collector class.
                classExamples[collectorClass].add(nextPatientIndex)
                writePatientIndices.write("{0:s}\t{1:d}\n".format(patientID, nextPatientIndex))
                dataMatrix["Patients"].extend([nextPatientIndex] * len(currentPatientRecord))
                dataMatrix["Codes"].extend([i["Code"] for i in currentPatientRecord])
                for i in currentPatientRecord:
                    writeIndexedDataset.write("{0:d}\t{1:d}\t{2:s}\n".format(nextPatientIndex, i["Code"], i["Count"]))
                nextPatientIndex += 1

    # Check whether there are any classes without examples in them.
    emptyClasses = [i for i in classExamples if len(classExamples[i]) == 0]
    if emptyClasses:
        print("\n\nThe following classes contain no examples that are not ambiguous:\n")
        print('\n'.join(emptyClasses))
        sys.exit()

    # Create a mapping from patient indices to the class of the patient.
    mapPatientsToClass = {}  # A mapping from the patient indices to the class they belong.
    mapClassToNumber = {}  # A mapping from each class to the integer used to represent it.
    for ind, i in enumerate(classExamples):
        mapClassToNumber[i] = ind
        for j in classExamples[i]:
            mapPatientsToClass[j] = i

    # Convert the data matrices to an appropriate sparse matrix format.
    dataMatrix = sparse.coo_matrix((numpy.ones(len(dataMatrix["Patients"])), (dataMatrix["Patients"], dataMatrix["Codes"])))
    dataMatrix = sparse.csr_matrix(dataMatrix)  # CSR is more efficient for computations and has sorted row indices.
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


def extract_child_codes(parentCodes, allCodes):
    """Extract all codes that are beneath the parent codes in the code hierarchy.

    A code (b) is 'beneath' another one (a) if b[:len(a)] == a.
    Example:
        parentCodes = ["ABC", "XYZ"]
        allCodes = ["ABCD", "ABC12", "1ABC", "AB", "XYZ", "XYZ01", "DEF12"]
        return = ["ABCD", "ABC12", "XYZ", "XYZ01"]

    :param parentCodes:     The codes at the root of the hierarchy substree(s) to be extracted.
    :type parentCodes:      list
    :param allCodes:        The codes to search for children in. Each code should appear once.
    :type allCodes:         list

    """

    regex = re.compile('|'.join(parentCodes))  # Compiled regular expression pattern code1|code2|code3|...|codeN.
    return [i for i in allCodes if regex.match(i)]


if __name__ == '__main__':
    main(sys.argv[1:])