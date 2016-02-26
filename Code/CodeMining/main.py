import argparse
import collections
import os
import re
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
    parser.add_argument("-c", "--classes", required=True, action="append", help="=====divide 3 things by ; and code by ,====y/Y for get children========")
    parser.add_argument("-f", "--folds", type=int, default=10, help="Number of folds to use in the cross validation process that optimises the model.")
    parser.add_argument("-d", "--discard", type=float, default=0.0, help="=================================")
    parser.add_argument("-m", "--comp", type=int, default=10, help="Maximum number of PLS components to test.")
    parser.add_argument("-o", "--outDir", default="Results", help="Directory where the results should be recorded.")
    parser.add_argument("-i", "--infrequent", type=int, default=50, help="Minimum number of patients that a code must appear in before it will be used.")
    args = parser.parse_args()

    fileDataset = args.dataset
    fileCodeMap = args.codeMap
    rawClasses = [i.split(';') for i in args.classes]
    foldsToUse = args.folds
    discardThreshold = args.discard
    maxComponents = args.comp
    dirResults = args.outDir
    minPatientsPerCode = args.infrequent

    #========================================#
    # Process and validate the user's input. #
    #========================================#
    errorsFound = []  # List recording all error messages to display.

    if not os.path.isfile(fileDataset):
        # The input dataset must exist to continue.
        errorsFound.append("The dataset file supplied does not exist.")
    if not os.path.isfile(fileCodeMap):
        # The mapping from codes to their descriptions must exist.
        errorsFound.append("The file of code mappings supplied does not exist.")
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
    classesWithoutCodes = []  # Names of all classes without codes.
    classNames = []  # Names of all classes.
    classes = []  # The processed list of classes.
    for i in rawClasses:
        i = i[:3]  # Strip off anything more than the first three elements, as these have no meaning.
        className = i[0]
        classNames.append(className)
        if len(i) == 1:
            # If there is only one element of the list, then no codes were supplied for the class.
            classesWithoutCodes.append(className)
            i.extend([[], False])
        elif len(i) == 2:
            # If there are two elements in the list, then whether child codes should be considered has been left off.
            i[1] = i[1].split(',')
            i.append(False)
        else:
            # All three components of the class information are here, so convert the third to a boolean value.
            i[1] = i[1].split(',')
            i[2] = i[2].lower() == 'y'
        classRecord = {"Name" : i[0], "Codes" : i[1], "Child" : i[2]}
        classes.append(classRecord)
    classNameOccurences = collections.Counter(classNames)  # Counts of all class names.

    if len(classesWithoutCodes) == 1:
        # There is one class that will contain all examples that do not belong to another class.
        isCollectorClass = True
        collectorClass = classesWithoutCodes[0]
    elif len(classesWithoutCodes) > 1:
        # There can be at most one class without supplied codes.
        errorsFound.append("The maximum number of classes without provided codes is 1. Classes without supplied codes were: {0:s}".format(','.join(classesWithoutCodes)))
    if any([classNameOccurences[i] > 1 for i in classNameOccurences]):
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
    codeIndices = dict([(x, ind) for ind, x in enumerate(codesToUse)])
    with open(fileCodeIndices, 'w') as writeCodeIndices:
        for i in sorted(codeIndices):
            writeCodeIndices.write("{0:s}\t{1:d}\n".format(i, codeIndices[i]))

    # Get codes for each class.
    mapCodesToClass = {}  # A mapping from the codes used to determine the classes to the class they are used to determine.
    for ind, x in enumerate(classes):
        if x["Codes"] and x["Child"]:
            # Only check child codes if there are codes representing the class and child codes are to be used.
            matchingCodes = extract_child_codes(x["Codes"], codesToUse)  # Get all codes being used that are children of the parent codes supplied.
            matchingCodes = [codeIndices[i] for i in matchingCodes]  # Convert codes to code indices.
            classes[ind]["Codes"] = matchingCodes
            mapCodesToClass.update(dict([(i, x["Name"]) for i in matchingCodes]))

    #=====================================================#
    # Create an indexed dataset without infrequent codes. #
    #=====================================================#
    # Generate the indexed dataset and the mapping from patient IDs to their respective indices.
    currentPatientIndex = -1  # Index for the current patient.
    currentPatientID = ''  # The ID of the current patient.
    isCurrentPatientUsed = True  # Whether the current patient belongs to a class specified by supplied codes.
                                 # This will immediately get set to False when reading the first line, bus needs to be True to prevent the first
                                 # patient potentially being recorded in both the container class and another one.
    classExamples = dict([(i["Name"], set([])) for i in classes])  # The indices of the examples for each class.
    with open(fileDataset, 'r') as readDataset, open(fileIndexedDataset, 'w') as writeIndexedDataset, open(filePatientIndices, 'w') as writePatientIndices:
        for line in readDataset:
            lineChunks = line.split('\t')
            patientID = lineChunks[0]
            codeIndex = codeIndices.get(lineChunks[1])
            
            # Check whether the start of a new patient's record block has been found.
            if currentPatientID != patientID:
                if isCollectorClass and (not isCurrentPatientUsed):
                    # There is a class that will contain all patients not belonging to a class specified by supplied codes,
                    # and the patient whose record has just finished did not belong to one of those classes.
                    classExamples[collectorClass].add(currentPatientIndex)
                currentPatientID = patientID
                isCurrentPatientUsed = False
                currentPatientIndex += 1
                writePatientIndices.write("{0:s}\t{1:d}\n".format(patientID, currentPatientIndex))
            
            # Check whether the line should be included in the indexed file.
            if codeIndex:
                # If the code exists in the code index mapping (and is therefore frequent enough to use),
                # then the line may need to be recorded in the indexed dataset.
                writeIndexedDataset.write("{0:d}\t{1:d}\t{2:s}".format(currentPatientIndex, codeIndex, lineChunks[2]))

                # Check whether the patient should be included as an example for one of the classes specified by supplied codes.
                patientClass = mapCodesToClass.get(codeIndex)
                if patientClass:
                    # The code on the line is one of the codes used to determine the classes.
                    classExamples[patientClass].add(currentPatientIndex)
                    isCurrentPatientUsed = True
    # Check whether the final patient needs to belong to the container class.
    if isCollectorClass and (not isCurrentPatientUsed):
        # There is a class that will contain all patients not belonging to a class specified by supplied codes,
        # and the last patient did not belong to one of those classes.
        classExamples[collectorClass].add(currentPatientIndex)

    # Check whether there are any classes without examples in them.
    emptyClasses = [i for i in classExamples if len(classExamples[i]) == 0]
    if emptyClasses:
        print("\n\nThe following classes contain no examples:\n")
        print('\n'.join(emptyClasses))
        sys.exit()

    print(classes)
    print(mapCodesToClass)
    print([(i, len(classExamples[i])) for i in classExamples])
    print(isCollectorClass)
    print(collectorClass)


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