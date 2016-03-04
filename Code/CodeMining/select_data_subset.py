import argparse
import collections
import extract_child_codes
import json
import os
import sys


def select_data_subset(args):
    """Selects a subset of the input data that meets given criteria.

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
    parser.add_argument("classes", help="Location of the JSON format file containing the class definitions.")
    parser.add_argument("-o", "--outDir", default="Results", help="Directory where the results should be recorded.")
    parser.add_argument("-i", "--infrequent", type=int, default=50, help="Minimum number of patients that a code must appear in before it will be used.")
    parser.add_argument("-c", "--codeDensity", type=int, default=0, help="Minimum number of unique FREQUENTLY COCCURING codes that a patient must have in their record to be used.")
    parser.add_argument("-s", "--suppress", action='store_true', default=False, help="Whether codes that only appear in one class should not be used for training.")
    args = parser.parse_args()

    fileDataset = args.dataset
    fileClasses = args.classes
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

    # Setup directory to record the cut down dataset.
    dirCutDownData = dirResults + "/CutDownData"  # Directory containing the cut down subset of the input dataset.
    if not os.path.isdir(dirCutDownData):
        os.mkdir(dirCutDownData)
    fileCutDownDataset = dirCutDownData + "/CutDownPatientData.tsv"  # The cut down subset of the input dataset.
    fileAmbigDataset = dirCutDownData + "/AmbiguousPatientData.tsv"  # File containing patients that would be in the cut down dataset, but belong to multiple classes.

    #=======================================================#
    # Determine the codes in the dataset that will be used. #
    #=======================================================#
    # Determine the number of different patients that each code occurs with.
    patientsPerCode = collections.defaultdict(int)  # Mapping recording the number of patients each code occurs with.
    codesPerPatient = collections.defaultdict(int)  # Mapping recording the number of codes each patient occurs with.
    with open(fileDataset, 'r') as readDataset:
        for line in readDataset:
            # As each line records all occurrences of a patient and a specific code (i.e. the code/patient pair should appear nowhere else in the file),
            # you can simply add one to the record of the number of patients the code appears with.
            lineChunks = line.split('\t')
            patientsPerCode[lineChunks[1]] += 1
            codesPerPatient[lineChunks[0]] += 1

    # Determine the codes that occur frequently enough to be used.
    codesToUse = set([i for i in patientsPerCode if patientsPerCode[i] >= minPatientsPerCode])

    # Record the number of patients per code and whether the code was used.
    filePatientsPerCode = dirResults + "/CodeStats.tsv"  # File storing the number of patients each code is found in.
    with open(filePatientsPerCode, 'w') as writePatientsPerCode:
        writePatientsPerCode.write("Code\tNumPatients\tCodeUsed\n")
        for i in sorted(patientsPerCode):
            writePatientsPerCode.write("{0:s}\t{1:d}\t{2:s}\n".format(i, patientsPerCode[i], ('Y' if patientsPerCode[i] >= minPatientsPerCode else 'N')))

    # Record the number of codes per patient.
    fileCodesPerPatient = dirResults + "/PatientStats.tsv"  # File storing the number of codes each patient is found in.
    with open(fileCodesPerPatient, 'w') as writeCodesPerPatient:
        writeCodesPerPatient.write("Patient\tNumCodes\n")
        for i in sorted(codesPerPatient):
            writeCodesPerPatient.write("{0:s}\t{1:d}\n".format(i, codesPerPatient[i]))

    # Get the codes that are to be used to indicate patients belonging to each class.
    classes = {}  # A mapping from class names to the codes that define the class.
    mapCodesToClass = {}  # A mapping from the codes to the class they are used to determine.
    for i in jsonClasses:
        classes[i] = []
        for j in jsonClasses[i]:
            if j.get("GetChildren") and (j["GetChildren"].lower() == 'y'):
                # Get child codes of the current code if the class definition requests it.
                matchingCodes = extract_child_codes.main([j["Code"]], codesToUse)  # Get all frequent codes that are children of the current code.
            else:
                # No need to get child codes, just use the parent.
                matchingCodes = [j["Code"]]
            classes[i].extend(matchingCodes)  # Add the codes to the list of codes defining class i.
            mapCodesToClass.update(dict([(k, i) for k in matchingCodes]))  # Update with the class of the child codes found.

    #=============================#
    # Create the cut down dataset. #
    #=============================#
    # The only lines from the original dataset that get added
    # to the cut down dataset will be those that contain a code that occurs frequently enough to be used AND are
    # for a patient belonging to one of the classes. Ambiguous patients (those that belong to more than one class)
    # will be entered into a separate dataset.
    currentPatientID = ''  # The ID of the current patient having their record examined.
    currentPatientRecord = []  # A list containing all rows of the current patient's record with a code in codesToUse (i.e. a frequently occurring code).
    classesOfPatient = set([])  # The classes that the current patient being looked at belongs to.
    classExamples = dict([(i, set([])) for i in classes])  # The IDs of the patients that belong to each class.
    with open(fileDataset, 'r') as readDataset, open(fileCutDownDataset, 'w') as writeCutDownDataset, \
         open(fileAmbigDataset, 'w') as writeAmbigDataset:
        for line in readDataset:
            lineChunks = (line.strip()).split('\t')
            patientID = lineChunks[0]
            code = lineChunks[1]
            isCodeUsed = code in codesToUse

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
                        classExamples[classesOfPatient.pop()].add(currentPatientID)
                        for i in currentPatientRecord:
                            writeCutDownDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))
                    elif len(classesOfPatient) > 1:
                        # The patient is ambiguous as they belong to multiple classes.
                        for i in currentPatientRecord:
                            writeAmbigDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))
                    elif isCollectorClass:
                        # The patient did not belong to any classes, but there is a collector class.
                        classExamples[collectorClass].add(currentPatientID)
                        for i in currentPatientRecord:
                            writeCutDownDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))
                # Reset values for the next patient.
                currentPatientID = patientID
                currentPatientRecord = []
                classesOfPatient = set([])

            # Check whether the current line should be included in the current patient's record.
            if isCodeUsed:
                # The code exists in the codeToIndexMap mapping, is in codesToUse and therefore frequently occurring.
                # We are therefore interested in this line, and it should be included in the patient's record.
                currentPatientRecord.append({"Code" : code, "Count" : lineChunks[2]})

                # Check whether the patient belongs to a class based on the code on this line of their record.
                patientClass = mapCodesToClass.get(code)  # The class of the patient (or None if the code does not indicate class membership).
                if patientClass:
                    classesOfPatient.add(patientClass)

        # Handle the last patient (who can't get checked within the loop).
        if len(currentPatientRecord) >= codeDensity:
            # The patient has at least codeDensity entries in their record with a frequently occurring code (as currentPatientRecord only
            # contains lines from the patient's record that contain a code in codesToUse).
            if len(classesOfPatient) == 1:
                # The patient belongs to exactly ONE class.
                classExamples[classesOfPatient.pop()].add(currentPatientID)
                for i in currentPatientRecord:
                    writeCutDownDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))
            elif len(classesOfPatient) > 1:
                # The patient is ambiguous as they belong to multiple classes.
                for i in currentPatientRecord:
                    writeAmbigDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))
            elif isCollectorClass:
                # The patient did not belong to any classes, but there is a collector class.
                classExamples[collectorClass].add(currentPatientID)
                for i in currentPatientRecord:
                    writeCutDownDataset.write("{0:s}\t{1:s}\t{2:s}\n".format(currentPatientID, i["Code"], i["Count"]))

    # Check whether there are any classes without examples in them.
    emptyClasses = [i for i in classExamples if len(classExamples[i]) == 0]
    if emptyClasses:
        print("\n\nThe following classes contain no examples that are not ambiguous:\n")
        print('\n'.join(emptyClasses))
        sys.exit(patientID)


if __name__ == '__main__':
    main(sys.argv[1:])