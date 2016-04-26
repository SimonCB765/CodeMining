# Python imports.
import collections
import os
import sys

# User imports.
import CodeMiningPython.extract_child_codes


def select_data_subset(fileDataset, jsonClasses, dirResults="Results", minPatientsPerCode=50, codeDensity=0, isCollectorClass=False, collectorClass=None):
    """Selects and records a subset of the input data that meets given criteria.

    :param fileDataset:         File containing the input dataset to create the subset of.
    :type fileDataset:          string
    :param jsonClasses:         The classes with their associated code and hierarchy information (see README for more information).
    :type jsonClasses:          dict (as produced by json.load(fp) where fp is a .read()-supporting file-like object containing a JSON document)
    :param dirResults:          Location to save the generated subset.
    :type dirResults:           string
    :param minPatientsPerCode:  The minimum number of patients that a code must be associated with before it is kept in the subset.
    :type minPatientsPerCode:   int
    :param codeDensity:         The minimum number of unique codes in a patient's record meeting the minPatientsPerCode criterion needed before they are recorded in the subset.
    :type codeDensity:          int
    :param isCollectorClass:    Whether there is a class that will be used to collect all examples not belonging to the other classes.
    :type isCollectorClass:     boolean
    :param collectorClass:      The class to use as a collector.
    :type collectorClass:       string (or None)

    """

    # Setup the directory structure where the cut down dataset will be recorded.
    dirCutDownData = dirResults + "/CutDownData"  # Directory containing the cut down subset of the input dataset.
    if os.path.exists(dirCutDownData):
        if not os.path.isdir(dirCutDownData):
            # Location exists but is not a directory.
            print("Location {0:s} exists but is not a directory.".format(dirCutDownData))
            sys.exit()
    else:
        # Create the directory as the location is free.
        os.mkdir(dirCutDownData)
    fileCutDownDataset = dirCutDownData + "/CutDownPatientData.tsv"  # The cut down subset of the input dataset.
    fileAmbigDataset = dirCutDownData + "/AmbiguousPatientData.tsv"  # File containing patients that would be in the cut down dataset, but belong to multiple classes.

    #================================================================#
    # Determine the codes in the dataset that will be in the subset. #
    #================================================================#
    # Determine the number of different patients that each code occurs with.
    patientsPerCode = collections.defaultdict(int)  # Mapping recording the number of patients each code occurs with.
    codesPerPatient = collections.defaultdict(int)  # Mapping recording the number of codes each patient occurs with.
    with open(fileDataset, 'r') as readDataset:
        for line in readDataset:
            # As each line records all occurrences of a patient and a specific code (i.e. the code/patient pair should appear nowhere else in the file),
            # you can simply increment the mappings without worrying about double counting from finding the code/patient pair later in the file.
            lineChunks = line.split('\t')
            patientsPerCode[lineChunks[1]] += 1
            codesPerPatient[lineChunks[0]] += 1

    # Determine the codes that occur frequently enough to be in the subset.
    codesToUse = set([i for i in patientsPerCode if patientsPerCode[i] >= minPatientsPerCode])

    # Record the number of patients per code and whether the code is to be in the subset.
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
                matchingCodes = CodeMiningPython.extract_child_codes.main([j["Code"]], codesToUse)  # Get all frequent codes that are children of the current code.
            else:
                # No need to get child codes, just use the parent.
                matchingCodes = [j["Code"]]
            classes[i].extend(matchingCodes)  # Add the codes to the list of codes defining class i.
            mapCodesToClass.update(dict([(k, i) for k in matchingCodes]))  # Update with the class of the child codes found.

    # Record parameters used.
    fileParameters = dirResults + "/Parameters.txt"
    with open(fileParameters, 'w') as writeParameters:
        writeParameters.write("minPatientsPerCode: {0:d}\n".format(minPatientsPerCode))
        writeParameters.write("codeDensity: {0:d}\n".format(codeDensity))
        writeParameters.write("Classes:\n")
        for i in classes:
            writeParameters.write("\t{0:s}: {1:s}\n".format(i, (','.join(classes[i]) if classes[i] else "Collector")))

    #==============================#
    # Create the cut down dataset. #
    #==============================#
    # For a line in the original dataset to be included in the subset it must:
    #     1) Contain a code that occurs in enough patients' records (i.e. that occurs frequently enough)
    #     2) Contain a patient that belongs to one of the classes
    #     3) Contain a patient that has enough unique frequently occurring codes in their record
    # In the case of a collector class, all patients will belong to a class and criterion 2 will not matter.
    # Ambiguous patients (those with codes in their record that indicate they belong to more than one class) will be recorded in a separate subset.
    currentPatientID = ''  # The ID of the current patient having their record examined.
    currentPatientRecord = []  # A list containing all rows of the current patient's record with a code in codesToUse (i.e. a frequently occurring code).
    classesOfPatient = set([])  # The classes that the current patient being looked at belongs to.
    classExamples = dict([(i, set([])) for i in classes])  # A mapping recording the IDs of the patients that belong to each class.
    with open(fileDataset, 'r') as readDataset, open(fileCutDownDataset, 'w') as writeCutDownDataset, open(fileAmbigDataset, 'w') as writeAmbigDataset:
        for line in readDataset:
            lineChunks = (line.strip()).split('\t')
            patientID = lineChunks[0]
            code = lineChunks[1]
            isCodeUsed = code in codesToUse

            # Only write out a patient's record once a new patient is encountered.
            # The old patient's record will only be recorded if there are enough unique frequently occurring codes in it and the patient belongs to a class.
            # Entries with non-frequently occurring codes are not added to the record, so do need need to be checked for here.
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
                # The code exists in the codeToIndexMap mapping, is in codesToUse and so is frequently occurring.
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
        sys.exit()

    return [fileCutDownDataset, fileAmbigDataset]