"""Create a sparse data matrix from a libsvm style bag of words data file."""

# Python imports.
import collections
import re

# 3rd party imports.
import numpy as np
from scipy import sparse


def expand_case_definitions(mapCodeToDescr, config):
    """Expand the definition of each case by enumerating all codes that define it.

    This will expand wildcards and remove trailing fullstops. For example, if you have a case definition:
        ["ABC%", "XYZ.."]
    and the entire set of available codes is:
        ["ABCD", "ABC12", "1ABC", "AB", "XYZ", "XYZ01", "DEF12"]
    this will result in an enumerated case definition of:
        ["ABCD", "ABC12", "XYZ"]

    :param mapCodeToDescr:  The mapping from codes to their descriptions.
    :type mapCodeToDescr:   dict
    :param config:          The JSON-like object containing the configuration parameters to use.
    :type config:           JsonschemaManipulation.Configuration
    :return:                The case definitions expanded to include all codes defining the cases.
    :rtype:                 dict

    """

    # Determine the base case definitions.
    caseDefs = config.get_param(["CaseDefinitions"])[1]

    # Expand each case:
    expandedDefs = {}
    for i, j in caseDefs.items():
        codes = [k.replace('.', '') for k in j]
        codesToExpand = [k[:-1] for k in codes if k.endswith('%')]
        baseCodes = [i for i in codes if i not in codesToExpand]
        expandedCodes = []
        if codesToExpand:
            regex = re.compile('|'.join(codesToExpand))  # Compiled regular expression code1|code2|code3|...|codeN.
            expandedCodes = [k for k in mapCodeToDescr.keys() if regex.match(k)]
        expandedDefs[i] = set(baseCodes) | set(expandedCodes)

    return expandedDefs


def main(fileDataset, dirOutput, mapCodeToDescr, config):
    """Generate a normalised sparse matrix from a data file.

    Each line in the input file is expected to have 3 columns:
        1st - Patient ID
        2nd - Code
        3rd - Number of occurrences of the code within the patient's record.
    Each line forms a tuple recording the number of times a given code occurs in a given patient's record.

    :param fileDataset:     The location of the dataset to turn into a sparse matrix.
    :type fileDataset:      str
    :param dirOutput:       The location to save information about the dataset.
    :type dirOutput:        str
    :param mapCodeToDescr:  The mapping from codes to their descriptions.
    :type mapCodeToDescr:   dict
    :param config:          The JSON-like object containing the configuration parameters to use.
    :type config:           JsonschemaManipulation.Configuration
    :return:                The sparse matrix, bidirectional index mappings for the patients and codes, a mapping from
                            case names to the codes defining the case and a mapping from case names to the IDs of the
                            patients that meet the case definition. Patients that are ambiguous (i.e. meet multiple
                            case definitions) are added to a separate "Ambiguous" case.
    :rtype:                 scipy.sparse.csr_matrix, dict, dict, dict, dict

    """

    # Identify all codes that define each case.
    caseDefs = expand_case_definitions(mapCodeToDescr, config)

    # Determine if there is a collector case present.
    collectorCase = None
    for i, j in caseDefs.items():
        if not j:
            collectorCase = i

    # Extract the record of patients, codes and counts from the data file. The data is extracted such that
    # patientIDs[i] is a patient who has been associated with the code codes[i] counts[i] times.
    patientIDs, codes, counts = parse_dataset(fileDataset)

    # Determine the number of codes each patient is associated with and the number of patients each code is associated
    # with.
    codesPerPatient = collections.defaultdict(int)  # Records the number of unique codes for each patient.
    for i in patientIDs:
        codesPerPatient[i] += 1
    patientsPerCode = collections.defaultdict(int)  # Records the number of unique patients for each code.
    for i in codes:
        patientsPerCode[i] += 1

    # Write out the records of patients per code and codes per patient.
    fileCodesPerPatient = dirOutput + "/CodesPerPatient.tsv"
    with open(fileCodesPerPatient, 'w') as writeCodesPerPatient:
        for i in sorted(codesPerPatient):
            writeCodesPerPatient.write("{0:s}\t{1:d}\n".format(i, codesPerPatient[i]))
    filePatientsPerCode = dirOutput + "/PatientsPerCode.tsv"
    with open(filePatientsPerCode, 'w') as writePatientsPerCode:
        for i in sorted(patientsPerCode):
            writePatientsPerCode.write(
                "{0:s}\t{1:s}\t{2:d}\n".format(i, mapCodeToDescr.get(i, "Unknown code."), patientsPerCode[i])
            )

    # Generate the lists of patient IDs, codes and counts where the patients and codes are frequent enough in the data.
    codesOccurWith = config.get_param(["CodesOccurWith"])[1]
    patientsOccurWith = config.get_param(["PatientsOccurWith"])[1]
    filteredData = [
        (i, j, k) for i, j, k in zip(patientIDs, codes, counts)
        if (codesPerPatient[i] > codesOccurWith and patientsPerCode[j] > patientsOccurWith)
    ]
    patientIDs, codes, counts = map(list, zip(*filteredData))

    # Create bidirectional mappings between patients and their row indices in the data matrix and between codes and
    # their column indices in the data matrix.
    mapPatientIndices = {}
    allPatients = set()  # The set of all patients in the dataset.
    currentPatientIndex = 0
    mapCodeIndices = {}
    allCodes = set()  # The set of all codes in the dataset.
    currentCodeIndex = 0
    cases = collections.defaultdict(set)  # Dictionary mapping case names to the IDs of patients meeting the definition.
    for ind, (i, j) in enumerate(zip(patientIDs, codes)):
        # Add the patient if they haven't already been seen.
        if i not in mapPatientIndices:
            mapPatientIndices[i] = currentPatientIndex
            mapPatientIndices[currentPatientIndex] = i
            allPatients.add(i)
            currentPatientIndex += 1

        # Add the code if it hasn't already been seen.
        if j not in mapCodeIndices:
            mapCodeIndices[j] = currentCodeIndex
            mapCodeIndices[currentCodeIndex] = j
            allCodes.add(i)
            currentCodeIndex += 1

        # Convert the patient ID and code to their numeric index.
        patientIDs[ind] = mapPatientIndices[i]
        codes[ind] = mapCodeIndices[j]

        # Check whether the patient meets any case definitions.
        for case, caseCodes in caseDefs.items():
            if j in caseCodes:
                cases[case].add(i)

    # Generate the sparse matrix.
    dt = np.dtype("Float64")  # 64-bit floating-point number.
    sparseMatrix = sparse.coo_matrix((counts, (patientIDs, codes)), dtype=dt)  # Create the sparse matrix.
    sparseMatrix = sparse.csr_matrix(sparseMatrix, dtype=dt)  # Convert to CSR format.

    # Determine patients that are ambiguous and set up the collector case if there is onw.
    ambiguousPatients = set(allPatients)
    collectorPatients = set(allPatients)
    for i, j in cases.items():
        ambiguousPatients &= j
        collectorPatients -= j
    if collectorCase:
        cases[collectorCase] = collectorPatients

    # Remove ambiguous patients and move them to their own case.
    for i, j in cases.items():
        cases[i] = j - ambiguousPatients

    return sparseMatrix, ambiguousPatients, mapPatientIndices, mapCodeIndices, caseDefs, cases


def parse_dataset(fileData):
    """Parses dataset files formatted for the code mining.

    Each line (except for the header) in the input file is expected to have 2 columns:
        1st - Patient ID
        2nd - The code counts in the format "Code1:Count\tCode2:Count\t...".

    :param fileData:    The location of the file containing the data to parse.
    :type fileData:     str
    :return :           The lists of the patient IDs, codes and counts ordered such that if all lists are indexed
                        by i, then you get the number of times patient i was associated with a specific code.
    :rtype :            dict

    """

    patientIDs = []
    codes = []
    counts = []
    with open(fileData, 'r') as fidData:
        # Remove the header.
        _ = fidData.readline()

        for line in fidData:
            # Split the line in order to separate codes and counts.
            chunks = (line.strip()).split('\t')

            # Add the patient ID once per code they're associated with.
            patientIDs.extend([chunks[0]] * (len(chunks) - 1))

            # Update the codes and counts.
            codeCounts = [i.split(':') for i in chunks[1:]]
            codes.extend([i[0] for i in codeCounts])
            counts.extend([int(i[1]) for i in codeCounts])

    return patientIDs, codes, counts
