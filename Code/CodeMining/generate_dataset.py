"""Create a sparse data matrix from a libsvm style bag of words data file."""

# Python imports.
import collections

# 3rd party imports.
import numpy as np
from scipy import sparse


def main(fileDataset, dirOutput, mapCodeToDescr):
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
    :return :               The sparse matrix and the bidirectional index mappings for the patients and codes.
    :rtype :                scipy.sparse.csr_matrix, dict, dict

    """

    # Extract the record of patients, codes and counts from the data file. The data is extracted such that
    # patientIDs[i] is a patient who has been associated with the code codes[i] counts[i] times.
    patientIDs, codes, counts = parse_dataset(fileDataset)

    # Create bidirectional mappings between patients and their row indices in the data matrix and between codes and
    # their column indices in the data matrix.
    mapPatientIndices = {}
    currentPatientIndex = 0
    codesPerPatient = collections.defaultdict(int)  # Records the number of unique codes for each patient.
    mapCodeIndices = {}
    currentCodeIndex = 0
    patientsPerCode = collections.defaultdict(int)  # Records the number of unique patients for each code.
    for ind, (i, j) in enumerate(zip(patientIDs, codes)):
        # Add the patient if they haven't already been seen.
        if i not in mapPatientIndices:
            mapPatientIndices[i] = currentPatientIndex
            mapPatientIndices[currentPatientIndex] = i
            currentPatientIndex += 1

        # Add the code if it hasn't already been seen.
        if j not in mapCodeIndices:
            mapCodeIndices[j] = currentCodeIndex
            mapCodeIndices[currentCodeIndex] = j
            currentCodeIndex += 1

        # Convert the patient ID and code to their numeric index.
        patientIDs[ind] = mapPatientIndices[i]
        codes[ind] = mapCodeIndices[j]

        # Update the code per patient and patient per code records.
        codesPerPatient[i] += 1
        patientsPerCode[j] += 1

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

    # Generate the sparse matrix.
    dt = np.dtype("Float64")  # 64-bit floating-point number.
    sparseMatrix = sparse.coo_matrix((counts, (patientIDs, codes)), dtype=dt)  # Create the sparse matrix.
    sparseMatrix = sparse.csr_matrix(sparseMatrix, dtype=dt)  # Convert to CSR format.

    return sparseMatrix, mapPatientIndices, mapCodeIndices


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
