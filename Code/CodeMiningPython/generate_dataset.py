"""Creates the normalised sparse data matrix."""

# Python imports.
import collections

# 3rd party imports.
import numpy as np
from scipy import sparse

# User imports.
import normalise_data_matrix


def main(fileDataset, dirOutput, mapCodeToDescr, normMethod=0, normParam=None):
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
    :param normMethod:      The code for the normalisation method to use.
    :type normMethod:       int
    :param normParam:       The parameters (if any) needed for the normalisation method.
    :type normParam:        list
    :return :               The sparse matrix and the index mappings for the patients and codes.
    :rtype :                scipy.sparse.csr_matrix, dict and dict

    """

    patientIDs, codes, counts = parse_dataset(fileDataset)  # Extract the information needed from the data file.

    # Determine a mapping from patients to their indices and a mapping from codes to their indices, and map
    # the lists of patients and codes into a list of integer indices.
    # The index of a patient/code is determined by how many unique patients/codes occur before it in the list.
    codesPerPatient = collections.defaultdict(int)  # Records the number of unique codes for each patient.
    patientsPerCode = collections.defaultdict(int)  # Records the number of unique patients for each code.
    currentPatientIndex = -1
    uniquePatients = {}
    currentCodeIndex = -1
    uniqueCodes = {}
    for ind, (p, c) in enumerate(zip(patientIDs, codes)):
        # Handle patients.
        if p not in uniquePatients:
            currentPatientIndex += 1
            uniquePatients[p] = currentPatientIndex
        patientIDs[ind] = currentPatientIndex
        codesPerPatient[p] += 1

        # Handle codes.
        if c not in uniqueCodes:
            currentCodeIndex += 1
            uniqueCodes[c] = currentCodeIndex
        codes[ind] = currentCodeIndex
        patientsPerCode[c] += 1

    # Write out the records of patients per code and codes per patient.
    fileCodesPerPatient = dirOutput + "/CodesPerPatient.tsv"
    with open(fileCodesPerPatient, 'w') as writeCodesPerPatient:
        for i in sorted(codesPerPatient):
            writeCodesPerPatient.write("{0:s}\t{1:d}\n".format(i, codesPerPatient[i]))
    filePatientsPerCode = dirOutput + "/PatientsPerCode.tsv"
    with open(filePatientsPerCode, 'w') as writePatientsPerCode:
        for i in sorted(patientsPerCode):
            writePatientsPerCode.write("{0:s}\t{1:s}\t{2:d}\n".format(i, mapCodeToDescr.get(i, "Unknown code."),
                                                                      patientsPerCode[i]))

    # Generate the sparse matrix. Use CSC matrix for normalisation as it gives fast column operations.
    # Use CSR for learning as it gives fast row operations.
    dt = np.dtype("Float64")  # 64-bit floating-point number
    sparseMatrix = sparse.coo_matrix((counts, (patientIDs, codes)), dtype=dt)  # Create the sparse matrix.
    sparseMatrix = sparse.csc_matrix(sparseMatrix, dtype=dt)  # Convert to CSC format for normalisation.
    sparseMatrix = normalise_data_matrix.main(sparseMatrix, normMethod, normParam)  # Normalise the data.
    sparseMatrix = sparse.csr_matrix(sparseMatrix, dtype=dt)  # Convert to CSR for code mining.

    return sparseMatrix, uniquePatients, uniqueCodes


def parse_dataset(fileData, delimiter='\t'):
    """Parses dataset files formatted for the code mining.

    Each line in the input file is expected to have 3 columns:
        1st - Patient ID
        2nd - Code
        3rd - Number of occurrences of the code within the patient's record.
    Each line forms a tuple recording the number of times a given code occurs in a given patient's record.

    :param fileData:    The location of the file containing the data to parse.
    :type fileData:     str
    :param delimiter:   The delimiter used to divide the columns in the dataset file.
    :type delimiter:    str
    :return :           The lists of the patient IDs, codes and counts.
    :rtype :            dict

    """

    patientIDs = []
    codes = []
    counts = []
    with open(fileData, 'r') as fidData:
        for line in fidData:
            chunks = (line.strip()).split(delimiter)

            # Some lines have blank values in one of the columns. Reject these.
            if ('' in chunks) or any([i.isspace() for i in chunks]):
                continue

            patientIDs.append(chunks[0])
            codes.append(chunks[1])
            counts.append(int(chunks[2]))

    return patientIDs, codes, counts