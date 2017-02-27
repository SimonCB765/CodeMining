"""Perform the code mining."""

# Python imports.
import logging
import sys

# User imports.
from . import generate_dataset
from . import train_model

# 3rd party imports.
import numpy as np

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileDataset, fileCodeMapping, dirResults, config):
    """Perform the clinical code mining.

    :param fileDataset:         File location containing the data to use.
    :type fileDataset:          str
    :param fileCodeMapping:     File containing the mapping between codes and their descriptions.
    :type fileCodeMapping:      str
    :param dirResults:          Location of the directory in which the results will be saved.
    :type dirResults:           str
    :param config:              The JSON-like object containing the configuration parameters to use.
    :type config:               JsonschemaManipulation.Configuration

    """

    # Generate the mapping from codes to their descriptions. Each line in the file should contain two entries (a code
    # and its description) separated by a tab.
    mapCodeToDescr = {}
    with open(fileCodeMapping, 'r') as fidCodes:
        for line in fidCodes:
            chunks = (line.strip()).split('\t')
            mapCodeToDescr[chunks[0]] = chunks[1]

    # Generate the data matrix, the IDs of the ambiguous patients, two index mappings, the mapping from case names to
    # defining codes and the case mapping.
    # The patient index map is a bidirectional mapping between patients and their row indices in the data matrix.
    # The code index map is a bidirectional mapping between codes and their column indices in the data matrix.
    # The case definition mapping records the codes that define each case.
    # The case mapping records which patients meet which case definition.
    dataMatrix, ambiguousPatients, mapPatientIndices, mapCodeIndices, caseDefs, cases = generate_dataset.main(
        fileDataset, dirResults, mapCodeToDescr, config
    )

    # Check whether there are any cases with no patients (the ambiguous patient case does not need accounting for as
    # it is only in the case definition when there are ambiguous patients).
    noExampleCases = [i for i, j in cases.items() if len(j) == 0]
    if noExampleCases:
        LOGGER.error("The following cases have no unambiguous patients: {:s}".format(
            ','.join(noExampleCases))
        )
        print("\nErrors were encountered following case identification. Please see the log file for details.\n")
        sys.exit()

    # Map case names to their integer representation and determine the integer case definition that each patient meets.
    # A value of NaN is used to indicate that a patient does not meet any case definition, and is therefore not used.
    dataClasses = np.empty(dataMatrix.shape[0])
    dataClasses.fill(np.nan)
    caseIntegerReps = {}
    for ind, (i, j) in enumerate(cases.items()):
        caseIntegerReps[i] = ind
        patientIndices = [k for k in j]
        dataClasses[patientIndices] = ind

    # Calculate masks for the patients and the codes. These will be used to select only those patients and codes
    # that are to be used for training/testing.
    patientsUsed = [k for i, j in cases.items() for k in j]
    patientMask = np.zeros(dataMatrix.shape[0], dtype=bool)
    patientMask[patientsUsed] = 1  # Set patients the meet a case definition to be used.
    codesUsed = [k for i, j in caseDefs.items() for k in j]
    codeMask = np.ones(dataMatrix.shape[1], dtype=bool)
    codeMask[codesUsed] = 0  # Mask out the codes used to calculate case membership.

    # Train the model.
    train_model.main(dataMatrix, dataClasses, dirResults, patientMask, codeMask, cases, config)
