"""Perform the code mining."""

# Python imports.
import logging
import sys

# User imports.
from . import generate_dataset
from . import train_model

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

    # Generate the data matrix, two index mappings, the mapping from case names to defining codes and the case mapping.
    # The patient index map is a bidirectional mapping between patients and their row indices in the data matrix.
    # The code index map is a bidirectional mapping between codes and their column indices in the data matrix.
    # The case definition mapping records the codes that define each case.
    # The case mapping records which patients meet which case definition. Ambiguous patients are added to a separate
    #   Ambiguous case.
    dataMatrix, mapPatientIndices, mapCodeIndices, caseDefs, cases = generate_dataset.main(
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

    # Train the model.
    train_model.main(dataMatrix, dirResults, mapPatientIndices, mapCodeIndices, caseDefs, cases, config)
