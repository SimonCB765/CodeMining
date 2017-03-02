"""Process a code count file into the libSVM style format needed by the code mining."""

# Python imports.
import logging
import string
import re

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileUnprocessedData, fileProcessedData, patientColumn, codeColumn, countColumn, delimiter='\t',
         colsToUnbookend=None):
    """Process a file of code counts and convert it to the desired libSVM style format.

    Each patient/code pair is assumed to occur on only one line, and all of a patient's entries are assumed to follow
    one after another (i.e. each patient appears in a single contiguous block of lines in the file).

    :param fileUnprocessedData:     The location of the file containing the code count data.
    :type fileUnprocessedData:      str
    :param fileProcessedData:       The location to save the processed journal table data.
    :type fileProcessedData:        str
    :param patientColumn:           The column index in which the patient IDs are stored.
    :type patientColumn:            int
    :param codeColumn:              The column index in which the codes are stored.
    :type codeColumn:               int
    :param countColumn:             The column index in which the code counts are stored.
    :type countColumn:              int
    :param delimiter:               String used to separate columns in the dataset file.
    :type delimiter:                string
    :param colsToUnbookend:         Dictionary mapping from indices of columns that have characters bookending their
                                        entries to a list containing the character that is doing the bookending and
                                        whether it is only entries starting with a number that are bookended.
                                    For example, if colsToUnbookend is {0 : ['*', False], 2 : ['n', True]], then:
                                        entries in column 0 are bookended by '*' (e.g. *XXXX*) and this bookending
                                            occurs around ALL entries (e.g. *1234* AND *abcd*)
                                        entries in column 2 are bookended by 'n' (e.g. nXXXXn) and this bookending
                                            only occurs around entries starting with a number (e.g. n1234n but
                                            NOT nabcdn)
    :type colsToUnbookend:          dict of lists

    """

    LOGGER.info("Starting code count pre-processing.")

    currentPatient = None  # The ID of the patient who's record is currently being built.
    patientHistory = {}  # The data for the current patient.
    uniqueCodes = set()  # The codes used in the dataset.

    # Process the input arguments.
    colsToUnbookend = colsToUnbookend if colsToUnbookend else {}
    colsToUnbookend = {int(i): j for i, j in colsToUnbookend.items()}

    # Clean the input dataset.
    with open(fileUnprocessedData, 'r') as fidUnprocessed, open(fileProcessedData, 'w') as fidProcessed:
        for line in fidUnprocessed:
            # Remove non-alphanumeric characters as all patient IDs, codes and counts should contain only
            # alphanumeric characters.
            line = re.sub("[^a-zA-Z0-9\t]+", '', line)
            lineChunks = (line.strip()).split(delimiter)

            # Unbookend entries.
            for ind, i in enumerate(lineChunks):
                if (ind in colsToUnbookend) and (len(i) >= 3) and (i[0] == i[-1] == colsToUnbookend[ind][0]):
                    # The column is being unbookended, contains at least 3 characters and the first and last
                    # character of the entry are the same as the bookending character.
                    if (i[1] in string.digits) or colsToUnbookend[ind][1]:
                        # The first character of the entry is a number or we're unbookending every entry in the column.
                        lineChunks[ind] = lineChunks[ind][1:-1]

            # Extract the columns of interest.
            patientID = lineChunks[patientColumn]
            code = lineChunks[codeColumn]
            count = lineChunks[countColumn]

            if patientID and code:
                # The entry is valid as it has both a patient ID and code recorded for it.
                uniqueCodes.add(code)

                if patientID != currentPatient and currentPatient:
                    # A new patient has been found and this is not the first line of the file, so record the old
                    # patient and reset the patient data for the new patient.

                    # Write out the patient's history sorted by date from oldest to newest.
                    fidProcessed.write("{:s}\t{:s}\n".format(
                        currentPatient, '\t'.join(["{:s}:{:s}".format(i, patientHistory[i]) for i in patientHistory])
                    ))

                    # Reset the history and record of codes the patient has to prepare for the next patient.
                    patientHistory.clear()
                currentPatient = patientID

                # Add the entry to the patient's history.
                patientHistory[code] = count

        # Write out the final patient's history.
        fidProcessed.write("{:s}\t{:s}\n".format(
            currentPatient, '\t'.join(["{:s}:{:s}".format(i, patientHistory[i]) for i in patientHistory])
        ))

    # Write out the codes in the dataset.
    uniqueCodes = sorted(uniqueCodes)
    with open(fileProcessedData, 'r+') as fidProcessed:
        processed = fidProcessed.read()
        fidProcessed.seek(0)
        fidProcessed.write("{:s}\n{:s}".format('\t'.join(uniqueCodes), processed))
