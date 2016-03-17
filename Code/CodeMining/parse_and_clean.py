import collections
import os
import string
import sys


def parse_and_clean(dataset, resultsDir, delimiter='\t', colsToStripCommas=None, colsToRemove=None, colsToUnbookend=None):
    """Parses and cleans the input dataset.

    No checking of the inputs is performed.

    Assumptions made about the input dataset:
    1) Each patient-code pair occurs only once in the file (i.e. appears on only one line).
    2) There is no header in the file.

    :param dataset:             Location of the file containing the input dataset.
    :type dataset:              string
    :param resultsDir:          Location to save the cleaned data and statistics about it.
    :type resultsDir:           string
    :param delimiter:           String used to separate columns in the dataset file.
    :type delimiter:            string
    :param colsToStripCommas:   Indices of columns that contain commas that need stripping out.
    :type colsToStripCommas:    list
    :param colsToRemove:        Indices of columns in the dataset to leave out of the cleaned dataset.
    :type colsToRemove:         list
    :param colsToUnbookend:     Lists indicating which columns have characters bookending their entries, which
                                    character is bookending the entries and whether only entries starting
                                    with a number are bookended.
                                For example, if colsToUnbookend is [[0, '*', False], [2, 'n', True]], then:
                                entries in column 0 are bookended by '*' (e.g. *XXXX*) and that this bookending
                                    occurs around ALL entries (e.g. *1234* AND *abcd*)
                                entries in column 2 are bookended by 'n' (e.g. nXXXXn) and that this bookending
                                    only occurs around entries starting with a number (e.g. n1234n but NOT nabcdn)
    :type colsToUnbookend:      list of lists
    :return :                   The location where the parsed and cleaned dataset was written out.
    :rtype :                    string

    """

    # Create the stats directory if needed.
    if os.path.exists(resultsDir):
        if not os.path.isdir(resultsDir):
            # Location exists but is not a directory.
            print("Location {0:s} exists but is not a directory.".format(resultsDir))
            sys.exit()
    else:
        # Create the directory as the location is free.
        os.mkdir(resultsDir)

    # Create the statistics output files.
    fileCleanedDataset = resultsDir + "/CleanedData.tsv"
    fileCodesPerPatient = resultsDir + "/CodesPerPatient.tsv"
    filePatientsPerCode = resultsDir + "/PatientsPerCode.tsv"

    # Process the input arguments.
    colsToStripCommas = colsToStripCommas if colsToStripCommas else []
    colsToRemove = colsToRemove if colsToRemove else []
    colsToUnbookend = colsToUnbookend if colsToUnbookend else []
    colsToUnbookend = dict([(i[0], i[1:]) for i in colsToUnbookend])

    # Clean the input dataset.
    codesPerPatient = collections.defaultdict(int)
    patientsPerCode = collections.defaultdict(int)
    with open(dataset, 'r') as readDataset, open(fileCleanedDataset, 'w') as writeCleanData:
        for line in readDataset:
            lineChunks = (line.strip()).split(delimiter)

            # Strip commas.
            lineChunks = [(i.replace(',', '') if ind in colsToStripCommas else i) for ind, i in enumerate(lineChunks)]

            # Unbookend entries.
            for ind, i in enumerate(lineChunks):
                if ((ind in colsToUnbookend) and (len(i) >= 3) and (i[0] == i[-1] == colsToUnbookend[ind][0])):
                    # The column is being unbookended, contains at least 3 characters and the first and last
                    # character of the entry are the same as the bookending character.
                    if ((i[1] in string.digits) or colsToUnbookend[ind][1]):
                        # The first character of the entry is a number or we're unbookending every entry in the column.
                        lineChunks[ind] = lineChunks[ind][1:-1]

            # Remove columns.
            lineChunks = [i for ind, i in enumerate(lineChunks) if not ind in colsToRemove]

            # Update the code/patient counts.
            codesPerPatient[lineChunks[0]] += 1
            patientsPerCode[lineChunks[1]] += 1

            # Rejoin the cleaned column entries, and then write them out.
            if all([len(i) > 0 for i in lineChunks]):
                # Only write the line out if there are no blanks entries in it.
                writeCleanData.write('\t'.join(lineChunks) + '\n')

    # Write out the records of patients per code and codes per patient.
    with open(fileCodesPerPatient, 'w') as writeCodesPerPatient:
        for i in sorted(codesPerPatient):
            writeCodesPerPatient.write("{0:s}\t{1:d}\n".format(i, codesPerPatient[i]))
    with open(filePatientsPerCode, 'w') as writePatientsPerCode:
        for i in sorted(patientsPerCode):
            writePatientsPerCode.write("{0:s}\t{1:d}\n".format(i, patientsPerCode[i]))

    return fileCleanedDataset