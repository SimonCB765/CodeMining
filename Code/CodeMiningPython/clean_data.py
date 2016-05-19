"""Cleans a dataset to get it ready for code mining

Each line in the data file is expected to have 3 columns:
    1st - The ID of the patient.
    2nd - The code.
    3rd - The number of times the patient has been associated with the code.

"""

# Python imports.
import re
import string


def main(fileDataset, fileCleanedDataset, codeColumn, delimiter='\t', colsToStripCommas=None, colsToRemove=None,
         colsToUnbookend=None):
    """Parses and cleans the input dataset.

    No checking of the inputs is performed.

    Assumptions made about the input dataset:
    1) Each patient-code pair occurs only once in the file (i.e. appears on only one line).
    2) There is no header in the file.

    :param fileDataset:         Location of the file containing the input dataset.
    :type fileDataset:          string
    :param fileCleanedDataset:  Location to save the cleaned data to.
    :type fileCleanedDataset:   string
    :param codeColumn:          The column in which the codes occur.
    :type codeColumn            int
    :param delimiter:           String used to separate columns in the dataset file.
    :type delimiter:            string
    :param colsToStripCommas:   Indices of columns that contain commas that need stripping out.
    :type colsToStripCommas:    list
    :param colsToRemove:        Indices of columns in the dataset to leave out of the cleaned dataset.
    :type colsToRemove:         list
    :param colsToUnbookend:     Dictionary indicating which columns have characters bookending their entries, which
                                    character is bookending the entries and whether only entries starting
                                    with a number are bookended.
                                For example, if colsToUnbookend is {0 : ['*', False], 2 : ['n', True]], then:
                                entries in column 0 are bookended by '*' (e.g. *XXXX*) and that this bookending
                                    occurs around ALL entries (e.g. *1234* AND *abcd*)
                                entries in column 2 are bookended by 'n' (e.g. nXXXXn) and that this bookending
                                    only occurs around entries starting with a number (e.g. n1234n but NOT nabcdn)
    :type colsToUnbookend:      dict of lists
    :return :                   The location where the parsed and cleaned dataset was written out.
    :rtype :                    string

    """

    # Process the input arguments.
    colsToStripCommas = colsToStripCommas if colsToStripCommas else []
    colsToRemove = colsToRemove if colsToRemove else []
    colsToUnbookend = colsToUnbookend if colsToUnbookend else {}

    # Clean the input dataset.
    with open(fileDataset, 'r') as readDataset, open(fileCleanedDataset, 'w') as writeCleanData:
        for line in readDataset:
            line = re.sub("^[^a-zA-Z0-9]*", '', line)  # Some lines start with garbage hexadecimal characters.
                                                       # Remove these by removing non alphanumeric characters.
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

            # Clean the codes in the code column.
            code = lineChunks[codeColumn]
            termCodePresent = re.search("-[0-9]*$", code)
            emisReqPresent = re.search("^EMISREQ\|", code)
            if termCodePresent:
                # If the code looks something like AAA-##, then strip off the hyphen and the trailing numbers.
                lineChunks[codeColumn] = code[:termCodePresent.start()]
            elif emisReqPresent:
                # If the code looks something like EMISREQ|440.., then strip off the EMISREQ|.
                lineChunks[codeColumn] = code[emisReqPresent.end():]

            # Remove columns.
            lineChunks = [i for ind, i in enumerate(lineChunks) if not ind in colsToRemove]

            # Rejoin the cleaned column entries, and then write them out.
            if all([len(i) > 0 for i in lineChunks]):
                # Only write the line out if there are no blanks entries in it.
                writeCleanData.write('\t'.join(lineChunks) + '\n')