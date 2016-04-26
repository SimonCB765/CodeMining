"""Generates the mapping from codes to their descriptions."""


def main(fileCodes, delimiter='\t'):
    """Generates code list mappings.

    Each line in the input file is expected to have 2 columns:
        1st - Code
        2nd - The description of the code.

    :param fileCodes:   The location of the file containing the code mapping to parse.
    :type fileCodes:    str
    :param delimiter:   The delimiter used to divide the columns in the code mapping file.
    :type delimiter:    str
    :returns :          The mapping from codes to descriptions.
    :rtype :            dict

    """

    codeMapping = {}
    with open(fileCodes, 'r') as fidCodes:
        for line in fidCodes:
            chunks = (line.strip()).split(delimiter)
            codeMapping[chunks[0]] = chunks[1]
    return codeMapping