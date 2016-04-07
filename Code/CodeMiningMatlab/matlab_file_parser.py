def dataset_parser(fileData, delimiter='\t'):
    """Parses dataset files formatted for the code mining.

    The input file is expected to be formatted in three columns:
        1st - Patient ID
        2nd - Code
        3rd - Number of occurrences of the code within the patient's record.
    Each line forms a tuple recording the number of times a given code occurs in a given patient's record.

    :param fileData:    The location of the file containing the data to parse.
    :type fileData:     str
    :param delimiter:   The delimiter used to divide the columns in the dataset file.
    :type delimiter:    str
    :returns :          The lists of the patient IDs, codes and counts.
    :rtype :            dict

    """

    patientIDs = []
    codes = []
    counts = []
    with open(fileData, 'r') as fidData:
        for line in fidData:
            chunks = (line.strip()).split(delimiter)

            # Some entries have blank codes. Reject these.
            if (chunks[1] == '') or chunks[1].isspace():
                continue

            patientIDs.append(int(chunks[0]))
            codes.append(chunks[1])
            counts.append(int(chunks[2].replace(',', '')))  # Some counts are over 1,000 and have commas in them.
    return {"IDs" : patientIDs, "Codes" : codes, "Counts" : counts}


def codelist_parser(fileCodes, delimiter='\t'):
    """Parses code list mappings.

    The input file is expected to be formatted in two columns:
        1st - Code
        2nd - The description of the code.

    :param fileCodes:   The location of the file containing the code mapping to parse.
    :type fileCodes:    str
    :param delimiter:   The delimiter used to divide the columns in the code mapping file.
    :type delimiter:    str
    :returns :          The lists of the codes and descriptions.
    :rtype :            dict

    """

    codes = []
    descriptions = []
    with open(fileCodes, 'r') as fidCodes:
        for line in fidCodes:
            chunks = (line.strip()).split(delimiter)
            codes.append(chunks[0])
            descriptions.append(chunks[1])
    return {"Codes" : codes, "Descriptions" : descriptions}