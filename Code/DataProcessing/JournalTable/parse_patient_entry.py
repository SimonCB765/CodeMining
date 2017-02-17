"""Function to parse a patient entry in the SQL insert style file."""


def main(line):
    """Parse an entry in the file of the patient-code associations.

    :param line:    The line in the file to parse.
    :type line:     str
    :return:        The entries on the line. The entries will be ordered (in ascending index order) as:
                        patient id, code, date, Val1, Val2, free text
    :rtype:         list

    """

    # The beginning of the line is expected to have the format:
    #   insert into `journal`(`id`,`code`,`date`,`value1`,`value2`,`text`) values (
    # The rest of the line will be formatted as:
    #   id,code,date,Val1,Val2,FreeText);
    # The first 75 and last 3 character are therefore stripped off to give: id,code,date,Val1,Val2,FreeText
    # The expected formats of these entries are:
    #   id - numerical patient ID value (e.g. 26044).
    #   code - clinical code as a single quoted string (e.g. 'C10E').
    #   date - the date of the association as a single quoted string (e.g. '1998-04-16').
    #   Val1 - the first float value (format .4f) associated with the code-patient pair (e.g. 120.0000).
    #   Val2 - the second float value (format .4f) associated with the code-patient pair (e.g. 45.0000).
    #   FreeText - null if there is no text associated with the code-patient pair or a single quoted string
    #       (e.g. 'ONE TO BE TAKEN FOUR TIMES A DAY').
    line = line[75:]  # Strip of the SQL insert syntax at the beginning.
    line = line[:-3]  # Strip off the ");\n" at the end.

    # Some codes are recorded with embedded data (e.g. as '2469,v=130,w=80'). In this case the code has its
    # two values recorded as part of the code (often with the two values also recorded in the correct Val1
    # and Val2 entries as well. It's also possible that the free text has commas in it (which is used as the
    # delimiter in the insert statement). Simply splitting the insert statement on a comma to get the
    # different values is therefore not feasible. Instead, the line has to be read character by character
    # to make sure that the line parsing is done correctly.
    # The data values within the code are ignored, and the Val1 and Val2 values recorded in the correct
    # place used instead.
    # Any values that are treated in a European manner with a comma in place of the decimal point will
    # cause the parsing to fail, unless they are quoted.
    entries = []
    currentEntry = ""
    inQuoteBlock = False
    for i in line:
        if i == ',' and not inQuoteBlock:
            # Found a separator and we aren't in a quote block. Therefore, record the end of the current
            # entry and initialise for the next entry.
            entries.append(currentEntry)
            currentEntry = ""
        elif i in ["'", '"']:
            # Either found the end or the start of a quote block.
            inQuoteBlock = not inQuoteBlock
        else:
            # Either current character is not a comma or we are currently in a quote block as are
            # ignoring commas.
            currentEntry += i
    entries.append(currentEntry)  # Add the final entry to the list of entries.

    # Update the code entry.
    code = entries[1].split(',')[0]  # If the code is recorded with its values, then just get the code.
    entries[1] = code

    return entries
