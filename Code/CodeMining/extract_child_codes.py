import re

def main(parentCodes, allCodes):
    """Extract all codes that are beneath the parent codes in the code hierarchy.

    A code (b) is 'beneath' another one (a) if b[:len(a)] == a.
    Example:
        parentCodes = ["ABC", "XYZ"]
        allCodes = ["ABCD", "ABC12", "1ABC", "AB", "XYZ", "XYZ01", "DEF12"]
        return = ["ABCD", "ABC12", "XYZ", "XYZ01"]

    :param parentCodes:     The codes at the root of the hierarchy substree(s) to be extracted.
    :type parentCodes:      list
    :param allCodes:        The codes to search for children in. Each code should appear once.
    :type allCodes:         list
    :returns :              The parent codes and their child codes.
    :type :                 list

    """

    regex = re.compile('|'.join(parentCodes))  # Compiled regular expression pattern code1|code2|code3|...|codeN.
    return [i for i in allCodes if regex.match(i)]