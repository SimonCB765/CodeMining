import re


OPERATORS = {'<' : 'LT_OP', '>' : 'GT_OP', '<=' : 'LTE_OP', '>=' : 'GTE_OP', '!=' : 'NE_OP', '==' : 'EQ_OP', '|' : 'OR_OP', '&' : 'AND_OP', '~' : 'NOT_OP'}


def is_alphanumeric(char):
    return re.match('[0-9a-zA-Z]', char)


def is_operator(char):
    return re.match('[<>!=|&~]', char)


def scan(input):
    tokenised = []  # The tokenised string.
    currentPos = 0  # Current position in the string during scanning.

    while currentPos < len(input):

        if input[currentPos].isspace():
            # The current position is a whitespace character, so skip it and all contiguous whitespace.
            currentPos = scan_whitespace(input, currentPos)
        elif input[currentPos] == '#':
            # The start of a number has been found.
            currentPos += 1
            returnedPos, returnedValue = scan_number(input, currentPos)
            if returnedPos == -1:
                # '#' was found, but the next character was not a digit.
                return tokenised, 'Expected a digit at index {0:d}, got {1:s} instead.'.format(currentPos, returnedValue)
            else:
                currentPos = returnedPos
                tokenised.append('NUM')
        elif is_alphanumeric(input[currentPos]):
            # A code has been found.
            currentPos, returnedValue = scan_code(input, currentPos)
            tokenised.append('CODE')
        elif is_operator(input[currentPos]):
            # The beginning of a comparison operator has been found.
            currentPos, returnedToken = scan_operator(input, currentPos)
            tokenised.append(returnedToken)
        else:
            return tokenised, 'Unexpected character {0:s} at index {1:d}.'.format(input[currentPos], currentPos)

    return tokenised


def scan_code(input, currentPos):
    codeMatch = re.match('[0-9a-zA-Z]+\.{0,1}', input[currentPos:])
    return currentPos + codeMatch.end(0), codeMatch.group(0)


def scan_number(input, currentPos):
    numberMatch = re.match('[0-9]+', input[currentPos:])  # Get all contiguous digits in the number.
    if numberMatch:
        # There were numbers at the beginning of the input section to look at.
        return currentPos + numberMatch.end(0), numberMatch.group(0)
    else:
        # Was expecting a number, but got something else.
        return -1, input[currentPos]


def scan_operator(input, currentPos):
    currentChar = input[currentPos]
    nextChar = input[currentPos]

    if (currentChar + nextChar) in OPERATORS:
        return currentPos + 2, OPERATORS[currentChar + nextChar]
    else:
        return currentPos + 1, OPERATORS[currentChar]


def scan_whitespace(input, currentPos):
    # Return the next spot to start looking at the input. This is the current position plus the point at which
    # the contiguous whitespace ends. So if the string is 123__456 (where _ represents a space), then this function
    # would be called with currentPos == 3, and the match would be performed on the string '__456' so the match would
    # end at index 2, and you would return 3 + 2 = 5 as the position to continue the lexer from.
    whitespaceMatch = re.match('\s', input[currentPos:])  # Contiguous whitespace in input starting at currentPos.
    return currentPos + whitespaceMatch.end(0)