"""Simple lexer to tokenise a class definition."""

# Python imports.
import re


class ClassLexer(object):
    """Class to perform the tokenisation of an input string."""

    operators = {
                '<' : 'COMP_OP', '>' : 'COMP_OP', '<=' : 'COMP_OP', '>=' : 'COMP_OP', '!=' : 'COMP_OP', '==' : 'COMP_OP',
                '|' : 'OR_OP', '&' : 'AND_OP', '~' : 'NOT_OP',
                '(' : 'L_PAREN', ')' : 'R_PAREN'
                }

    def __init__(self, classString):
        self.classString = classString  # The string to tokenise.
        self.tokenised = []  # Tokenisation of the classString.

    @staticmethod
    def is_alphanumeric(char):
        return re.match('[0-9a-zA-Z]', char)

    @staticmethod
    def is_operator(char):
        return re.match('[><!=|&~)(]', char)

    def reset(self, classString):
        self.classString = classString
        self.tokenised = []

    def scan_code(self, currentPos):
        codeMatch = re.match('[0-9a-zA-Z]+\.{0,1}', self.classString[currentPos:])
        return currentPos + codeMatch.end(0), codeMatch.group(0)

    def scan_number(self, currentPos):
        numberMatch = re.match('[0-9]+', self.classString[currentPos:])  # Get all contiguous digits in the number.
        if numberMatch:
            # There were numbers at the beginning of the input section to look at.
            return currentPos + numberMatch.end(0), numberMatch.group(0)
        else:
            # Was expecting a number, but got something else.
            return -1, self.classString[currentPos]

    def scan_operator(self, currentPos):
        currentChar = self.classString[currentPos]
        nextChar = self.classString[currentPos]

        if (currentChar + nextChar) in self.operators:
            return currentPos + 2, currentChar + nextChar
        else:
            return currentPos + 1, currentChar

    def tokenise(self):
        currentPos = 0  # Current position in the string during scanning.

        while currentPos < len(self.classString):

            if self.classString[currentPos].isspace():
                # The current position is a whitespace character, so skip it.
                currentPos += 1
            elif self.classString[currentPos] == '#':
                # The start of a number has been found.
                currentPos += 1
                returnedPos, returnedValue = self.scan_number(currentPos)
                if returnedPos == -1:
                    # '#' was found, but the next character was not a digit.
                    return self.tokenised, 'Expected a digit at index {0:d}, got {1:s} instead.'\
                        .format(currentPos, returnedValue)
                else:
                    currentPos = returnedPos
                    self.tokenised.append(('NUM', int(returnedValue)))
            elif self.is_alphanumeric(self.classString[currentPos]):
                # A code has been found.
                currentPos, returnedValue = self.scan_code(currentPos)
                self.tokenised.append(('CODE', returnedValue))
            elif self.is_operator(self.classString[currentPos]):
                # The beginning of a comparison operator has been found.
                currentPos, returnedValue = self.scan_operator(currentPos)
                self.tokenised.append((self.operators[returnedValue], returnedValue))
            else:
                return self.tokenised, 'Unexpected character {0:s} at index {1:d}.'\
                    .format(self.classString[currentPos], currentPos)