"""TDOP parser for a class definition string.

See http://effbot.org/zone/simple-top-down-parsing.htm for more details on TDOP parsers.

The grammar that this parser parsers is:

PROGRAM     ::= OR_TERM
OR_TERM     ::= AND_TERM ('|' AND_TERM)*
AND_TERM    ::= NOT_TERM ('&' NOT_TERM)*
NOT_TERM    ::= '~' NOT_TERM |
                COMP_TERM
COMP_TERM   ::= L_EXPR (COMP_OP R_EXPR)*
R_EXPR      ::= L_EXPR |
                NUM
L_EXPR      ::= '(' OR_TERM ')' |
                CODE
COMP_OP     ::= '<' | '>' | '<=' | '>=' | '==' | '!='

CODE        ::= CODE_PROPER CHILD_FLAG
CODE_PROPER ::= DIGIT CODE_PROPER |
                CHAR CODE_PROPER
CHILD_FLAG  ::= '.' | ''

NUM         ::= '#' NUM_PROPER
NUM_PROPER  ::= DIGIT NUM_PROPER | DIGIT

CHAR        ::= a-zA-Z
DIGIT       ::= 0-9

"""

# Python imports.
import re


def main(stringToParse, symbolTable):
    ClassParser.parse(stringToParse, symbolTable)

def create_class_def_symbol_table():
    # Create the symbol table.
    symbolTable = {}
    token_factory(symbolTable, "EOF", 0)  # Add the end of file token.
    literal_token_factory(symbolTable, "CODE")  # Add the code token.
    literal_token_factory(symbolTable, "NUM")  # Add the number token.

    infix_token_factory(symbolTable, '&', 10)  # Add the logical and token.
    infix_token_factory(symbolTable, '|', 5)  # Add the logical or token.
    prefix_token_factory(symbolTable, '~', 15)  # Add the logical not token.

    infix_token_factory(symbolTable, "==", 20)  # Add the equals token.
    infix_token_factory(symbolTable, '>', 20)  # Add the greater than token.
    infix_token_factory(symbolTable, ">=", 20)  # Add the greater than or equal token.
    infix_token_factory(symbolTable, '<', 20)  # Add the less than token.
    infix_token_factory(symbolTable, "<=", 20)  # Add the less than or equal token.
    infix_token_factory(symbolTable, "!=", 20)  # Add the not equal token.

    container_token_factory(symbolTable, '(', ')', 1)  # Add the left parenthesis token.
    token_factory(symbolTable, ')', 1)  # Add the not parenthesis token.

    return symbolTable

#==================================#
# Token Creation Functions/Classes #
#==================================#
class BaseToken(object):
    """Class containing the basic functionality of a token."""

    tokenType = None
    value = None  # Value of the token. Used by literal tokens.
    firstChild = secondChild = None  # Children of a node in the parse tree. Only used for non-literal tokens.
    lbp = 0  # The left binding power of the token.

    def nud(self):
        raise SyntaxError("Syntax error '{0:s}'.".format(self.tokenType))

    def led(self, left):
        raise SyntaxError("Unknown operator '{0:s}'.".format(self.tokenType))

    def pretty_print(self, indent=0, indentType="  "):
        if self.tokenType in ["CODE", "NUM"]:
            return "{0:s}<< {1:s} {2:s} >>".format(indentType * indent, self.tokenType, self.value)
        else:
            output = "{0:s}<< OP {1:s} >>".format(indentType * indent, self.tokenType)
            output += '\n'
            output += self.firstChild.pretty_print(indent=indent + 1)
            if self.secondChild:
                output +='\n'
                output += self.secondChild.pretty_print(indent=indent + 1)
            return output

    def __repr__(self):
        return self.pretty_print()

def container_token_factory(symbolTable, tokenType, advanceTo, bindingPower=0):
    def nud(self):
        expr = ClassParser.expression(bindingPower)
        ClassParser.advance(advanceTo)
        return expr
    token_factory(symbolTable, tokenType).nud = nud

def infix_token_factory(symbolTable, tokenType, bindingPower=0):
    def led(self, left):
        self.firstChild = left
        self.secondChild = ClassParser.expression(bindingPower)
        return self
    token_factory(symbolTable, tokenType, bindingPower).led = led

def literal_token_factory(symbolTable, tokenType):
    token_factory(symbolTable, tokenType).nud = lambda self: self

def token_factory(symbolTable, tokenType, bindingPower=0):
    """Function to create or update a token.

    If the token type already exists in the symbol table, then the binding power of the token is updated
    to be the max of its current binding power and the bindingPower argument.

    :param symbolTable:
    :type symbolTable:
    :param tokenType:       The type name of the token to be created/updated.
    :type tokenType:        str
    :param bindingPower:    The binding power of the token to create, or the new binding power of an existing token.
    :type bindingPower:     int
    :return :
    :rtype :

    """

    token = symbolTable.get(tokenType, None)  # Get the token class from the symbol table if it already exists.
    if token:
        # If the token class already exists in the symbol table, then update its binding power.
        token.lbp = bindingPower
    else:
        # Create a new token class.
        class token(BaseToken):
            pass
        token.__name__ = "Token-{0:s}".format(tokenType)  # Set the token class name for debugging purposes.
        token.tokenType = tokenType
        token.lbp = bindingPower
        symbolTable[tokenType] = token
    return token

def prefix_token_factory(symbolTable, tokenType, bindingPower=0):
    def nud(self):
        self.firstChild = ClassParser.expression(bindingPower)
        self.secondChild = None
        return self
    token_factory(symbolTable, tokenType).nud = nud

#==============#
# Parser Class #
#==============#
class ClassParser(object):
    """Class to parse a string using a predefined symbol table."""

    # Define the token generator needed for the parsing.
    tokenGenerator = None

    # Define the current token being consumed.
    currentToken = None

    @classmethod
    def advance(cls, tokenType=None):
        """Checks that the current token has the expected token type, and then advances to the next token."""
        if tokenType and cls.currentToken.tokenType != tokenType:
            raise SyntaxError("Expected '{0:s}', but got '{1:s}'.".format(tokenType, cls.currentToken.tokenType))
        cls.currentToken = next(cls.tokenGenerator)

    @classmethod
    def expression(cls, rightBindingPower=0):
        previousToken = cls.currentToken
        cls.currentToken = next(cls.tokenGenerator)
        left = previousToken.nud()
        while rightBindingPower < cls.currentToken.lbp:
            previousToken = cls.currentToken
            cls.currentToken = next(cls.tokenGenerator)
            left = previousToken.led(left)
        return left

    @classmethod
    def parse(cls, stringToParse, symbolTable):
        cls.tokenGenerator = cls._tokeniser(stringToParse, symbolTable)
        cls.currentToken = next(cls.tokenGenerator)
        parseTree = cls.expression()
        print(parseTree)

    @classmethod
    def _tokeniser(cls, stringToParse, symbolTable):
        for tokenType, value in cls._tokenise_character_stream(stringToParse, symbolTable):
            if tokenType in ["CODE", "NUM"]:
                # The token should be a literal.
                token = symbolTable[tokenType]()
                token.value = value
            else:
                # The token is an operator of some sort.
                token = symbolTable[tokenType]()
                token.value = value
            yield token

    @classmethod
    def _tokenise_character_stream(cls, stringToParse, symbolTable):
        currentPos = 0  # Current position in the string during scanning.

        while currentPos < len(stringToParse):

            if stringToParse[currentPos].isspace():
                # The current position is a whitespace character, so skip it.
                currentPos += 1
            elif stringToParse[currentPos] == '#':
                # The start of a number has been found.
                currentPos += 1
                numberMatch = re.match('[0-9]+', stringToParse[currentPos:])  # Get all contiguous digits.
                if numberMatch:
                    # Correctly found a number following the '#'.
                    currentPos += numberMatch.end(0)
                    yield "NUM", numberMatch.group(0)
                else:
                    # '#' was found, but the next character was not a digit.
                    raise SyntaxError("Expected a digit at index {0:d}, got {1:s} instead."
                                      .format(currentPos, repr(stringToParse[currentPos])))
            elif re.match('[0-9a-zA-Z]', stringToParse[currentPos]):
                # The start of a code has been found.
                codeMatch = re.match('[0-9a-zA-Z]+\.?', stringToParse[currentPos:])  # Find the code.
                currentPos += codeMatch.end(0)
                yield "CODE", codeMatch.group(0)
            elif re.match('[><!=|&~)(]', stringToParse[currentPos]):
                # The beginning of a comparison operator has been found.
                currentChar = stringToParse[currentPos]
                nextChar = stringToParse[currentPos]
                if (currentChar + nextChar) in symbolTable:
                    currentPos += 2
                    yield currentChar + nextChar, currentChar + nextChar
                else:
                    currentPos += 1
                    yield currentChar, currentChar
            else:
                raise SyntaxError("Unexpected character {0:s} at index {1:d}."
                                  .format(repr(stringToParse[currentPos]), currentPos))

        yield "EOF", "EOF"

