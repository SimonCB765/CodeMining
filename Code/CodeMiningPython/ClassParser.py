"""Recursive descent parser for a class definition string.

The grammar that this parser parsers is:

PROGRAM     ::= OR_TERM
OR_TERM     ::= AND_TERM ('|' AND_TERM)*
AND_TERM    ::= NOT_TERM ('&' NOT_TERM)*
NOT_TERM    ::= '!' NOT_TERM |
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

# User imports.
from . import ClassLexer


class ClassParser(object):
    """Class to perform the class string parsing."""

    def __init__(self, classString):
        self.lexer = ClassLexer.ClassLexer(classString)  # The lexer to tokenise the input.
        self.parseTree = None  # The generated parse tree.

    def parse(self):
        self.lexer.tokenise()
        self.program(0)

    def reset(self, classString):
        self.lexer = ClassLexer.ClassLexer(classString)

    #===================#
    # GRAMMAR FUNCTIONS #
    #===================#
    def program(self, currentTokenIndex):
        currentNode = self.NonTermParseNode("PROGRAM")
        self.parseTree = currentNode
        self.or_term(currentNode, currentTokenIndex)
        self.parseTree.print()

    def or_term(self, parentNode, currentTokenIndex):
        currentNode = self.NonTermParseNode("OR_TERM")
        parentNode.add_child(currentNode)

        self.and_term(currentNode, currentTokenIndex)
        nextToken = self.lexer.tokenised[currentTokenIndex]
        while nextToken[0] == 'OR_OP':
            literalChild = self.TerminalParseNode(nextToken[1])
            parentNode.add_child(literalChild)
            currentTokenIndex += 1
            self.and_term(currentNode, currentTokenIndex)

    def and_term(self, parentNode, currentTokenIndex):
        currentNode = self.NonTermParseNode("AND_TERM")
        parentNode.add_child(currentNode)


    #========================#
    # PARSE TREE DEFINITIONS #
    #========================#
    class ParseNode(object):
        """Abstract base class for a parse tree node."""

        def print(self, offset=''):
            pass

    class NonTermParseNode(ParseNode):
        """Nonterminal node in a parse tree."""

        def __init__(self, rule):
            self.rule = rule
            self.children = []  # Will be ordered from left to right.

        def add_child(self, childNode):
            self.children.append(childNode)

        def print(self, offset=''):
            print("{0:s}{1:s}\n".format(offset, self.rule))
            offset += "  "
            for i in self.children:
                i.print(offset)

    class TerminalParseNode(ParseNode):
        """Terminal node in a parse tree."""

        def __init__(self, literal):
            self.literal = literal

        def print(self, offset):
            print("{0:s}{1:s}".format(offset, self.literal))
