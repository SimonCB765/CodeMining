"""Recursive descent parser for a class definition string.

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

# User imports.
from . import ClassLexer


class ClassParser(object):
    """Class to perform the class string parsing."""

    def __init__(self, classString):
        self.lexer = ClassLexer.ClassLexer(classString)  # The lexer to tokenise the input.
        self.parseTree = None  # The generated parse tree.

    def parse(self):
        self.currentTokenIndex = 0
        self.lexer.tokenise()
        print(self.lexer.tokenised)
        self.program()

    def reset(self, classString):
        self.lexer = ClassLexer.ClassLexer(classString)
        self.parseTree = None

    #===================#
    # GRAMMAR FUNCTIONS #
    #===================#
    def program(self):
        currentNode = self.NonTermParseNode("PROGRAM")
        self.parseTree = currentNode
        self.or_term(currentNode)
        self.parseTree.print()

    def or_term(self, parentNode):
        currentNode = self.NonTermParseNode("OR_TERM")
        parentNode.add_child(currentNode)

        self.and_term(currentNode)
        nextToken = self.lexer.tokenised[self.currentTokenIndex]
        while nextToken[0] == 'OR_OP':
            literalChild = self.TerminalParseNode(nextToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            self.and_term(currentNode)
            nextToken = self.lexer.tokenised[self.currentTokenIndex]

    def and_term(self, parentNode):
        currentNode = self.NonTermParseNode("AND_TERM")
        parentNode.add_child(currentNode)

        self.not_term(currentNode)
        nextToken = self.lexer.tokenised[self.currentTokenIndex]
        while nextToken[0] == 'AND_OP':
            literalChild = self.TerminalParseNode(nextToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            self.not_term(currentNode)
            nextToken = self.lexer.tokenised[self.currentTokenIndex]

    def not_term(self, parentNode):
        currentNode = self.NonTermParseNode("NOT_TERM")
        parentNode.add_child(currentNode)

        currentToken = self.lexer.tokenised[self.currentTokenIndex]
        if currentToken[0] == 'NOT_OP':
            # Found the start of a negated term.
            literalChild = self.TerminalParseNode(currentToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            self.not_term(currentNode)
        else:
            self.comp_term(currentNode)

    def comp_term(self, parentNode):
        currentNode = self.NonTermParseNode("COMP_TERM")
        parentNode.add_child(currentNode)

        self.left_expr(currentNode)
        nextToken = self.lexer.tokenised[self.currentTokenIndex]
        while nextToken[0] == 'COMP_OP':
            literalChild = self.TerminalParseNode(nextToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            self.right_expr(currentNode)
            nextToken = self.lexer.tokenised[self.currentTokenIndex]

    def left_expr(self, parentNode):
        currentNode = self.NonTermParseNode("LEFT_EXPR")
        parentNode.add_child(currentNode)

        currentToken = self.lexer.tokenised[self.currentTokenIndex]
        if currentToken[0] == 'CODE':
            # The left expression is a code.
            literalChild = self.TerminalParseNode(currentToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            # End recursion.
        elif currentToken[0] == 'L_PAREN':
            literalChild = self.TerminalParseNode(currentToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            self.or_term(currentNode)
            nextToken = self.lexer.tokenised[self.currentTokenIndex]
            if nextToken[0] == 'R_PAREN':
                # Successfully closed the parentheses.
                literalChild = self.TerminalParseNode(currentToken[1])
                currentNode.add_child(literalChild)
                self.currentTokenIndex += 1
                # End recursion.
            else:
                raise SyntaxError("Error parsing class definition. Expected ')', but received {0:s}."
                                  .format(nextToken[1]))
        else:
            # Error.
            raise SyntaxError("Error parsing class definition. Expected a code or '(', but received {0:s}."
                              .format(currentToken[1]))

    def right_expr(self, parentNode):
        currentNode = self.NonTermParseNode("RIGHT_EXPR")
        parentNode.add_child(currentNode)

        currentToken = self.lexer.tokenised[self.currentTokenIndex]
        if currentToken[0] == 'NUM':
            # The right expression is a numerical value.
            literalChild = self.TerminalParseNode(currentToken[1])
            currentNode.add_child(literalChild)
            self.currentTokenIndex += 1
            # End recursion.
        else:
            # Must be a left expression.
            self.left_expr(currentNode)



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
