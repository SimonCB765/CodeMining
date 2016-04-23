"""Tests for the class definition parser."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import class_parser


class CompletionTests(unittest.TestCase):
    """Class to check the successful completion of the parsing."""

    def test_completion(self):
        symbolTable = class_parser.create_class_def_symbol_table()

        print("AAA > #45 & ~ B. < CC | DDDD.")
        class_parser.main("AAA > #45 & ~ B. < CC | DDDD.", symbolTable)
        print("\n\n")
        print("A | B | C")
        class_parser.main("A | B | C", symbolTable)
        print("\n\n")
        print("~A")
        class_parser.main("~A", symbolTable)
        print("\n\n")
        print("~#5")
        class_parser.main("~#5", symbolTable)
        print("\n\n")
        print("A & B | C")
        class_parser.main("A & B | C", symbolTable)
        print("\n\n")
        print("A & (B | C)")
        class_parser.main("A & (B | C)", symbolTable)
        print("\n\n")
        print("(A & B) | C")
        class_parser.main("(A & B) | C", symbolTable)
        print("\n\n")
        print("(A & B | C)")
        class_parser.main("(A & B | C)", symbolTable)