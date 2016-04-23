"""Tests for the class definition lexer."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import class_parser


class SuccessTests(unittest.TestCase):
    """Tests where the lexer should perform a successful tokenisation."""

    def test_simple_defs(self):
        symbolTable = class_parser.create_class_def_symbol_table()
        tests = [
            "CODE1",
            "CODE1.",
            "C1 & c2 | c3",
            "#3 > #7",
            "D. < #5 ~ #6 > ~ FF"]
        expectedResults = [
            [("CODE", "CODE1"), ("EOF", "EOF")],
            [("CODE", "CODE1."), ("EOF", "EOF")],
            [("CODE", "C1"), ('&', '&'), ("CODE", "c2"), ('|', '|'), ("CODE", "c3"), ("EOF", "EOF")],
            [("NUM", '3'), ('>', '>'), ("NUM", '7'), ("EOF", "EOF")],
            [("CODE", "D."), ('<', '<'), ("NUM", '5'), ('~', '~'), ("NUM", '6'), ('>', '>'), ('~', '~'),
                ("CODE", "FF"), ("EOF", "EOF")]
        ]

        for i in zip(tests, expectedResults):
            testResult = [(j.tokenType, j.value) for j in class_parser.ClassParser._tokeniser(i[0], symbolTable)]
            self.assertEqual(testResult, i[1])