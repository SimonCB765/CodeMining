"""Tests for the class definition lexer."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import ClassLexer


class SuccessTests(unittest.TestCase):
    """Tests where the lexer should perform a successful tokenisation."""

    def test_simple_defs(self):
        tests = [
            'CODE1',
            'CODE1.',
            'C1 & c2 | c3',
            '#3 > #7',
            'D. < #5 ~ #6 > ~ FF']
        results = [
            ['CODE'],
            ['CODE'],
            ['CODE', 'AND_OP', 'CODE', 'OR_OP', 'CODE'],
            ['NUM', 'GT_OP', 'NUM'],
            ['CODE', 'LT_OP', 'NUM', 'NOT_OP', 'NUM', 'GT_OP', 'NOT_OP', 'CODE']]

        for i in zip(tests, results):
            lexer = ClassLexer.ClassLexer(i[0])  # Create the lexer.
            lexer.tokenise()  # Tokenise the input.
            self.assertEqual(lexer.tokenised, i[1])