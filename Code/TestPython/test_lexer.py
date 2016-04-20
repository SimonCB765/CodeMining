"""Tests for the class definition lexer."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import ClassLexer


class SuccessTests(unittest.TestCase):
    """Tests where the lexer should perform a successful tokenisation."""

    def test_simple_defs(self):
        tests = [
            "CODE1",
            "CODE1.",
            "C1 & c2 | c3",
            "#3 > #7",
            "D. < #5 ~ #6 > ~ FF"]
        results = [
            [("CODE", "CODE1"), ('EOF', 'EOF')],
            [("CODE", "CODE1."), ('EOF', 'EOF')],
            [("CODE", "C1"), ("AND_OP", "&"), ("CODE", "c2"), ("OR_OP", "|"), ("CODE", "c3"), ('EOF', 'EOF')],
            [("NUM", '3'), ("COMP_OP", ">"), ("NUM", '7'), ('EOF', 'EOF')],
            [("CODE", "D."), ("COMP_OP", "<"), ("NUM", '5'), ("NOT_OP", "~"), ("NUM", '6'), ("COMP_OP", ">"),
                ("NOT_OP", "~"), ("CODE", "FF"), ('EOF', 'EOF')]]

        for i in zip(tests, results):
            lexer = ClassLexer.ClassLexer(i[0])  # Create the lexer.
            lexer.tokenise()  # Tokenise the input.
            self.assertEqual(lexer.tokenised, i[1])