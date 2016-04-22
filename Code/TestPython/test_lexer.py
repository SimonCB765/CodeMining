"""Tests for the class definition lexer."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import ClassLexer
from CodeMiningPython import tokens


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
            [tokens.CodeLiteralToken("CODE1"), tokens.EOFToken()],
            [tokens.CodeLiteralToken("CODE1."), tokens.EOFToken()],
            [tokens.CodeLiteralToken("C1"), tokens.AndToken(), tokens.CodeLiteralToken("c2"), tokens.OrToken(),
                tokens.CodeLiteralToken("c3"), tokens.EOFToken()],
            [tokens.NumLiteralToken("7"), tokens.GreaterThanToken(), tokens.NumLiteralToken("7"), tokens.EOFToken()],
            [tokens.CodeLiteralToken("D"), tokens.LessThanToken(), tokens.NumLiteralToken("5"), tokens.NotToken(),
                tokens.NumLiteralToken("6"), tokens.GreaterThanToken(), tokens.NotToken(),
                tokens.CodeLiteralToken("FF"), tokens.EOFToken()]]

        for i in zip(tests, results):
            lexer = ClassLexer.ClassLexer(i[0])  # Create the lexer.
            lexer.tokenise()  # Tokenise the input.
            self.assertEqual([type(j) for j in lexer.tokenised], [type(j) for j in i[1]])