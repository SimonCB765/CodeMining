"""Tests for the class definition parser."""

# Python imports.
import unittest

# User imports.
from CodeMiningPython import ClassParser


class CompletionTests(unittest.TestCase):
    """Class to check the successful completion of the parsing."""

    def test_completion(self):
        parser = ClassParser.ClassParser("AAA > #45 & ~ B. < CC | DDDD.")
        parser.parse()
        parser.reset('A | B | C')
        parser.parse()