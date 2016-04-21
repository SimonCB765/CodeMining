"""Tokens for the class definition tokeniser."""

class NumLiteralToken(object):
    def __init__(self, value):
        self.value = int(value)

class CodeLiteralToken(object):
    def __init__(self, value):
        self.value = value

class AndToken(object):
    lbp = 10

class OrToken(object):
    lbp = 5

class NotToken(object):
    lbp = 15

class LessThanToken(object):
    lbp = 20

class LessThanEqualToken(object):
    lbp = 20

class GreaterThanToken(object):
    lbp = 20

class GreaterThanEqualToken(object):
    lbp = 20

class EqualToken(object):
    lbp = 20

class NotEqualToken(object):
    lbp = 20

class LParenToken(object):
    lbp = 1

class RParenToken(object):
    lbp = 1

class EOFToken(object):
    lbp = 0