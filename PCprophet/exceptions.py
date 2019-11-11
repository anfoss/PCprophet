import os
import sys


class NaRowError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "NaRow error {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class MissingColumnError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "MissingColumnError {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class DuplicateRowError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "DuplicateRowError {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class EmptyColumnError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "EmptyColumnError {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class DuplicateIdentifierError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "DuplicateIdError {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class NaInMatrixError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "DuplicateIdError {0}".format(dErrArguments))
        self.dErrorArguments = dErrorArguements


class NotImplementedError(Exception):
    def __init___(self, dErrorArguments):
        Exception.__init__(self, "Not implemented error")
