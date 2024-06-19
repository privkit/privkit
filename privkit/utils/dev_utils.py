import sys
import warnings


def log(message):
    print(message)


def error(message):
    raise Exception(message)


def warn(message):
    warnings.warn(message)
