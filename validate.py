"""
This code defines a script which in turn runs pytest on our source code.
"""

# Non-standard imports.
import pytest

# Local constants.
DEFAULT_MIN_CODE_COVERAGE = 50
DEFAULT_MIN_LINT_SCORE = 8

#############
# FUNCTIONS #
#############

def run_tests(min_code_coverage=DEFAULT_MIN_CODE_COVERAGE):
    """ Run PyTest. """
    arguments = [
        "--cov-report", "term",
        "--cov-fail-under="+str(min_code_coverage),
        "--cov=./src"
    ]
    return_code = pytest.main(arguments)
    assert return_code == 0, "PyTest returned code: "+str(return_code)

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    run_tests()

if __name__ == "__main__":
    run()
