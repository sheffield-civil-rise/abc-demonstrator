"""
This code defines a script which in turn runs pytest on our source code.
"""

# Standard imports.
import os
from pathlib import Path

# Non-standard imports.
import pytest

# Local constants.
DEFAULT_MIN_CODE_COVERAGE = 80

##################
# MAIN FUNCTIONS #
##################

def run_tests(min_code_coverage=DEFAULT_MIN_CODE_COVERAGE):
    """ Run PyTest. """
    os.chdir(Path(__).parent)
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
