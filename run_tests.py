"""
This code defines a script which in turn runs pytest on our source code.
"""

# Non-standard imports.
import pytest

# Local constants.
HANSEL_PATH_TO_REPO = r"G:\photogrammetry_e110a"

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    pytest.main()

if __name__ == "__main__":
    run()
