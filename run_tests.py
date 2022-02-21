"""
This code defines a script which in turn runs pytest on our source code.
"""

# Non-standard imports.
import pytest

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    pytest.main()

if __name__ == "__main__":
    run()
