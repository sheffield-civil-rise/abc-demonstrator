"""
This code defines a script which in turn runs pytest on our source code.
"""

# Standard imports.
import os
import subprocess

# Non-standard imports.
import pytest

# Local constants.
HANSEL_PATH_TO_REPO = r"G:\photogrammetry_e110a"

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    print(os.cwd())
    pytest.main(["tests"])
#    try:
#        subprocess.run(["pytest", HANSEL_PATH_TO_REPO], check=True)
#    except subprocess.CalledProcessError:
#        return False
#    return True

if __name__ == "__main__":
    run()
