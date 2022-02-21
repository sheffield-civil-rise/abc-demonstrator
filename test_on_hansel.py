"""
This code defines a script which runs this repo's automated tests on Hansel.
"""

# Standard imports.
import os
import subprocess
import sys
from pathlib import Path

# Local imports.
from check_with_hansel import run_on_hansel_with_auth, DEFAULT_PATH_TO_REPO

# Constants.
PATH_TO_SCRIPT = os.path.join(DEFAULT_PATH_TO_REPO, "test_on_hansel.sh")

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    result = run_on_hansel_with_auth(path_to_script=PATH_TO_SCRIPT)
    return result

if __name__ == "__main__":
    run()