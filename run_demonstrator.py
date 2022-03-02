"""
This code defines a script which in turn runs the demonstrator script.
"""

# Local imports.
import pathlib
import subprocess
import sys

# Local constants.
PYTHON_COMMAND = "python"
PATH_TO_DEMONSTRATOR_SCRIPT = \
    str(pathlib.Path(__file__).parent/"src"/"demonstrator.py")

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    try:
        subprocess.run(
            [PYTHON_COMMAND, PATH_TO_DEMONSTRATOR_SCRIPT], check=True
        )
    except subprocess.CalledProcessError:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    run()
