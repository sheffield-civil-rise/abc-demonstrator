"""
This code defines a script which quickly pushes any changes and then runs the
continuous integration script.
"""

# Standard imports.
import subprocess

# Local imports.
from run_on_hansel import get_current_branch

# Local constants.
DEFAULT_MESSAGE = "Debugging..."

#############
# FUNCTIONS #
#############

def quick_run(arguments, check=True):
    """ Run some arguments, without any frills. """
    subprocess.run(arguments, check=check)

def push_and_validate(message=DEFAULT_MESSAGE):
    """ Push and run the continuous integration script. """
    quick_run(["git", "add", "."])
    quick_run(["git", "commit", "-m", message])
    quick_run(["git", "push", "origin", get_current_branch()])
    quick_run(["python3", "continuous_integration.py"])

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    push_and_validate()

if __name__ == "__main__":
    run()
