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

def push_and_validate(message=DEFAULT_MESSAGE):
    """ Push and run the continuous integration script. """
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", message])
    subprocess.run(["git", "push", "origin", get_current_branch()])
    try:
        subprocess.run(["python3", "continuous_integration.py"], check=True)
    except subprocess.CalledProcessError:
        return False
    return True

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    push_and_validate()

if __name__ == "__main__":
    run()
