"""
This code defines a script which checks with Hansel whether the demonstrator
is running correctly.
"""

# Standard imports.
import os
import subprocess

# Local constants.
HANSEL_SSH_ID = "hansel@172.16.67.13"
HANSEL_PATH_TO_BAT = r"G:\photogrammetry_e110a\check_with_hansel.bat"

#############
# FUNCTIONS #
#############

def run_on_hansel(arguments):
    """ Run a given command on Hansel. """
    arguments = ["ssh", HANSEL_SSH_ID]+arguments
    try:
        subprocess.run(arguments, check=True)
    except subprocess.CalledProcessError:
        print("Error running arguments:")
        print(arguments)
        return False
    return True

def run_bat():
    """ Run check_with_hansel.bat over SSH. """
    if not run_on_hansel([HANSEL_PATH_TO_BAT]):
        print("Sorry. That check has failed.")
        return False
    print("Check passed!")
    return True

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    run_bat()

if __name__ == "__main__":
    run()
