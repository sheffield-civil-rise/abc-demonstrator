"""
This code defines a script which in turn runs pytest on our source code.
"""

# Local imports.
import subprocess

# Local constants.
HANSEL_PATH_TO_SRC = r"G:\photogrammetry_e110a\src"

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    try:
        subprocess.run(["pip", "install", "pytest"], check=True)
        subprocess.run(["pytest", HANSEL_PATH_TO_SRC], check=True)
    except subprocess.CalledProcessError:
        return False
    return True

if __name__ == "__main__":
    run()
