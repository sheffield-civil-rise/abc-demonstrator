"""
This code defines a script which in turn runs pytest on our source code.
"""

# Local imports.
import subprocess

# Local constants.
HANSEL_PATH_TO_REPO = r"G:\photogrammetry_e110a"

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    try:
        subprocess.run(["python", "-m" "pytest", HANSEL_PATH_TO_REPO], check=True)
    except subprocess.CalledProcessError:
        return False
    return True

if __name__ == "__main__":
    run()
