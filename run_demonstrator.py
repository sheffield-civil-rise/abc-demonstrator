"""
This code defines a script which in turn runs the demonstrator script.
"""

# Local imports.
import subprocess

# Local constants.
HANSEL_PATH_TO_DEMONSTRATOR_SCRIPT = \
    r"G:\photogrammetry_e110a\src\demonstrator.py"

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    try:
        subprocess.run(
            ["python", HANSEL_PATH_TO_DEMONSTRATOR_SCRIPT], check=True
        )
    except subprocess.CalledProcessError:
        print("Sorry, but the demonstrator failed.")
        return False
    print("Demonstrator ran successfully!")
    return True

if __name__ == "__main__":
    run()
