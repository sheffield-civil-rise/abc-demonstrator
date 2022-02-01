"""
This code defines a script which in turn runs the demonstrator script.
"""

# Local imports.
import os
import shutil
import subprocess

# Local constants.
PATH_TO_DEMONSTRATOR_GPS_DATA = "G:\\gps\\210513_113847.csv"
PATH_TO_DEMONSTRATOR_LADY_BUG_INFO = "G:\\ladybug_frame_gps_info_23627.txt"
PATH_TO_DEMONSTRATOR_LADY_BUG_IMAGES = "G:\\ladybug\\"
PATH_TO_DEMONSTRATOR_POLYGON = "G:\\polygons\\demo\\demo_0.poly"
PATH_TO_DEMONSTRATOR_OUTPUT = "G:\\demonstrator_output"

#############
# FUNCTIONS #
#############

def run_demonstrator():
    """ Run the demonstrator script. """
    if os.path.exists(PATH_TO_DEMONSTRATOR_OUTPUT):
        shutil.rmtree(PATH_TO_DEMONSTRATOR_OUTPUT)
    arguments = (
        "python",
        "src\\demonstrator.py"
        PATH_TO_DEMONSTRATOR_GPS_DATA,
        PATH_TO_DEMONSTRATOR_LADY_BUG_INFO,
        PATH_TO_DEMONSTRATOR_LADY_BUG_IMAGES,
        PATH_TO_DEMONSTRATOR_POLYGON,
        "--wd",
        PATH_TO_DEMONSTRATOR_OUTPUT
    )
    try:
        subprocess.run(arguments, check=True)
    except subprocess.CalledProcessError:
        print("Sorry, but the demonstrator failed.")
        return False
    print("Demonstrator ran successfully!")
    return True

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    run_demonstrator()

if __name__ == "__main__":
    run()
