"""
This code defines a script which in turn runs the demonstrator script.
"""

# Local imports.
import os
import shutil
import subprocess

# Local constants.
#PATH_TO_DEMONSTRATOR_GPS_DATA = r"G:\photogrammetry_input\210513_113847.csv"
#PATH_TO_DEMONSTRATOR_LADY_BUG_INFO = r"G:\photogrammetry_input\ladybug_frame_gps_info_23627.txt"
#PATH_TO_DEMONSTRATOR_LADY_BUG_IMAGES = r"G:\photogrammetry_input\ladybug_images"
#PATH_TO_DEMONSTRATOR_POLYGON = r"G:\polygons\demo\demo_0.poly"
#PATH_TO_DEMONSTRATOR_OUTPUT = r"G:\demonstrator_output"
PATH_TO_DEMONSTRATOR_SCRIPT = r"G:\photogrammetry_e110a\src\demonstrator_new.py"

#############
# FUNCTIONS #
#############

def run_demonstrator():
    """ Run the demonstrator script. """
#    if os.path.exists(PATH_TO_DEMONSTRATOR_OUTPUT):
#        shutil.rmtree(PATH_TO_DEMONSTRATOR_OUTPUT)
    arguments = (
        "python",
        PATH_TO_DEMONSTRATOR_SCRIPT,
        #PATH_TO_DEMONSTRATOR_GPS_DATA,
        #PATH_TO_DEMONSTRATOR_LADY_BUG_INFO,
        #PATH_TO_DEMONSTRATOR_LADY_BUG_IMAGES,
        #PATH_TO_DEMONSTRATOR_POLYGON,
        #"--wd",
        #PATH_TO_DEMONSTRATOR_OUTPUT
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
