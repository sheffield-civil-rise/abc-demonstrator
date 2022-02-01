"""
The code in this file holds some configurations for the proximate scripts.
"""

# Standard imports.
import os
from pathlib import Path

###########
# CONFIGS #
###########

# Paths.
PATH_TO_HOME = str(Path.home())
PATH_TO_BINARIES = os.path.join(PATH_TO_HOME, "photogrammetry_binaries")
PATH_TO_DATA = os.path.join(PATH_TO_HOME, "photogrammetry_data")
PATH_TO_TEST_DATA = os.path.join(PATH_TO_DATA, "test")
DEFAULT_PATH_TO_GPS_DATA = os.path.join(PATH_TO_DATA, "210513_113847.csv")
DEFAULT_PATH_TO_LADYBUG_GPS_DATA = \
    os.path.join(PATH_TO_DATA, "ladybug_frame_gps_info_23627.txt")
DEFAULT_PATH_TO_LADYBUG_IMAGES = os.path.join(PATH_TO_DATA, "ladybug_images")
DEFAULT_PATH_TO_WORKING_DIR = os.path.join(PATH_TO_DATA, "working")
DEFAULT_PATH_TO_DEEPLAB_BINARY = \
    os.path.join(PATH_TO_BINARIES, "Deeplabv3plus-xception-ce.hdf5")
# Paths to test files.
DEFAULT_PATH_TO_IMAGE_LIST_FILE = \
    os.path.join(PATH_TO_TEST_DATA, "image_list.p")
DEFAULT_PATH_TO_MINIMAL_IMAGE = os.path.join(PATH_TO_TEST_DATA, "tiny.jpg")
DEFAULT_PATH_TO_TEST_IMAGE_DIR = \
    os.path.join(PATH_TO_TEST_DATA, "ladybug_images")
DEFAULT_PATH_TO_TEST_LADYBUG_GPS_DATA = \
    os.path.join(PATH_TO_DATA, "ladybug_frame_gps_info_test.txt")

# Other defaults.
DEFAULT_ENCODING = "utf-8"
