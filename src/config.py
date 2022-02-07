"""
The code in this file holds some configurations for the proximate scripts.
"""

# Standard imports.
import os
from pathlib import Path

# Non-standard imports.
import numpy

###########
# CONFIGS #
###########

# Paths.
#PATH_TO_HOME = str(Path.home())
PATH_TO_HOME = r"G:\"
PATH_TO_BINARIES = os.path.join(PATH_TO_HOME, "photogrammetry_binaries")
PATH_TO_INPUT = os.path.join(PATH_TO_HOME, "photogrammetry_input")
DEFAULT_PATH_TO_GPS_DATA = os.path.join(PATH_TO_INPUT, "210513_113847.csv")
DEFAULT_PATH_TO_LADYBUG_GPS_DATA = \
    os.path.join(PATH_TO_INPUT, "ladybug_frame_gps_info_23627.txt")
DEFAULT_PATH_TO_LADYBUG_IMAGES = os.path.join(PATH_TO_INPUT, "ladybug_images")
DEFAULT_PATH_TO_DEEPLAB_BINARY = \
    os.path.join(PATH_TO_BINARIES, "Deeplabv3plus-xception-ce.hdf5")
DEFAULT_PATH_TO_POLYGON = os.path.join(PATH_TO_BINARIES, "polygon0.poly")
DEFAULT_PATH_TO_OUTPUT = os.path.join(PATH_TO_HOME, "photogrammetry_output")

# Other defaults.
DEFAULT_ENCODING = "utf-8"
DEFAULT_COORDINATE_REFERENCE_SYSTEM = "epsg:4326"
DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM = "epsg:27700"
DEFAULT_RADIUS = 20
DEFAULT_VIEW_DISTANCE = 40
DEFAULT_FIELD_OF_VIEW = numpy.pi/2
DEFAULT_CIRCLE_RESOLUTION = 100
