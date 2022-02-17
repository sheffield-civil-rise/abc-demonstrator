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
PATH_TO_HOME = "G:"
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
DEFAULT_PATH_TO_DEMO_OUTPUT = \
    os.path.join(PATH_TO_HOME, "photogrammetry_output_demo")
DEFAULT_PATH_TO_ENERGYPLUS = os.path.join(PATH_TO_BINARIES, "EnergyPlusV9-5-0")
DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "Energy+.idd")
DEFAULT_PATH_TO_WEATHER_DATA = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "WeatherData")
DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE = \
    os.path.join(DEFAULT_PATH_TO_WEATHER_DATA, "GBR_Finningley.033600_IWEC.epw")
DEFAULT_PATH_TO_REPO = os.path.join(PATH_TO_HOME, "photogrammetry_e110a")
DEFAULT_PATH_TO_IDF_FILES = os.path.join(DEFAULT_PATH_TO_REPO, "idf_files")
DEFAULT_PATH_TO_STARTING_POINT_IDF = \
    os.path.join(DEFAULT_PATH_TO_IDF_FILES, "starting_point.idf")
DEFAULT_PATH_TO_OUTPUT_IDF = os.path.join(DEFAULT_PATH_TO_OUTPUT, "output.idf")
DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR = \
    os.path.join(DEFAULT_PATH_TO_OUTPUT, "energy_model_output")

# Other defaults.
DEFAULT_ENCODING = "utf-8"
DEFAULT_COORDINATE_REFERENCE_SYSTEM = "epsg:4326"
DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM = "epsg:27700"
DEFAULT_RADIUS = 20
DEFAULT_VIEW_DISTANCE = 40
DEFAULT_FIELD_OF_VIEW = numpy.pi/2
DEFAULT_CIRCLE_RESOLUTION = 100
DEFAULT_NUMBER_OF_CAMERAS = 5
DEFAULT_IMAGE_EXTENSIONS = (".exr", ".jpeg", ".jpg", ".png")
DEFAULT_OUTPUT_IMAGE_EXTENSION = ".png"
DEFAULT_BYTE_LENGTH = 8
DEFAULT_BATCH_PROCESS_TIMEOUT = 7200 # I.e. two hours in SECONDS.
DEFAULT_CHECK_INTERVAL = 30 # I.e. thirty SECONDS.
DEFAULT_WINDOW_SHGC = 0.5 # SHGC = Solar Heat Gain Coefficient.
DEFAULT_AIR_CHANGE_PER_HOUR = 0.5
DEFAULT_SETPOINT_HEATING = 18
DEFAULT_SETPOINT_COOLING = 26
DEFAULT_BOILER_EFFICIENCY = 0.8

# Other.
SEMICIRCLE_DEGREES = 180
MAX_RGB_CHANNEL = 255.0
LABEL_VALUE_DICT = {
    "background": 0,
    "chimney": 3,
    "door": 5,
    "window": 4,
    "roof": 2,
    "wall": 1
}
RGB_MAX = (255, 192, 128)
