"""
This code defines a class which holds some configurations used in the other
files in this directory.
"""

# Standard imports.
from pathlib import Path

# Non-standard imports.
from wp17_configs import get_configs_object
import numpy

#############################
# NON-OBJECT CONFIGURATIONS #
#############################

EXPECTED_PATH_TO_CONFIG_JSON = str(Path.home()/"wp17demo_config.json")
SEMICIRCLE_DEGREES = 180
INTERNAL_PYTHON_COMMAND = "python"

############
# DEFAULTS #
############

# General.
DEFAULT_ENCODING = "utf-8"
DEFAULT_MAX_RGB_CHANNEL = 255.0
DEFAULT_LABEL_VALUE_DICT = {
    "background": 0,
    "chimney": 3,
    "door": 5,
    "window": 4,
    "roof": 2,
    "wall": 1
}
DEFAULT_RGB_MAX = (255, 192, 128)
DEFAULT_LOGGING_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"

# Paths.
DEFAULT_PATH_TO_HOME = Path.home()
DEFAULT_PATH_TO_BINARIES = \
    DEFAULT_PATH_TO_HOME/"wp17demo_binaries_and_3rd_party"
DEFAULT_PATH_TO_INPUT = DEFAULT_PATH_TO_HOME/"wp17demo_input"
DEFAULT_PATH_TO_TEST_INPUT = DEFAULT_PATH_TO_HOME/"wp17demo_input_test"
DEFAULT_PATH_TO_OUTPUT = DEFAULT_PATH_TO_HOME/"wp17demo_output"
DEFAULT_PATH_TO_TEST_OUTPUT = DEFAULT_PATH_TO_HOME/"wp17demo_output_test"
DEFAULT_PATH_TO_DEEPLAB_BINARY = \
    DEFAULT_PATH_TO_BINARIES/"Deeplabv3plus-xception-ce.hdf5"
DEFAULT_PATH_TO_POLYGON = DEFAULT_PATH_TO_BINARIES/"polygon0.poly"
DEFAULT_PATH_TO_VOCAB_TREE = (
    DEFAULT_PATH_TO_BINARIES/
    "aliceVision"/
    "share"/
    "aliceVision"/
    "vlfeat_K80L3.SIFT.tree"
)
DEFAULT_PATH_TO_ENERGYPLUS = DEFAULT_PATH_TO_BINARIES/"EnergyPlusV9-5-0"
DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY = \
    DEFAULT_PATH_TO_ENERGYPLUS/"Energy+.idd"
DEFAULT_PATH_TO_WEATHER_DATA = DEFAULT_PATH_TO_ENERGYPLUS/"WeatherData"
DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE = \
    DEFAULT_PATH_TO_WEATHER_DATA/"GBR_Finningley.033600_IWEC.epw"
DEFAULT_PATH_TO_IDF_FILES = Path(__file__).parent/"idf_files"
DEFAULT_PATH_TO_STARTING_POINT_IDF = \
    DEFAULT_PATH_TO_IDF_FILES/"starting_point.idf"
DEFAULT_PATH_TO_OUTPUT_IDF = DEFAULT_PATH_TO_OUTPUT/"output.idf"
DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR = \
    DEFAULT_PATH_TO_OUTPUT/"energy_model_output"
# Path components.
LADYBUG_GPS_DATA_FILENAME = "ladybug_gps_data.txt"
GPS_DATA_FILENAME = "gps_data.csv"
LADYBUG_IMAGES_DIRNAME = "ladybug_images"

# Reconstruction dir.
DEFAULT_COORDINATE_REFERENCE_SYSTEM = "epsg:4326"
DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM = "epsg:27700"
DEFAULT_RADIUS = 20
DEFAULT_VIEW_DISTANCE = 40
DEFAULT_FIELD_OF_VIEW = numpy.pi/2
DEFAULT_CIRCLE_RESOLUTION = 100
DEFAULT_NUMBER_OF_CAMERAS = 5
DEFAULT_IMAGE_EXTENSIONS = (".exr", ".jpeg", ".jpg", ".png")
DEFAULT_OUTPUT_IMAGE_EXTENSION = ".png"

# Batch processes.
DEFAULT_BYTE_LENGTH = 8
DEFAULT_BATCH_PROCESS_TIMEOUT = 7200 # I.e. two hours in SECONDS.

# Energy model.
DEFAULT_WINDOW_SHGC = 0.5 # SHGC = Solar Heat Gain Coefficient.
DEFAULT_AIR_CHANGE_PER_HOUR = 0.5
DEFAULT_SETPOINT_HEATING = 18
DEFAULT_SETPOINT_COOLING = 26
DEFAULT_BOILER_EFFICIENCY = 0.8

#####################
# SPECIAL FUNCTIONS #
#####################

def merge_stem_and_filename(stem, filename):
    """ Attach a stem, either a string or a path object, to a filename. """
    to_string = False
    if isinstance(stem, str):
        to_string = True
        stem = Path(stem)
    result = stem/filename
    if to_string:
        result = str(result)
    return result

def make_path_to_gps_data(
        stem=DEFAULT_PATH_TO_INPUT, filename=DEFAULT_GPS_DATA_FILENAME,
    ):
    """ Make the path, filling in the blanks with defaults. """
    return merge_stem_and_filename(stem, filename)

def make_path_to_ladybug_gps_data(
        stem=DEFAULT_PATH_TO_INPUT, filename=DEFAULT_LADYBUG_GPS_DATA_FILENAME
    ):
    """ Make the path, filling in the blanks with defaults. """
    return merge_stem_and_filename(stem, filename)

def make_path_to_ladybug_images(
        stem=DEFAULT_PATH_TO_INPUT, dirname=DEFAULT_LADYBUG_IMAGES_DIRNAME
    ):
    """ Make the path, filling in the blanks with defaults. """
    return merge_stem_and_filename(stem, filename)

#########################
# CONFIGURATIONS OBJECT #
#########################

# Defaults dictionary.
DEFAULTS: {
    "general": {
        "coordinate_reference_system": DEFAULT_COORDINATE_REFERENCE_SYSTEM,
        "source_coordinate_reference_system": \
            DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM,
        "encoding": DEFAULT_ENCODING,
        "max_rgb_channel": DEFAULT_MAX_RGB_CHANNEL,
        "label_value_dict": DEFAULT_LABEL_VALUE_DICT,
        "rgb_max": DEFAULT_RGB_MAX,
        "logging_format": DEFAULT_LOGGING_FORMAT
    },
    "paths": {
        "path_to_home": str(DEFAULT_PATH_TO_HOME),
        "path_to_binaries": str(DEFAULT_PATH_TO_BINARIES),
        "path_to_input": str(DEFAULT_PATH_TO_INPUT),
        "path_to_output": str(DEFAULT_PATH_TO_OUTPUT),
        "path_to_polygon": str(DEFAULT_PATH_TO_POLYGON),
        "path_to_gps_data": str(make_path_to_gps_data()),
        "path_to_ladybug_gps_data": str(make_path_to_ladybug_gps_data()),
        "path_to_ladybug_images": str(make_path_to_ladybug_images()),
        "path_to_deeplab_binary": str(DEFAULT_PATH_TO_DEEPLAB_BINARY),
        "path_to_energyplus": str(DEFAULT_PATH_TO_ENERGYPLUS),
        "path_to_energyplus_input_data_dictionary": \
            str(DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY),
        "path_to_weather_data": str(DEFAULT_PATH_TO_WEATHER_DATA),
        "path_to_energyplus_weather_file": \
            str(DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE),
        "path_to_idf_files": str(DEFAULT_PATH_TO_IDF_FILES),
        "path_to_starting_point_idf": str(DEFAULT_PATH_TO_STARTING_POINT_IDF),
        "path_to_output_idf": str(DEFAULT_PATH_TO_OUTPUT_IDF),
        "path_to_energy_model_output_dir": \
            str(DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR),
        "path_to_test_input": str(DEFAULT_PATH_TO_TEST_INPUT)
    },
    "reconstruction_dir": {
        "radius": DEFAULT_RADIUS,
        "view_distance": DEFAULT_VIEW_DISTANCE,
        "field_of_view": DEFAULT_FIELD_OF_VIEW,
        "circle_resolution": DEFAULT_CIRCLE_RESOLUTION,
        "number_of_cameras": DEFAULT_NUMBER_OF_CAMERAS,
        "image_extensions": DEFAULT_IMAGE_EXTENSIONS,
        "output_image_extension": DEFAULT_OUTPUT_IMAGE_EXTENSION
    },
    "batch_process": {
        "byte_length": DEFAULT_BYTE_LENGTH,
        "timeout": DEFAULT_BATCH_PROCESS_TIMEOUT,
        "path_to_vocab_tree": DEFAULT_PATH_TO_VOCAB_TREE
    },
    "energy_model": {
        "window_shgc": DEFAULT_WINDOW_SHGC,
        "air_change_per_hour": DEFAULT_AIR_CHANGE_PER_HOUR,
        "setpoint_heating": DEFAULT_SETPOINT_HEATING,
        "setpoint_cooling": DEFAULT_SETPOINT_COOLING,
        "boiler_efficiency": DEFAULT_BOILER_EFFICIENCY
    }
}

# Hey presto.
CONFIGS = \
    get_configs_object(DEFAULTS, path_to_overrides=EXPECTED_PATH_TO_CONFIG_JSON)
