"""
This code defines a class which holds some configurations used in the other
files in this directory.
"""

# Standard imports.
import json
import os
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

# Non-standard imports.
import numpy

############
# DEFAULTS #
############

# General.
#DEFAULT_PATH_TO_HOME = str(Path.home())
DEFAULT_PATH_TO_HOME = "G:\\"
DEFAULT_PATH_TO_BINARIES = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_binaries_and_3rd_party")
DEFAULT_PATH_TO_INPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_input")
DEFAULT_PATH_TO_OUTPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_output")
DEFAULT_PATH_TO_DEMO_OUTPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_output_demo")
DEFAULT_PATH_TO_REPO = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_e110a")
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

# Reconstruction dir.
DEFAULT_GPS_DATA_FILENAME = "gps_data.csv"
DEFAULT_LADYBUG_GPS_DATA_FILENAME = "ladybug_gps_data.txt"
DEFAULT_LADYBUG_IMAGES_DIRNAME = "ladybug_images"
DEFAULT_PATH_TO_DEEPLAB_BINARY = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "Deeplabv3plus-xception-ce.hdf5")
DEFAULT_PATH_TO_POLYGON = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "polygon0.poly")
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
DEFAULT_PATH_TO_VOCAB_TREE = \
    os.path.join(
        DEFAULT_PATH_TO_BINARIES,
        r"aliceVision\share\aliceVision\vlfeat_K80L3.SIFT.tree"
    )

# Energy model.
DEFAULT_PATH_TO_ENERGYPLUS = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "EnergyPlusV9-5-0")
DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "Energy+.idd")
DEFAULT_PATH_TO_WEATHER_DATA = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "WeatherData")
DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE = \
    os.path.join(DEFAULT_PATH_TO_WEATHER_DATA, "GBR_Finningley.033600_IWEC.epw")
DEFAULT_PATH_TO_IDF_FILES = os.path.join(DEFAULT_PATH_TO_REPO, "idf_files")
DEFAULT_PATH_TO_STARTING_POINT_IDF = \
    os.path.join(DEFAULT_PATH_TO_IDF_FILES, "starting_point.idf")
DEFAULT_PATH_TO_OUTPUT_IDF = os.path.join(DEFAULT_PATH_TO_OUTPUT, "output.idf")
DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR = \
    os.path.join(DEFAULT_PATH_TO_OUTPUT, "energy_model_output")
DEFAULT_WINDOW_SHGC = 0.5 # SHGC = Solar Heat Gain Coefficient.
DEFAULT_AIR_CHANGE_PER_HOUR = 0.5
DEFAULT_SETPOINT_HEATING = 18
DEFAULT_SETPOINT_COOLING = 26
DEFAULT_BOILER_EFFICIENCY = 0.8

# Test.
DEFAULT_PATH_TO_TEST_INPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_input_test")

# Other.
EXPECTED_PATH_TO_CONFIG_JSON = \
    os.path.join(str(Path.home()), "photogrammetry_config.json")
SEMICIRCLE_DEGREES = 180

##############
# MAIN CLASS #
##############

@dataclass
class Configs:
    """ The class in question. """
    # Class attributes.
    DEFAULTS: ClassVar[dict] = {
        "general": {
            "path_to_home": DEFAULT_PATH_TO_HOME,
            "path_to_binaries": DEFAULT_PATH_TO_BINARIES,
            "path_to_input": DEFAULT_PATH_TO_INPUT,
            "path_to_output": DEFAULT_PATH_TO_OUTPUT,
            "path_to_demo_output": DEFAULT_PATH_TO_DEMO_OUTPUT,
            "path_to_repo": DEFAULT_PATH_TO_REPO,
            "path_to_polygon": DEFAULT_PATH_TO_POLYGON,
            "coordinate_reference_system": DEFAULT_COORDINATE_REFERENCE_SYSTEM,
            "source_coordinate_reference_system": \
                DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM,
            "encoding": DEFAULT_ENCODING,
            "max_rgb_channel": DEFAULT_MAX_RGB_CHANNEL,
            "label_value_dict": DEFAULT_LABEL_VALUE_DICT,
            "rgb_max": DEFAULT_RGB_MAX
        },
        "reconstruction_dir": {
            "path_to_gps_data": make_path_to_gps_data(),
            "path_to_ladybug_gps_data": make_path_to_ladybug_gps_data(),
            "path_to_ladybug_images": make_path_to_ladybug_images(),
            "path_to_deeplab_binary": DEFAULT_PATH_TO_DEEPLAB_BINARY,
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
            "path_to_energyplus": DEFAULT_PATH_TO_ENERGYPLUS,
            "path_to_energyplus_input_data_dictionary": \
                DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY,
            "path_to_weather_data": DEFAULT_PATH_TO_WEATHER_DATA,
            "path_to_energyplus_weather_file": \
                DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE,
            "path_to_idf_files": DEFAULT_PATH_TO_IDF_FILES,
            "path_to_starting_point_idf": DEFAULT_PATH_TO_STARTING_POINT_IDF,
            "path_to_output_idf": DEFAULT_PATH_TO_OUTPUT_IDF,
            "path_to_energy_model_output_dir": \
                DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR,
            "window_shgc": DEFAULT_WINDOW_SHGC,
            "air_change_per_hour": DEFAULT_AIR_CHANGE_PER_HOUR,
            "setpoint_heating": DEFAULT_SETPOINT_HEATING,
            "setpoint_cooling": DEFAULT_SETPOINT_COOLING,
            "boiler_efficiency": DEFAULT_BOILER_EFFICIENCY
        },
        "test": {
            "path_to_input": DEFAULT_PATH_TO_TEST_INPUT
        }
    }

    # Fields.
    path_to_json: str = EXPECTED_PATH_TO_CONFIG_JSON
    enc: str = DEFAULT_ENCODING
    general: dict = None
    reconstruction_dir: dict = None
    batch_process: dict = None
    energy_model: dict = None
    # Generated fields.
    json_dict: dict = None

    def __post_init__(self):
        self.general = self.DEFAULTS["general"]
        self.reconstruction_dir = self.DEFAULTS["reconstruction_dir"]
        self.batch_process = self.DEFAULTS["batch_process"]
        self.energy_model = self.DEFAULTS["energy_model"]
        self.test = self.DEFAULTS["test"]
        self.set_from_json()

    def set_from_json(self):
        """ Attempt to override some of the defaults from our JSON file. """
        if self.path_to_json and os.path.exists(self.path_to_json):
            with open(self.path_to_json, "r", encoding=self.enc) as json_file:
                json_str = json_file.read()
                self.json_dict = json.loads(json_str)
            self.set_sub_dictionary_from_json("general", self.general)
            self.set_sub_dictionary_from_json(
                "reconstruction_dir", self.reconstruction_dir
            )
            self.set_sub_dictionary_from_json(
                "batch_process", self.batch_process
            )
            self.set_sub_dictionary_from_json(
                "energy_model", self.energy_model
            )
            self.set_sub_dictionary_from_json("test", self.test)

    def set_sub_dictionary_from_json(self, sub_dict_key, sub_dict):
        """ Attempt to override some of the values in a given dictionary using
        our JSON file. """
        if (sub_dict_key in self.json_dict) and self.json_dict[sub_dict_key]:
            for key in self.json_dict[sub_dict_key]:
                if key in sub_dict:
                    if self.json_dict[sub_dict_key][key] is not None:
                        sub_dict[key] = self.json_dict[sub_dict_key][key]

    def export_as_immutable(self):
        """ Export the data in this class into an immutable form. """
        Config = \
            namedtuple(
                "Config",
                [
                    "general",
                    "reconstruction_dir",
                    "batch_process",
                    "energy_model",
                    "test"
                ]
            )
        result = \
            Config(
                general=self.export_general(),
                reconstruction_dir=self.export_reconstruction_dir(),
                batch_process=self.export_batch_process(),
                energy_model=self.export_energy_model(),
                test=self.export_test()
            )
        return result

    def export_general(self):
        """ Convert this dictionary into a named tuple. """
        General = namedtuple("General", list(self.general.keys()))
        result = General(**self.general)
        return result

    def export_reconstruction_dir(self):
        """ Convert this dictionary into a named tuple. """
        ReconstructionDir = \
            namedtuple(
                "ReconstructionDir", list(self.reconstruction_dir.keys())
            )
        result = ReconstructionDir(**self.reconstruction_dir)
        return result

    def export_batch_process(self):
        """ Convert this dictionary into a named tuple. """
        BatchProcess = \
            namedtuple("BatchProcess", list(self.batch_process.keys()))
        result = BatchProcess(**self.batch_process)
        return result

    def export_energy_model(self):
        """ Convert this dictionary into a named tuple. """
        EnergyModel = namedtuple("EnergyModel", list(self.energy_model.keys()))
        result = EnergyModel(**self.energy_model)
        return result

    def export_test(self):
        """ Convert this dictionary into a named tuple. """
        Test = namedtuple("Test", list(self.test.keys()))
        result = Test(**self.test)
        return result

####################
# HELPER FUNCTIONS #
####################

def get_configs():
    """ Get an immutable config object. """
    config_obj = Configs()
    result = config_obj.export_as_immutable()
    return result

def make_path_to_gps_data(
        stem=DEFAULT_PATH_TO_INPUT, filename=DEFAULT_GPS_DATA_FILENAME
    ):
    """ Make the path, filling in the blanks with defaults. """
    result = os.path.join(stem, filename)
    return result

def make_path_to_ladybug_gps_data(
        stem=DEFAULT_PATH_TO_INPUT, filename=DEFAULT_LADYBUG_GPS_DATA_FILENAME
    ):
    """ Make the path, filling in the blanks with defaults. """
    result = os.path.join(stem, filename)
    return result

def make_path_to_ladybug_images(
        stem=DEFAULT_PATH_TO_INPUT, dirname=DEFAULT_LADYBUG_IMAGES_DIRNAME
    ):
    """ Make the path, filling in the blanks with defaults. """
    result = os.path.join(stem, filename)
    return result
