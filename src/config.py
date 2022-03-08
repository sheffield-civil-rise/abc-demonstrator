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
DEFAULT_PATH_TO_HOME = str(Path.home())
DEFAULT_PATH_TO_BINARIES = \
    os.path.join(DEFAULT_PATH_TO_HOME, "wp17demo_binaries_and_3rd_party")
DEFAULT_PATH_TO_INPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "wp17demo_input")
DEFAULT_PATH_TO_OUTPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "wp17demo_output")
DEFAULT_PATH_TO_DEEPLAB_BINARY = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "Deeplabv3plus-xception-ce.hdf5")
DEFAULT_PATH_TO_POLYGON = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "polygon0.poly")
DEFAULT_PATH_TO_VOCAB_TREE = \
    os.path.join(
        DEFAULT_PATH_TO_BINARIES,
        r"aliceVision\share\aliceVision\vlfeat_K80L3.SIFT.tree"
    )
DEFAULT_PATH_TO_ENERGYPLUS = \
    os.path.join(DEFAULT_PATH_TO_BINARIES, "EnergyPlusV9-5-0")
DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "Energy+.idd")
DEFAULT_PATH_TO_WEATHER_DATA = \
    os.path.join(DEFAULT_PATH_TO_ENERGYPLUS, "WeatherData")
DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE = \
    os.path.join(DEFAULT_PATH_TO_WEATHER_DATA, "GBR_Finningley.033600_IWEC.epw")
DEFAULT_PATH_TO_IDF_FILES = str(Path(__file__).parent/"idf_files")
DEFAULT_PATH_TO_STARTING_POINT_IDF = \
    os.path.join(DEFAULT_PATH_TO_IDF_FILES, "starting_point.idf")
DEFAULT_PATH_TO_OUTPUT_IDF = os.path.join(DEFAULT_PATH_TO_OUTPUT, "output.idf")
DEFAULT_PATH_TO_ENERGY_MODEL_OUTPUT_DIR = \
    os.path.join(DEFAULT_PATH_TO_OUTPUT, "energy_model_output")
DEFAULT_PATH_TO_TEST_INPUT = \
    os.path.join(DEFAULT_PATH_TO_HOME, "photogrammetry_input_test")

# Reconstruction dir.
DEFAULT_GPS_DATA_FILENAME = "gps_data.csv"
DEFAULT_LADYBUG_GPS_DATA_FILENAME = "ladybug_gps_data.txt"
DEFAULT_LADYBUG_IMAGES_DIRNAME = "ladybug_images"
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

# Other.
EXPECTED_PATH_TO_CONFIG_JSON = \
    os.path.join(str(Path.home()), "wp17demo_config.json")
SEMICIRCLE_DEGREES = 180
INTERNAL_PYTHON_COMMAND = "python"

#####################
# SPECIAL FUNCTIONS #
#####################

def reroot_list(path_list, new_path, old_path):
    """ Change a path in a list of paths in a tree, and thereby change each
    path the derives from, recursively. """
    for index, path_string in enumerate(path_list):
        if Path(path_string) == Path(old_path):
            path_list[index] = new_path
        elif Path(path_string).parent == Path(old_path):
            new_path_dash = str(Path(new_path)/Path(path_string).name)
            old_path_dash = path_string
            path_list[index] = new_path_dash
            reroot(path_list, new_path_dash, old_path_dash)

def reroot(path_dict, new_path, old_path):
    """ As above, but for a dictionary. """
    key_storage = []
    value_storage = []
    for key, value in path_dict.items():
        key_storage.append(key)
        value_storage.append(value)
    reroot_list(value_storage, new_path, old_path)
    result = dict()
    for index, value in enumerate(value_storage):
        result[key_storage[index]] = value
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
    result = os.path.join(stem, dirname)
    return result

##############
# MAIN CLASS #
##############

@dataclass
class Configs:
    """ The class in question. """
    # Class attributes.
    DEFAULTS: ClassVar[dict] = {
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
            "path_to_home": DEFAULT_PATH_TO_HOME,
            "path_to_binaries": DEFAULT_PATH_TO_BINARIES,
            "path_to_input": DEFAULT_PATH_TO_INPUT,
            "path_to_output": DEFAULT_PATH_TO_OUTPUT,
            "path_to_polygon": DEFAULT_PATH_TO_POLYGON,
            "path_to_gps_data": make_path_to_gps_data(),
            "path_to_ladybug_gps_data": make_path_to_ladybug_gps_data(),
            "path_to_ladybug_images": make_path_to_ladybug_images(),
            "path_to_deeplab_binary": DEFAULT_PATH_TO_DEEPLAB_BINARY,
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
            "path_to_test_input": DEFAULT_PATH_TO_TEST_INPUT
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
    GENERAL_KEY: ClassVar[str] = "general"
    PATHS_KEY: ClassVar[str] = "paths"
    RECONSTRUCTION_DIR_KEY: ClassVar[str] = "reconstruction_dir"
    BATCH_PROCESS_KEY: ClassVar[str] = "batch_process"
    ENERGY_MODEL_KEY: ClassVar[str] = "energy_model"

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
        self.general = self.DEFAULTS[self.GENERAL_KEY]
        self.paths = self.DEFAULTS[self.PATHS_KEY]
        self.reconstruction_dir = self.DEFAULTS[self.RECONSTRUCTION_DIR_KEY]
        self.batch_process = self.DEFAULTS[self.BATCH_PROCESS_KEY]
        self.energy_model = self.DEFAULTS[self.ENERGY_MODEL_KEY]
        self.set_from_json()

    def set_from_json(self):
        """ Attempt to override some of the defaults from our JSON file. """
        if self.path_to_json and os.path.exists(self.path_to_json):
            with open(self.path_to_json, "r", encoding=self.enc) as json_file:
                json_str = json_file.read()
                self.json_dict = json.loads(json_str)
            self.set_sub_dictionary_from_json(self.GENERAL_KEY, self.general)
            self.set_paths_from_json(self.PATHS_KEY)
            self.set_sub_dictionary_from_json(
                self.RECONSTRUCTION_DIR_KEY, self.reconstruction_dir
            )
            self.set_sub_dictionary_from_json(
                self.BATCH_PROCESS_KEY, self.batch_process
            )
            self.set_sub_dictionary_from_json(
                self.ENERGY_MODEL_KEY, self.energy_model
            )

    def set_sub_dictionary_from_json(self, sub_dict_key, sub_dict):
        """ Attempt to override some of the values in a given dictionary using
        our JSON file. """
        if (sub_dict_key in self.json_dict) and self.json_dict[sub_dict_key]:
            for key in self.json_dict[sub_dict_key]:
                if key in sub_dict:
                    if self.json_dict[sub_dict_key][key] is not None:
                        sub_dict[key] = self.json_dict[sub_dict_key][key]

    def set_paths_from_json(self):
        """ Set the paths from the config file, which has to be done in a
        slightly more crafty way than with the others. """
        self.set_sub_dictionary_from_json(paths_key, self.paths)
        if (
            self.PATHS_KEY in self.json_dict) and
            self.json_dict[self.PATHS_KEY]
        ):
            paths_dict = self.json_dict[self.PATHS_KEY]
            for key, new_path in paths_dict.items():
                if new_path:
                    old_path = self.paths[key]
                    self.paths = reroot(self.paths, new_path, old_path)
        self.set_sub_dictionary_from_json(self.PATHS_KEY, self.paths)

    def export_as_immutable(self):
        """ Export the data in this class into an immutable form. """
        Config = \
            namedtuple(
                "Config",
                [
                    self.GENERAL_KEY,
                    self.PATHS_KEY,
                    self.RECONSTRUCTION_DIR_KEY,
                    self.BATCH_PROCESS_KEY,
                    self.ENERGY_MODEL_KEY
                ]
            )
        result = \
            Config(
                general=self.export_general(),
                paths=self.export_paths(),
                reconstruction_dir=self.export_reconstruction_dir(),
                batch_process=self.export_batch_process(),
                energy_model=self.export_energy_model()
            )
        return result

    def export_general(self):
        """ Convert this dictionary into a named tuple. """
        General = namedtuple("General", list(self.general.keys()))
        result = General(**self.general)
        return result

    def export_paths(self):
        """ Convert this dictionary into a named tuple. """
        Paths = namedtuple("Paths", list(self.paths.keys()))
        result = Paths(**self.paths)
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

####################
# HELPER FUNCTIONS #
####################

def get_configs():
    """ Get an immutable config object. """
    config_obj = Configs()
    result = config_obj.export_as_immutable()
    return result
