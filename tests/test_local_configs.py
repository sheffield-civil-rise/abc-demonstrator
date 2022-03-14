"""
This code defines some unit tests for the Config class.
"""

# Standard imports.
from pathlib import Path

# Source imports.
from local_configs import (
    make_path_to_gps_data,
    make_path_to_ladybug_gps_data,
    make_path_to_ladybug_images,
    DEFAULT_PATH_TO_INPUT,
    DEFAULT_GPS_DATA_FILENAME,
    DEFAULT_LADYBUG_GPS_DATA_FILENAME,
    DEFAULT_LADYBUG_IMAGES_DIRNAME
)

###########
# TESTING #
###########

def test_make_path_to_gps_data():
    """ Test that this function produces the desired path. """
    actual_result = make_path_to_gps_data()
    expected_result = DEFAULT_PATH_TO_INPUTS/DEFAULT_GPS_DATA_FILENAME
    assert actual_result == expected_result

def test_make_path_to_ladybug_gps_data():
    """ Test that this function produces the desired path. """
    different_filename = "test.txt"
    actual_result = make_path_to_ladybug_gps_data(filename=different_filename)
    expected_result = DEFAULT_PATH_TO_INPUTS/different_filename
    assert actual_result == expected_result

def test_make_path_to_ladybug_images():
    """ Test that this function produces the desired path. """
    different_stem = "/different/stem"
    actual_result = make_path_to_ladybug_images(stem=different_stem)
    expected_result = str(Path(different_stem)/DEFAULT_LADYBUG_IMAGES_DIRNAME)
    assert actual_result == expected_result
