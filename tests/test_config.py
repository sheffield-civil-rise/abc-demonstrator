"""
This code defines some unit tests for the Config class.
"""

# Standard imports.
import os
from pathlib import Path

# Non-standard imports.
import pytest

# Source imports.
import config

# Local imports.
import expected

###########
# TESTING #
###########

@pytest.fixture(scope="module")
def config_obj():
    """ Create a Configs object. """
    result = config.Configs(path_to_json=False)
    return result

def test_defaults(config_obj):
    """ Test that the expected defaults are present in the dictionary fields
    of the config object. """
    assert (
        config_obj.general["encoding"] == config.DEFAULT_ENCODING
    )
    assert (
        config_obj.energy_model["boiler_efficiency"] ==
            config.DEFAULT_BOILER_EFFICIENCY
    )

def test_immutability(config_obj):
    """ Test that trying to change one of the fields raises the appropriate
    exception. """
    config_immutable = config_obj.export_as_immutable()
    with pytest.raises(AttributeError):
        config_immutable.batch_process.timeout = 1234

def test_custom_json():
    """ Test that we can set configurations via a JSON file. """
    new_byte_length = 17
    test_config_json_path = "test_config.json"
    test_config_json_str = (
        '{ '+
            '"batch_process": { '+
                '"byte_length": '+str(new_byte_length)+", "+
                '"timeout": null '+
            '} '+
        '}'
    )
    with open(test_config_json_path, "w", encoding=expected.ENCODING) as jsonf:
        jsonf.write(test_config_json_str)
    custom_config = config.Configs(test_config_json_path)
    immutable_custom_config = custom_config.export_as_immutable()
    assert immutable_custom_config.batch_process.byte_length == new_byte_length
    assert (
        immutable_custom_config.batch_process.timeout == \
            config.DEFAULT_BATCH_PROCESS_TIMEOUT
    )
    os.remove(test_config_json_path)

def test_path_overrides():
    """ Test the config file can override paths as expected. """
    new_path_to_home = "/home/smeg"
    new_path_to_output = "/smeghan/smarkle"
    test_config_json_path = "test_config.json"
    test_config_json_str = (
        '{ '+
            '"paths": { '+
                '"path_to_home": "'+new_path_to_home+'", '+
                '"path_to_output": "'+new_path_to_output+'" '+
            '} '+
        '}'
    )
    with open(test_config_json_path, "w", encoding=expected.ENCODING) as jsonf:
        jsonf.write(test_config_json_str)
    custom_config = config.Configs(test_config_json_path)
    immutable_custom_config = custom_config.export_as_immutable()
    assert immutable_custom_config.paths.path_to_home == new_path_to_home
    assert immutable_custom_config.paths.path_to_output == new_path_to_output
    assert (
        Path(immutable_custom_config.paths.path_to_binaries).parent ==
            Path(new_path_to_home)
    )
    os.remove(test_config_json_path)
