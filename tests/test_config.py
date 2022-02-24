"""
This code defines some unit tests for the Config class.
"""

# Standard imports.
import os

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
        config_obj.general["path_to_output"] == config.DEFAULT_PATH_TO_OUTPUT
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
    new_path_to_output = "/something/else"
    test_config_json_path = "test_config.json"
    test_config_json_str = (
        '{ '+
            '"general": { '+
                '"path_to_output": "'+new_path_to_output+'", '+
                '"path_to_input": null '+
            '} '+
        '}'
    )
    with open(test_config_json_path, "w", encoding=expected.ENCODING) as jsonf:
        jsonf.write(test_config_json_str)
    custom_config = config.Configs(test_config_json_path)
    immutable_custom_config = custom_config.export_as_immutable()
    assert immutable_custom_config.general.path_to_output == new_path_to_output
    assert (
        immutable_custom_config.general.path_to_input == \
            config.DEFAULT_PATH_TO_INPUT
    )
    os.remove(test_config_json_path)
