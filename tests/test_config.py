"""
This code defines some unit tests for the Config class.
"""

# Non-standard imports.
import pytest

# Local imports.
import config

###########
# TESTING #
###########

@pytest.fixture(scope="module")
def config_obj():
    result = config.Config(path_to_json=False)
    return result

def test_defaults(config_obj):
    """ Test that the expected defaults are present in the dictionary fields
    of the config object. """
    assert config_obj.general["path_to_output"] == DEFAULT_PATH_TO_OUTPUT
    assert (
        config_obj.energy_model["boiler_efficiency"] ==
            DEFAULT_BOILER_EFFICIENCY
    )

def test_immutability(config_obj):
    """ Test that trying to change one of the fields raises the appropriate
    exception. """
    config_immutable = config_obj.export_to_immutable()
    with pytest.raises(TypeError):
        config_immutable.batch_process.timeout = 1234
