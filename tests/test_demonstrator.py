"""
This code defines a series of tests which check the output - both files and the
fields of sub-objects - of the Demonstrator class.
"""

# Standard imports.
import os

# Non-standard imports.
import pytest

# Local imports.
import config
import expected
from demonstrator import Demonstrator
from make_checksum import make_checksum

###########
# TESTING #
###########

@pytest.fixture(scope="module")
def demo_obj():
    result = Demonstrator()
    result.demonstrate()
    return result

def test_rec_dir_gen_fields(demo_obj):
    """ Test the FIELDS of the ReconstructionDirGenerator sub-object. """
    assert (
        demo_obj.rec_dir_gen.path_to_output ==
            config.DEFAULT_PATH_TO_DEMO_OUTPUT
    )
    assert (
        demo_obj.rec_dir_gen.path_to_output_images ==
            os.path.join(config.DEFAULT_PATH_TO_DEMO_OUTPUT, "images")
    )
    assert (
        demo_obj.rec_dir_gen.path_to_labelled_images ==
            os.path.join(config.DEFAULT_PATH_TO_DEMO_OUTPUT, "labelled")
    )
    assert (
        demo_obj.rec_dir_gen.path_to_masked_images ==
            os.path.join(config.DEFAULT_PATH_TO_DEMO_OUTPUT, "masked")
    )

def test_rec_dir_gen_files(demo_obj):
    """ Test that the ReconstructionDirGenerator sub-object outputs the right
    files. """
    actual_labelled_image_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_labelled_images,
                expected.LABELLED_IMAGE_FILENAME
            )
        )
    actual_masked_image_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_masked_images,
                expected.MASKED_IMAGE_FILENAME
            )
        )
    actual_camera_init_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_output, "cameraInit.sfm"
            )
        )
    actual_camera_init_label_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_output, "cameraInit_label.sfm"
            )
        )
    assert actual_labelled_image_checksum == expected.LABELLED_IMAGE_CHECKSUM
    assert actual_masked_image_checksum == expected.MASKED_IMAGE_CHECKSUM
    assert actual_camera_init_checksum == expected.CAMERA_INIT_CHECKSUM
    assert (
        actual_camera_init_label_checksum ==
            expected.CAMERA_INIT_LABEL_CHECKSUM
    )

def test_batch_processor_files(demo_obj):
    """ Test the files which the BatchProcessor sub-object outputs. """
    assert os.path.exists(demo_obj.path_to_cache)
    for subdirectory in expected.CACHE_SUBDIRECTORIES:
        path_to = os.path.join(demo_obj.path_to_cache, subdirectory)
        assert os.path.exists(path_to)
        assert len(os.listdir(path_to)) > 0

def test_height_calculator(demo_obj):
    """ Test that the height calculator actually produces an output. """
    assert demo_obj.height_calculator.result is not None

def test_window_to_wall_ratio_calculator(demo_obj):
    """ Test that the WWR calculator actually produces an output. """
    assert demo_obj.window_to_wall_ratio_calculator.result is not None

def test_energy_model_idf(demo_obj):
    """ Test that the energy model's IDF object has the fields we want. """
    print(demo_obj.energy_model_generator.idf_obj.idfobjects.keys())
    assert False
