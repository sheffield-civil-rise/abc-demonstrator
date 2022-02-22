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

@pytest.fixture
def demo_obj():
    return Demonstrator()

def test_rec_dir_gen_fields(demo_obj):
    """ Test the FIELDS of the ReconstructionDirGenerator sub-object. """
    demo_obj.make_and_run_reconstruction_dir_generator()
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

