"""
This code defines a series of tests which check the output - both files and the
fields of sub-objects - of the Demonstrator class.
"""

# Standard imports.
import hashlib
import os

# Local imports.
import config
from demonstrator import Demonstrator

# Local constants.
CHUNK_SIZE = 4096
EXPECTED_LABELLED_IMAGE_FILENAME = "68273.png"
EXPECTED_LABELLED_IMAGE_CHECKSUM = "8112283d0796bb55930c1d2a5ba450ba"
EXPECTED_MASKED_IMAGE_FILENAME = (
    "Ladybug-Stream-20210513-141154_ColorProcessed_006827_Cam3_192355_125-"+
    "6896.jpg"
)
EXPECTED_MASKED_IMAGE_CHECKSUM = "66f9b85957e8c6f5e644a86f4e8a76ae"

####################
# HELPER FUNCTIONS #
####################

def make_checksum(path_to_file, chunk_size=CHUNK_SIZE):
    """ Make a checksum for a given file. """
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as hash_me:
        for chunk in iter(lambda: hash_me.read(chunk_size), b""):
            hash_md5.update(chunk)
    result = hash_md5.hexdigest()
    return result

###########
# TESTING #
###########

def check_rec_dir_gen_fields(demo_obj):
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

def check_rec_dir_gen_files(demo_obj):
    """ Test that the ReconstructionDirGenerator sub-object outputs the right
    files. """
    actual_labelled_image_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_labelled_images,
                EXPECTED_LABELLED_IMAGE_FILENAME
            )
        )
    actual_masked_image_checksum = \
        make_checksum(
            os.path.join(
                demo_obj.rec_dir_gen.path_to_masked_images,
                EXPECTED_MASKED_IMAGE_FILENAME
            )
        )
    assert actual_labelled_image_checksum == EXPECTED_LABELLED_IMAGE_CHECKSUM
    assert actual_masked_image_checksum == EXPECTED_MASKED_IMAGE_CHECKSUM

def test():
    """ Run the unit tests. """
    demo_obj = Demonstrator()
    demo_obj.make_and_run_reconstruction_dir_generator()
    check_rec_dir_gen_fields(demo_obj)
    check_rec_dir_gen_files(demo_obj)

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    test()

if __name__ == "__main__":
    run()
