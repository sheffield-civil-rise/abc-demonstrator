"""
This code defines a series of tests which check the output - both files and the
fields of sub-objects - of the Demonstrator class.
"""

# Standard imports.
import os

# Local imports.
import config
from demonstrator import Demonstrator

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

def test():
    """ Run the unit tests. """
    demo_obj = Demonstrator()
    demo_obj.make_and_run_reconstruction_dir_generator()
    test_rec_dir_gen_fields(demo_obj)

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    test()

if __name__ == "__main__":
    run()
