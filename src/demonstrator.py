"""
This code defines a class which demonstrates what the codebase can do.
"""

# Standard imports.
import argparse
import os
import shutil
import sys

# Non-standard imports.
import numpy

# Local imports.
import config
from batch_processor import BatchProcessor
from calculate_height import main as calculate_height
from calculate_wwr import calculate as calculate_wwr
from generate_idf import main as generate_energy_model
from reconstruction_dir_generator import ReconstructionDirGenerator

##############
# MAIN CLASS #
##############

class Demonstrator:
    """ The class in question. """
    def __init__(
            self,
            path_to_output=config.DEFAULT_PATH_TO_DEMO_OUTPUT
        ):
        self.path_to_output = path_to_output
        self.path_to_cache = os.path.join(self.path_to_output, "cache")
        # Generated fields.
        self.rec_dir_gen = None
        self.batch_processor = None

    def make_and_run_reconstruction_dir_generator(self):
        """ Run the generator object, deleting any existing output as
        necessary. """
        if os.path.exists(self.path_to_output):
            shutil.rmtree(self.path_to_output)
        self.rec_dir_gen = \
            ReconstructionDirGenerator(path_to_output=self.path_to_output)
        self.rec_dir_gen.generate()

    def make_and_run_batch_processor(self):
        """ Build the batch processor object, and then run it. """
        camera_init0 = \
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_FILENAME
            )
        camera_init1 = \
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_LABEL_FILENAME
            )
        self.batch_processor = \
            BatchProcessor(
                search_recursively=True,
                path_to_output_images=self.rec_dir_gen.path_to_output_images,
                pipeline="custom",
                path_to_cache=self.path_to_cache,
                paths_to_init_files=[camera_init0, camera_init1],
                path_to_labelled_images=self.rec_dir_gen.path_to_labelled_images
            )
        batch_processor.run()

    def demonstrate(self):
        """ Run the demonstrator script. """
        self.make_and_run_reconstruction_dir_generator()
        self.run_batch_processes()
        print("THIS IS AS FAR AS THE SCRIPT SHOULD GET RIGHT NOW")
        sys.exit(0)

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    demonstrator = Demonstrator()
    demonstrator.demonstrate()

if __name__ == "__main__":
    run()
