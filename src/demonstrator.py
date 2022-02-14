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
from height_calculator import HeightCalculator
from reconstruction_dir_generator import ReconstructionDirGenerator

from o_calculate_height import main as calculate_height
from calculate_wwr import calculate as calculate_wwr
from generate_idf import main as generate_energy_model

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
        self.paths_to_init_files = None
        self.batch_processor = None
        self.height_calculator = None

    def make_paths_to_init_files(self):
        """ Make the paths to these special files. """
        self.paths_to_init_files = [
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_FILENAME
            ),
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_LABEL_FILENAME
            )
        ]
        return result

    def make_and_run_reconstruction_dir_generator(self):
        """ Run the generator object, deleting any existing output as
        necessary. """
        if os.path.exists(self.path_to_output):
            shutil.rmtree(self.path_to_output)
        self.rec_dir_gen = \
            ReconstructionDirGenerator(path_to_output=self.path_to_output)
        self.rec_dir_gen.generate()
        self.make_paths_to_init_files()

    def make_and_run_batch_processor(self):
        """ Build the batch processor object, and then run it. """
        self.batch_processor = \
            BatchProcessor(
                search_recursively=True,
                path_to_output_images=self.rec_dir_gen.path_to_output_images,
                pipeline="custom",
                path_to_cache=self.path_to_cache,
                paths_to_init_files=self.paths_to_init_files,
                path_to_labelled_images=self.rec_dir_gen.path_to_labelled_images
            )
        self.batch_processor.run()

    def make_and_run_height_calculator(self):
        """ Build the height calculator object - it runs on its own. """
        sfm_base = os.path.join(cache_dir, "SfMTransfer")
        sfm_base = os.path.join(sfm_base, os.listdir(sfm_base)[-1])
        mesh_base = os.path.join(cache_dir, "Texturing")
        mesh_base = os.path.join(mesh_base, os.listdir(mesh_base)[0])
        path_to_reference = self.paths_to_init_files[1]
        path_to_sfm = os.path.join(sfm_base, "cameras.sfm")
        path_to_mesh = os.path.join(mesh_base, "texturedMesh.obj")
        self.height_calculator = \
            HeightCalculator(
                path_to_reference=path_to_reference,
                path_to_sfm=path_to_sfm,
                path_to_mesh=path_to_mesh
            )

    def demonstrate(self):
        """ Run the demonstrator script. """
        self.make_and_run_reconstruction_dir_generator()
        self.make_and_run_batch_processor()
        self.make_and_run_height_calculator()
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
