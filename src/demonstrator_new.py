"""
This code defines a class which demonstrates what the codebase can do.
"""

# Standard imports.
import argparse
import os
import shutil
import sys
import time

# Non-standard imports.
import numpy

# Local imports.
import config
from batch_process import run as batch_process
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
            timeout=config.DEFAULT_DEMO_TIMEOUT,
            check_every=config.DEFAULT_CHECK_EVERY,
            path_to_output=config.DEFAULT_PATH_TO_DEMO_OUTPUT
        ):
        self.timeout = timeout # In seconds.
        self.check_every = check_every
        self.path_to_output = path_to_output
        self.rec_dir_gen = None
        self.batch_processor = None

    def make_and_run_reconstruction_dir_generator(self):
        """ Run the generator class, deleting any existing output as
        necessary. """
        if os.path.exists(self.path_to_output):
            shutil.rmtree(self.path_to_output)
        self.rec_dir_gen = \
            ReconstructionDirGenerator(path_to_output=self.path_to_output)
        self.rec_dir_gen.generate()

    def demonstrate(self):
        """ Run the demonstrator script. """
        self.make_and_run_reconstruction_dir_generator()
        print("THIS IS AS FAR AS THE SCRIPT SHOULD GET RIGHT NOW")
        sys.exit(0)

#############
# FUNCTIONS #
#############

#def run_batch_processes():
#    recon_thread_running = \
#        batch_process(
#            image_dir,
#            'custom',
#            cache=cache_dir,
#            init=[camera_init_0, camera_init_1],
#            label_dir=label_dir
#        )

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    demonstrator = Demonstrator()
    demonstrator.demonstrate()

if __name__ == "__main__":
    run()
