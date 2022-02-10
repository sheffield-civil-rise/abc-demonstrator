"""
This code defines a script which demonstrates what the codebase can do.
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

#############
# FUNCTIONS #
#############

def make_and_run_reconstruction_dir_generator(
        path_to_output=config.DEFAULT_PATH_TO_DEMO_OUTPUT
    ):
    """ Run the generator class, deleting any existing output as necessary. """
    if os.path.exists(path_to_output):
        shutil.rmtree(path_to_output)
    result = ReconstructionDirGenerator(path_to_output=path_to_output)
    result.generate()
    return result

def run():
    rec_dir_gen = make_and_run_reconstruction_dir_generator()
    print("THIS IS AS FAR AS THE SCRIPT SHOULD GET RIGHT NOW")
    sys.exit(0)

if __name__ == "__main__":
    run()
