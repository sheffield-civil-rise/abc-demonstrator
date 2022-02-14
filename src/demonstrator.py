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
            timeout=config.DEFAULT_DEMO_TIMEOUT,
            check_every=config.DEFAULT_CHECK_EVERY,
            path_to_output=config.DEFAULT_PATH_TO_DEMO_OUTPUT
        ):
        self.timeout = timeout # In seconds.
        self.check_every = check_every # In seconds.
        self.path_to_output = path_to_output
        self.rec_dir_gen = None
        self.path_to_cache = os.path.join(self.path_to_output, "cache")

    def make_and_run_reconstruction_dir_generator(self):
        """ Run the generator class, deleting any existing output as
        necessary. """
        if os.path.exists(self.path_to_output):
            shutil.rmtree(self.path_to_output)
        self.rec_dir_gen = \
            ReconstructionDirGenerator(path_to_output=self.path_to_output)
        self.rec_dir_gen.generate()

    def run_batch_processes(self):
        camera_init0 = \
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_FILENAME
            )
        camera_init1 = \
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_LABEL_FILENAME
            )
        batch_pcssr = \
            BatchProcessor(
                search_recursively=True,
                path_to_output_images=self.rec_dir_gen.path_to_output_images,
                pipeline="custom",
                path_to_cache=self.path_to_cache,
                paths_to_init_files: [camera_init0, camera_init1]
                path_to_labelled_images=self.rec_dir_gen.path_to_labelled_images
            )
        recon_thread_running = batch_pcssr.function_to_run
        # Pause execution while photogrammetry running externally.
        start_time = time.time()
        while True:
            if not recon_thread_running():
                break
            if (time.time()-start_time) > self.timeout:
                raise DemonstratorException(
                    "Timed out after "+str(self.timeout)+" seconds."
                )
            time.sleep(
                self.check_every-((time.time()-start_time)%self.check_every)
            )


#    def run_batch_processes(self):
#        camera_init_0 = \
#            os.path.join(
#                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_FILENAME
#            )
#        camera_init_1 = \
#            os.path.join(
#                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_LABEL_FILENAME
#            )
#        recon_thread_running = \
#            batch_process(
#                self.rec_dir_gen.path_to_output_images,
#                "custom",
#                cache=self.path_to_cache,
#                init=[camera_init_0, camera_init_1],
#                label_dir=self.rec_dir_gen.path_to_labelled_images
#            )
#        # Pause execution while photogrammetry running externally.
#        start_time = time.time()
#        while True:
#            if not recon_thread_running():
#                break
#            if (time.time()-start_time) > self.timeout:
#                raise DemonstratorException(
#                    "Timed out after "+str(self.timeout)+" seconds."
#                )
#            time.sleep(
#                self.check_every-((time.time()-start_time)%self.check_every)
#            )

    def demonstrate(self):
        """ Run the demonstrator script. """
        self.make_and_run_reconstruction_dir_generator()
        self.run_batch_processes()
        print("THIS IS AS FAR AS THE SCRIPT SHOULD GET RIGHT NOW")
        sys.exit(0)

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class DemonstratorException(Exception):
    pass

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    demonstrator = Demonstrator()
    demonstrator.demonstrate()

if __name__ == "__main__":
    run()
