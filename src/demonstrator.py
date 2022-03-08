"""
This code defines a class which demonstrates what the codebase can do.
"""

# Standard imports.
import logging
import os
import pathlib
import shutil
import subprocess

# Local imports.
from config import (
    get_configs,
    make_path_to_gps_data,
    make_path_to_ladybug_gps_data,
    make_path_to_ladybug_images,
    INTERNAL_PYTHON_COMMAND
)
from height_calculator import HeightCalculator
from reconstruction_dir_generator import ReconstructionDirGenerator
from window_to_wall_ratio_calculator import WindowToWallRatioCalculator

# Local constants.
CONFIGS = get_configs()

##############
# MAIN CLASS #
##############

class Demonstrator:
    """ The class in question. """
    def __init__(
            self,
            path_to_input_override=None, # Overides several configs if set.
            path_to_output=CONFIGS.general.path_to_output,
            path_to_polygon=CONFIGS.general.path_to_polygon,
            debug=False
        ):
        self.path_to_input_override = path_to_input_override
        self.path_to_output = path_to_output
        self.path_to_polygon = path_to_polygon
        self.path_to_cache = os.path.join(self.path_to_output, "cache")
        self.debug = debug
        # Generated fields.
        self.rec_dir_gen = None
        self.paths_to_init_files = None
        self.batch_process = None
        self.height_calculator = None
        self.window_to_wall_ratio_calculator = None
        self.energy_model_process = None
        self.start_logging()

    def start_logging(self):
        """ Configure logging, and log that we've started. """
        log_level = logging.INFO
        if self.debug:
            log_level = logging.DEBUG
        logging.basicConfig(
            level=log_level,
            format=CONFIGS.general.logging_format
        )
        logging.info("Initiating "+str(self.__class__.__name__)+" object...")
        if self.debug:
            logging.info("Switched to DEBUG mode.")

    def run_subprocess(self, arguments, timeout=None):
        """ Run a given subprocess - quietly or otherwise. """
        if self.debug:
            result = \
                subprocess.run(
                    arguments,
                    check=True,
                    timeout=timeout
                )
        else:
            result = \
                subprocess.run(
                    arguments,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout
                )
        return result

    def make_and_run_reconstruction_dir_generator(self):
        """ Run the generator object, deleting any existing output as
        necessary. """
        if os.path.exists(self.path_to_output):
            shutil.rmtree(self.path_to_output)
        if self.path_to_input_override:
            path_to_gps_data = \
                make_path_to_gps_data(stem=self.path_to_input_override)
            path_to_ladybug_gps_data = \
                make_path_to_ladybug_gps_data(stem=self.path_to_input_override)
            path_to_ladybug_images = \
                make_path_to_ladybug_images(stem=self.path_to_input_override)
            self.rec_dir_gen = \
                ReconstructionDirGenerator(
                    path_to_gps_data=path_to_gps_data,
                    path_to_ladybug_gps_data=path_to_ladybug_gps_data,
                    path_to_ladybug_images=path_to_ladybug_images,
                    path_to_output=self.path_to_output,
                    path_to_polygon=self.path_to_polygon
                )
        else:
            self.rec_dir_gen = \
                ReconstructionDirGenerator(
                    path_to_output=self.path_to_output,
                    path_to_polygon=self.path_to_polygon
                )
        self.rec_dir_gen.generate()
        self.make_paths_to_init_files()

    def make_paths_to_init_files(self):
        """ Make the paths to these special files. """
        self.paths_to_init_files = [
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_LABEL_FILENAME
            ),
            os.path.join(
                self.path_to_output, self.rec_dir_gen.CAMERA_INIT_FILENAME
            )
        ]

    def make_and_run_batch_process(self):
        """ Build the batch process, and run it. """
        path_to_py_file = \
            str(pathlib.Path(__file__).parent/"batch_processor.py")
        if len(self.paths_to_init_files) >= 2:
            path_to_init_file_a = self.paths_to_init_files[0]
            path_to_init_file_b = self.paths_to_init_files[1]
        elif len(self.paths_to_init_files) == 1:
            path_to_init_file_a = self.paths_to_init_files[0]
            path_to_init_file_b = None
        else:
            path_to_init_file_a = None
            path_to_init_file_b = None
        path_to_labelled_images = self.rec_dir_gen.path_to_labelled_images
        arguments = [
            INTERNAL_PYTHON_COMMAND, path_to_py_file,
            "--path-to-output-images", self.rec_dir_gen.path_to_output_images,
            "--path-to-cache", self.path_to_cache,
            "--path-to-init-file-a", path_to_init_file_a,
            "--path-to-init-file-b", path_to_init_file_b,
            "--path-to-labelled-images", path_to_labelled_images
        ]
        self.batch_process = \
            self.run_subprocess(
                arguments, timeout=CONFIGS.batch_process.timeout
            )

    def make_and_run_height_calculator(self):
        """ Build the height calculator object - it runs on its own. """
        sfm_base = os.path.join(self.path_to_cache, "SfMTransfer")
        sfm_base = os.path.join(sfm_base, os.listdir(sfm_base)[-1])
        mesh_base = os.path.join(self.path_to_cache, "Texturing")
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

    def make_and_run_window_to_wall_ratio_calculator(self):
        """ Build the window-to-wall ratio calculator object - it runs on its
        own. """
        self.window_to_wall_ratio_calculator = \
            WindowToWallRatioCalculator(
                path_to_reference=self.height_calculator.path_to_reference,
                path_to_sfm=self.height_calculator.path_to_sfm,
                path_to_mesh=self.height_calculator.path_to_mesh,
                path_to_labelled_images=self.rec_dir_gen.path_to_labelled_images
            )

    def make_and_run_energy_model_process(self):
        """ Build the energy model generator object, and then run it. """
        path_to_py_file = \
            str(pathlib.Path(__file__).parent/"energy_model_generator.py")
        path_to_output_idf = \
            os.path.join(self.path_to_output, "output.idf")
        path_to_output_dir = \
            os.path.join(self.path_to_output, "energy_model_output")
        arguments = [
            INTERNAL_PYTHON_COMMAND, path_to_py_file,
            "--height", str(self.height_calculator.result),
            "--wwr", str(self.window_to_wall_ratio_calculator.result),
            "--path-to-output-idf", path_to_output_idf,
            "--path-to-output-dir", path_to_output_dir,
            "--path-to-polygon", self.path_to_polygon
        ]
        self.energy_model_process = self.run_subprocess(arguments)

    def demonstrate(self):
        """ Run the demonstrator script. """
        logging.info("Demonstation initiated.")
        logging.info("Running reconstruction dir generator...")
        self.make_and_run_reconstruction_dir_generator()
        logging.info("Running batch process...")
        self.make_and_run_batch_process()
        logging.info("Running height calculator...")
        self.make_and_run_height_calculator()
        logging.info("Running window-to-wall ratio calculator...")
        self.make_and_run_window_to_wall_ratio_calculator()
        logging.info("Running energy model process...")
        self.make_and_run_energy_model_process()
        logging.info("Demonstration complete.")

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    demonstrator = Demonstrator()
    demonstrator.demonstrate()

if __name__ == "__main__":
    run()
