"""
This code defines a class which demonstrates what the codebase can do.
"""

# Standard imports.
import os
import shutil
from multiprocessing import Pool

# Local imports.
from batch_processor import BatchProcessor
from config import (
    get_configs,
    make_path_to_gps_data,
    make_path_to_ladybug_gps_data,
    make_path_to_ladybug_images,
)
from energy_model_generator import EnergyModelGenerator
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
            path_to_output=CONFIGS.general.path_to_demo_output,
            path_to_polygon=CONFIGS.general.path_to_polygon
        ):
        self.path_to_input_override = path_to_input_override
        self.path_to_output = path_to_output
        self.path_to_polygon = path_to_polygon
        self.path_to_cache = os.path.join(self.path_to_output, "cache")
        # Generated fields.
        self.rec_dir_gen = None
        self.paths_to_init_files = None
        self.batch_processor = None
        self.height_calculator = None
        self.window_to_wall_ratio_calculator = None
        self.energy_model_generator = None

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

    def make_and_run_batch_processor(self):
        """ Build the batch processor object, and then run it. """
        arg_list = [
            self.rec_dir_gen.path_to_output_images,
            self.path_to_cache,
            self.paths_to_init_files,
            self.rec_dir_gen.path_to_labelled_images
        ]
        pool = Pool()
        pool.map(make_and_run_batch_processor_quietly, arg_list)

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

    def make_and_run_energy_model_generator(self):
        """ Build the energy model generator object, and then run it. """
        wwr = self.window_to_wall_ratio_calculator.result
        path_to_output_idf = \
            os.path.join(self.path_to_output, "output.idf")
        path_to_output_dir = \
            os.path.join(self.path_to_output, "energy_model_output")
        self.energy_model_generator = \
            EnergyModelGenerator(
                height=self.height_calculator.result,
                window_to_wall_ratio=wwr,
                path_to_output_idf=path_to_output_idf,
                path_to_output_dir=path_to_output_dir,
                path_to_polygon=self.path_to_polygon
            )
        self.energy_model_generator.generate_and_run()

    def demonstrate(self):
        """ Run the demonstrator script. """
        self.make_and_run_reconstruction_dir_generator()
        self.make_and_run_batch_processor()
        self.make_and_run_height_calculator()
        self.make_and_run_window_to_wall_ratio_calculator()
        self.make_and_run_energy_model_generator()

def make_and_run_batch_processor_quietly(arg_list):
    batch_processor = \
        BatchProcessor(
            search_recursively=True,
            path_to_output_images=arg_list[0],
            pipeline="custom",
            path_to_cache=arg_list[1],
            paths_to_init_files=arg_list[2],
            path_to_labelled_images=arg_list[3]
        )
    batch_processor.start()

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    demonstrator = Demonstrator()
    demonstrator.demonstrate()

if __name__ == "__main__":
    run()
