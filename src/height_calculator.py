"""
This code defines a class which calculates the heights of buildings.
"""

# Standard imports.
import json
import os
from dataclasses import dataclass, field
from typing import Callable, ClassVar

# Non-standard imports.
import open3d
import numpy
import scipy
from sklearn.decomposition import PCA

##############
# MAIN CLASS #
##############

@dataclass
class HeightCalculator:
    """ The class in question. """
    # Required fields.
    path_to_reference: str = None
    path_to_sfm: str = None # SFM = Structure From Motion
    path_to_mesh: str = None
    # Non-required fields.
    path_to_labelled_images: str = None
    ignore_roof: bool = False
    # Generated fields.
    transform: list = None
    mesh: open3d.cpu.pybind.geometry.TriangleMesh = None
    result: int = None

    def __post_init__(self):
        self.check_required_fields()
        self.make_transform()
        self.mesh = load_mesh(self.path_to_mesh)
        self.calculate()

    def check_required_fields(self):
        """ Check that the required fields have been initialised properly. """
        required_fields = {
            "path_to_reference": self.path_to_reference,
            "path_to_sfm": self.path_to_sfm,
            "path_to_mesh": self.path_to_mesh,
        }
        for field_name in required_fields.keys():
            if required_fields[field_name] is None:
                raise HeightCalculatorError(field_name+" cannot be None.")

    def make_transform(self):
        """ Calculate the transformation between relative and reference
        spaces. """
        reference = read_sfm(self.path_to_reference)
        reconstruction = read_sfm(self.path_to_sfm)
        _, rcn_centers = get_geometries(reconstruction["poses"])
        _, ref_centers = get_geometries(reference["poses"], match=rcn_centers)
        self.transform, _, _ = \
            generate_transform(
                *[
                    numpy.vstack(list(center.values())).T
                    for center in [ref_centers, rcn_centers]
                ]
            )

    def calculate_geometry(self):
        """ Get the dimensions of the building in question. """
        tmesh = self.mesh.transform(self.transform)
        vector = numpy.asarray(tmesh.vertices)
        width, height, depth = vector.max(axis=0)-vector.min(axis=0)
        return width, height, depth

    def calculate(self):
        """ Calculate the height. """
        _, result, _ = self.calculate_geometry()
        if not self.ignore_roof:
            result = result*(2/3)
        self.result = result

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class HeightCalculatorError(Exception):
    """ A custom exception. """
    pass

def get_geometries(poses, match=None):
    """ Extract centroid and rotations of SFM poses. """
    rotations = {}
    centers = {}
    for index, pose in enumerate(poses):
        id_num = int(pose["poseId"])
        if match:
            if id_num not in match.keys():
                continue
        centers[id_num] = \
            numpy.array(
                [
                    float(center)
                    for center in pose["pose"]["transform"]["center"]
                ]
            )
        rotations[id_num] = \
            numpy.reshape(
                numpy.array(
                    [
                        float(center)
                        for center in pose["pose"]["transform"]["rotation"]
                    ]
                ),
                [3, 3]
            )
    return rotations, centers

def load_mesh(path_to):
    """ Load mesh from file. """
    cwd = os.getcwd()
    mesh_base, filename = os.path.split(path_to)
    os.chdir(mesh_base)  # Change directory to read to avoid textures issues.
    result = open3d.io.read_triangle_mesh(filename)
    os.chdir(cwd)
    return result
