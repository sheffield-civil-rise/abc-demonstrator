"""
This code defines a class which calculates the heights of buildings.
"""

# Standard imports.
import json
import os
from dataclasses import dataclass
from typing import ClassVar

# Non-standard imports.
import open3d
import numpy
import scipy
from sklearn.decomposition import PCA

# Local imports.
from local_configs import CONFIGS

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
    ignore_roof: bool = True
    # Generated fields.
    transform: list = None
    mesh: open3d.cpu.pybind.geometry.TriangleMesh = None
    result: float = None

    # Class attributes.
    EYE_ROWS: ClassVar[int] = 4

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

    def to_homogeneous(self, transform):
        """
        `T` is a translation vector.
        `R` is a rotation matrix.
        `S` is a scale vector.
        `h` denotes a homogeneous transformation thereof.
        """
        T, R, S = transform
        Th, Rh, Sh = [
            numpy.zeros((self.EYE_ROWS, self.EYE_ROWS)) for _ in range(3)
        ]
        Th[0:self.EYE_ROWS, 0:self.EYE_ROWS] = numpy.eye(self.EYE_ROWS)
        Th[0, 3] = T[0]
        Th[1, 3] = T[1]
        Th[2, 3] = T[2]
        Rh[0:3, 0:3] = R
        Rh[3, 3] = 1
        Sh[0, 0] = numpy.sqrt(S[0])
        Sh[1, 1] = numpy.sqrt(S[1])
        Sh[2, 2] = numpy.sqrt(S[2])
        Sh[3, 3] = 1.0
        return Th, Rh, Sh

    def joint_transform(self, forward, backward, separate_scale=False):
        """
        `f` and `b` denote forwards and backwards.
        """
        Tf, Rf, Sf = self.to_homogeneous(forward)
        Tb, Rb, Sb = self.to_homogeneous(backward)
        Rb[0:3, 0:3] = scipy.linalg.inv(Rb[0:3, 0:3])
        Sb = numpy.diag(1/numpy.diag(Sb))
        S = Sf@Sb
        _s = (S[0, 0]+S[1, 1])/2
        S = _s*numpy.eye(self.EYE_ROWS)
        S[3, 3] = 1.
        Tb[0:3, 3] = -Tb[0:3, 3]
        if separate_scale:
            return Tf@(Rf@(Rb@(Tb))), S
        return Tf@(Rf@(S@(Rb@(Tb))))

    def generate_transform(self, tru_xyz, sfm_xyz):
        """ Generate joint, forward and backward transforms. """
        forward = (
            get_translation(tru_xyz),
            get_rotation(tru_xyz),
            get_scale(tru_xyz)
        )
        backward = (
            get_translation(sfm_xyz),
            get_rotation(sfm_xyz),
            get_scale(sfm_xyz)
        )
        joint = self.joint_transform(forward, backward)
        return joint, forward, backward

    def make_transform(self):
        """ Calculate the transformation between relative and reference
        spaces. """
        reference = read_sfm_as_dict(self.path_to_reference)
        reconstruction = read_sfm_as_dict(self.path_to_sfm)
        _, rcn_centers = get_geometries(reconstruction["poses"])
        _, ref_centers = get_geometries(reference["poses"], match=rcn_centers)
        self.transform, _, _ = \
            self.generate_transform(
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

def get_geometries(poses, match=None):
    """ Extract centroid and rotations of SFM poses. """
    rotations = {}
    centers = {}
    for _, pose in enumerate(poses):
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

def read_sfm_as_dict(path_to, encoding=CONFIGS.general.encoding):
    """ Read an SFM file and return dictionary with the data therein. """
    with open(path_to, "r", encoding=encoding) as sfm_file:
        result = json.load(sfm_file)
    return result

def get_rotation(points):
    """ Make the rotation matrix. """
    pca = PCA()
    pca.fit(points.T)
    result = pca.components_.T
    return result

def get_translation(points):
    """ Make the translation matrix. """
    result = numpy.mean(points, axis=1)
    return result

def get_scale(points):
    """ Make the scale matrix. """
    pca = PCA()
    pca.fit(points.T)
    result = pca.explained_variance_
    return result
