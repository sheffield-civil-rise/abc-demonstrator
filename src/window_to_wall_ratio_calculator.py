"""
This code defines a class which calculates the window-to-wall ratio (WWR).
"""

# Standard imports.
import os
from dataclasses import dataclass
from typing import ClassVar

# Non-standard imports.
import numpy
from PIL import Image

# Local imports.
from utility_functions import make_label_color_dict, encode_color

##############
# MAIN CLASS #
##############

@dataclass
class WindowToWallRatioCalculator:
    """ The class in question. """
    # Required fields.
    path_to_reference: str = None
    path_to_sfm: str = None # SFM = Structure From Motion
    path_to_mesh: str = None
    path_to_labelled_images: str = None
    # Generated fields.
    ensemble_ratios: list = None
    result: float = None

    # Class attributes.
    MAX_RATIO: ClassVar[float] = 0.8
    LABEL_COLOR_DICT: ClassVar[dict] = make_label_color_dict()
    DEFAULT_RESULT: ClassVar[float] = 0.4

    def __post_init__(self):
        self.check_required_fields()
        self.make_ensemble_ratios()
        self.calculate()

    def check_required_fields(self):
        """ Check that the required fields have been initialised properly. """
        required_fields = {
            "path_to_reference": self.path_to_reference,
            "path_to_sfm": self.path_to_sfm,
            "path_to_mesh": self.path_to_mesh,
            "path_to_labelled_images": self.path_to_labelled_images
        }
        for field_name in required_fields:
            if required_fields[field_name] is None:
                raise WWRCalculatorError(field_name+" cannot be None.")

    def calculate_wwr_for_image(self, image_file):
        """ Calculate the window-to-wall ratio for a single image. """
        dim = encode_color(numpy.asarray(image_file))
        window = len(dim[dim==encode_color(self.LABEL_COLOR_DICT["window"])])
        wall = len(dim[dim==encode_color(self.LABEL_COLOR_DICT["wall"])])
        if (window > wall) or (wall == 0):
            return numpy.nan
        result = window/wall
        return result

    def make_ensemble_ratios(self):
        """ Get a list of WWRs, from which to take an average. """
        result = []
        for filename in os.listdir(self.path_to_labelled_images):
            path_to = os.path.join(self.path_to_labelled_images, filename)
            with Image.open(path_to) as image_file:
                ratio = self.calculate_wwr_for_image(image_file)
            if ratio < self.MAX_RATIO:
                result.append(ratio)
        self.ensemble_ratios = result

    def calculate(self):
        """ Calculate the WWR. """
        prelim = numpy.nanmedian(self.ensemble_ratios)
        if numpy.isnan(prelim):
            self.result = self.DEFAULT_RESULT
        else:
            self.result = prelim

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class WWRCalculatorError(Exception):
    """ A custom exception. WWR = Window-to-Wall Ratio. """
