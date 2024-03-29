"""
This code defines some utility functions.
"""

# Non-standard imports.
import numpy

# Local imports.
from local_configs import CONFIGS

#############
# FUNCTIONS #
#############

def make_label_color_dict(
        label_value_dict=None,
        rgb_max=CONFIGS.general.rgb_max
    ):
    """ Ronseal. """
    if label_value_dict is None:
        label_value_dict = CONFIGS.general.label_value_dict
    result = {
        i:[int(j_) for j_ in j]
        for i, j in zip(
            label_value_dict.keys(),
            decode_color(
                numpy.linspace(
                    0, encode_color(rgb_max), len(label_value_dict)
                ).astype("int")
            )
        )
    }
    return result

def encode_color(to_encode, byte_length=CONFIGS.batch_process.byte_length):
    """ Encode a colour as an integer representation thereof. """
    to_encode = numpy.array(to_encode).astype("int")
    result = (
        (to_encode[..., 2] << byte_length*2)+
        (to_encode[..., 1] << byte_length)+
        to_encode[..., 0]
    )
    return result

def decode_color(to_decode, byte_length=CONFIGS.batch_process.byte_length):
    """ Decode an integer representation of a colour. """
    result = \
        numpy.stack(
            [
                to_decode & 0xFF,
                (to_decode & 0xFF00) >> byte_length,
                (to_decode & 0xFF0000) >> byte_length*2
            ], axis=-1
        ).astype(numpy.uint8)
    return result
