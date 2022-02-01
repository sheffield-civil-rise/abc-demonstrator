from localise_ladybug import run as localise_ladybug
from label_dir import main as label_data

from batch_process import run as batch_photogrammetry

import argparse
import os


def main():

    # get files

    # create localised directory with orientations and positions
    localise_ladybug(args)

    # copy select images to directory


    # label images / create masks
    label_data(args)

    # construct cameraInit SFM file with masks and orientations
    # create_cameraInit()

    # run photogrammetry
    batch_photogrammetry(args)

    # project labelling


    # feature extraction


    # energy plus [?]

if __name__ == '__main__':
    main()
