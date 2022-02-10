import sys
import os
import argparse
import time
import numpy as np

from generate_recon_dir import autogenerate as generate_recon_dir
from batch_process import run as batch_process
from calculate_height import main as calculate_height
from calculate_wwr import calculate as calculate_wwr

from generate_idf import main as generate_energy_model

import shutil

import config
from reconstruction_dir_generator import ReconstructionDirGenerator

TIME_TOO_LONG = 7200  # seconds (2 hours)

def generate_reconstruction_dir(
        path_to_output=config.DEFAULT_PATH_TO_DEMO_OUTPUT
    ):
    """ Run the generator class, deleting any existing output as necessary. """
    if os.path.exists(path_to_output):
        shutil.rmtree(path_to_output)
    rec_dir_gen = ReconstructionDirGenerator(path_to_output=path_to_output)
    result = rec_dir_gen.generate()
    return result

def run(args):


    class Attribute:
        pass

    args.id = os.path.splitext(os.path.split(args.polygon)[-1])[0]
    if args.wd is None:
        args.wd = os.path.abspath(args.id)

    args_0 = Attribute()
    args_0.gps = args.gps
    args_0.ldb = args.ldb
    args_0.dir = args.dir
    args_0.out = os.path.join(args.wd, 'working_dir')

    args_0.polygon = os.path.abspath(args.polygon)

    generate_reconstruction_dir()
    wd_path = r"G:\photogrammetry_output_demo"

    #generate_recon_dir(args_0)
    #wd_path = r"G:\demonstrator_output\working_dir"

    image_dir = os.path.join(wd_path, 'images')
    label_dir = os.path.join(wd_path, 'labels')
    camera_init_0 = os.path.join(wd_path, 'cameraInit_label.sfm')
    camera_init_1 = os.path.join(wd_path, 'cameraInit.sfm')

    cache_dir = os.path.abspath(os.path.join(args.wd, 'cache'))

    print('caching at {}'.format(cache_dir))

    recon_thread_running = batch_process( # As of 10 Feb 2022, this is where the script crashes.
        image_dir,
        'custom',
        cache=cache_dir,
        init=[camera_init_0, camera_init_1],
        label_dir=label_dir)


    ## Pause execution while photogrammetry running externally
    starttime = time.time()
    while True:
        if not recon_thread_running():
            break
        if (time.time() - starttime) > TIME_TOO_LONG:
            # if it takes longer than two hours fail
            raise RuntimeError("this is taking too long, i'm giving up")
        time.sleep(30.0 - ((time.time() - starttime) % 30.0))  # check every 30s


    sfm_base = os.path.join(cache_dir, 'SfMTransfer')
    sfm_base = os.path.join(sfm_base, os.listdir(sfm_base)[-1])

    mesh_base = os.path.join(cache_dir, 'Texturing')
    mesh_base = os.path.join(mesh_base, os.listdir(mesh_base)[0])


    args_1 = Attribute()
    args_1.ref = camera_init_1
    args_1.sfm = os.path.join(sfm_base, 'cameras.sfm')
    args_1.mesh = os.path.join(mesh_base, 'texturedMesh.obj')
    args_1.dir = label_dir

    height = calculate_height(args_1)

    wwr = calculate_wwr(args_1)

    print(wwr)
    args_2 = Attribute()
    args_2.id = args.id
    args_2.init = r'src\starting_point.idf'
    args_2.height = np.max([7, 2 * height / 3])
    args_2.wwr = {0: wwr, 90: 0.2, 180: wwr, 270: 0.2}
    args_2.polygon = args.polygon
    args_2.output = os.path.join(args.wd, args.id + '_autogenerate.idf')
    args_2.outdir = os.path.join(args.wd, args.id + '_autogenerate')

    generate_energy_model(args_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('gps', help='gps file')
    parser.add_argument('ldb', help='ldb file')
    parser.add_argument('dir', help='ladybug images')
    parser.add_argument('polygon', help='polygon file')
    parser.add_argument('--wd', help='working directory')
    parser.add_argument('--id', help='identifier')

    args = parser.parse_args()

    run(args)