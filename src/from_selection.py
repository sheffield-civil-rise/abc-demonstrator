from fix_rig_setup import re, ex, np, sw
from shutil import copyfile

import pandas as pd
# import geopandas as gpd

import os
import sys
import argparse


cameras = [0, 1, 2, 3, 4, 5]


def prog(k, N):
    fstr = '\rcopied %'+str(sw(N))+'d/%d files'
    sys.stdout.write(fstr % (k+1, N))
    sys.stdout.flush()


def read_selection(filename):
    # return gpd.GeoDataFrame(pd.read_csv(filename))
    return pd.read_csv(filename)


def convert(filename):

    opath, filename = os.path.split(filename)
    fn, ext = os.path.splitext(filename)

    cam = fn[56]
    dt, ix, ui = ex(fn)


    if re(dt, ix, int(cam), ui) != fn:
        raise ValueError

    return cam, np(dt, ix, ui)+ext


def get_full_paths(df, base_dir=''):
    for i, row in df.iterrows():
        df.loc[i, 'full_path'] = os.path.abspath(
            os.path.join(base_dir, 'rig', *convert(row['image'])))


def copy_subset(df, folder=''):
    if len(folder) == 0:
        folder = 'selection'
    for cam in cameras:
        newdir = os.path.join(folder, 'rig', str(cam))
        if not os.path.exists(newdir):
            os.makedirs(newdir)

    for i, row in df.iterrows():
        path, fn = os.path.split(row['full_path'])
        _, cam = os.path.split(path)
        # print(
        #     row['full_path'] + ' ---> ' +
        #     os.path.join(folder, 'rig', cam, fn))
        print(row['image'])
        print(os.path.join(folder, 'rig', cam, fn))

        copyfile(
            row['image'],
            # row['full_path'],
            os.path.join(folder, 'rig', cam, fn))
        prog(i, len(df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--base_dir", type=str, help="location of full dataset")
    parser.add_argument(
        "-o", "--out_dir", type=str,
        help="location of data subset (automatically same as selection file)")
    parser.add_argument(
        "file", type=str, help="csv file containing subset")

    args = parser.parse_args()
    sel_file = os.path.abspath(args.file)

    if args.base_dir is None:
        base_dir = os.path.abspath('.')
    else:
        base_dir = args.base_dir

    if args.out_dir is not None:
        out_dir = os.path.join('tmp', 'selection')
    else:
        sel_name = os.path.splitext(os.path.split(sel_file)[-1])[0]
        out_dir = os.path.join('tmp', sel_name)

    df = read_selection(sel_file)
    get_full_paths(df, base_dir=base_dir)

    # print(df.loc[0,'full_path'])
    copy_subset(df, out_dir)
