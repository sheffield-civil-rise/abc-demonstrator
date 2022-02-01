import pandas as pd
import numpy as np
import geopandas as gpd

import argparse
import os
from datetime import datetime

from scipy.interpolate import interp1d

def parse_time(strtime):
    return datetime.strptime(strtime, '%H:%M:%S.%f')

def seconds_since(origin, time):
    return (time-origin).total_seconds()


def read_files(gps_path, ldb_path):
    ''' '''
    gps = pd.read_csv(gps_path, skipinitialspace=True)

    ldb = pd.read_csv(
        ldb_path, skipinitialspace=True,
        usecols=['FRAME', 'CAMERA TIME'])

    try:
        ldb.drop(ldb[ldb.FRAME.str.contains('ERROR')].index, inplace=True)
    except:
        pass

    return gps, ldb


def append_distance_from_centroid(df, inplace=False):
    ''' '''
    gdf = gpd.GeoDataFrame(
        data=df.index,
        geometry=gpd.points_from_xy(
            df['longitude'],
            df['latitude']),
        crs='epsg:4326').to_crs('epsg:27700')

    centroid = (
        gdf.geometry.x.mean(),
        gdf.geometry.y.mean(),
        df['altitude'].mean())

    xs = gdf.geometry.x.values - centroid[0]
    ys = gdf.geometry.y.values - centroid[1]
    zs = df.altitude.values - centroid[2]

    if inplace:
        df['centroid'] = [centroid for _ in range(len(df))]
        df['x'] = xs
        df['y'] = ys
        df['z'] = zs
    else:
        ndf = df.copy()
        ndf['centroid'] = [(xs, ys, zs) for _ in range(len(df))]
        ndf['x'] = xs
        ndf['y'] = ys
        ndf['z'] = zs
        return ndf


def build_rotation(heading, pitch, roll):
    ''' '''
    cosa, cosb, cosg = np.cos(heading), np.cos(pitch), np.cos(roll)
    sina, sinb, sing = np.sin(heading), np.sin(pitch), np.sin(roll)

    R = np.array([
        [cosa*cosb, cosa*sinb*sing-sina*cosg, cosa*sinb*cosg+sina*sing],
        [sina*cosb, sina*sinb*sing+cosa*cosg, sina*sinb*sing-cosa*sing],
        [-sinb, cosb*sing, cosb*cosg]])

    def theta(i):
        return (1+2*i)*np.pi/5.

    return [R @ (lambda i: np.array([
        [np.cos(theta(i)), -np.sin(theta(i)), 0.],
        [np.sin(theta(i)), np.cos(theta(i)), 0.],
        [0., 0., 1.]]))(cam) for cam in range(5)]


def append_orientation(df, inplace=False):
    ''' append camera orientations to dataframe '''
    ndf = df.copy()
    ndf['rotations'] = None  # initialise empty column
    for i, row in df.iterrows():
        Rs = build_rotation(row['heading'], row['pitch'], row['roll'])
        ndf.at[i, 'rotations'] = Rs
    if inplace:  # append rotations to input dataframe
        df['rotations'] = ndf['rotations'].values
    else:  # return copy with added rotations
        return ndf

def get_images_by_frame(dir, frame):
    ''' Returns list of images by frame(s) '''

    if type(frame) is list:
        out = {i: [] for i in frame}  # create empty lists
        for file in os.listdir(dir):
            if ex(file)[1] in frame:
                out[ex(file[1])].append(file)
    else:
        if type(frame) is int:
            frame = '%06d' % frame
        out = [file for file in os.listdir(dir) if ex(file)[1] == frame]
    return out


def interpolate(ldb, gps):
    ''' interpolate '''
    gps_cols = [
        'Latitude (deg)',
        'Longitude (deg)',
        'Altitude (m)',
        'Heading (deg)',
        'Pitch (deg)',
        'Roll (deg)']
    gps_time = 'Time (HH:mm:ss.fff)'

    ldb_cols = [s.split()[0].lower() for s in gps_cols]

    origin = parse_time(gps[gps_time][0])
    t = np.array([
        (lambda s: seconds_since(origin, s))(parse_time(st))
        for st in gps[gps_time]])

    tnew = np.array([
        (lambda s: seconds_since(origin, s))(parse_time(st))
        for st in ldb['CAMERA TIME']])

    ldb_copy = ldb.copy()
    for i, col in enumerate(gps_cols):

        y = gps[col].to_numpy()
        f_pos = interp1d(t, y, kind='linear', copy=False)

        ldb_copy[ldb_cols[i]] = f_pos(tnew)

    return ldb_copy


def log(idx):
    print([
        'loading files',
        'localising frames',
        'calculating camera orientations',
        'calculating transform from centroid',
        'writing out',
        'printing stuff'][idx])


def run(args):
    ''' run main body of code '''

    log(0)
    gps, ldb = read_files(args.gps, args.ldb)
    log(1)
    ldbi = interpolate(ldb, gps)
    log(2)
    append_orientation(ldbi, inplace=True)
    # log(3)
    # append_distance_from_centroid(ldbi, inplace=True)
    log(4)
    write_out(ldbi, args.output)


def write_out(df, path):
    df.to_csv(path, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'gps', type=str, help='imu/gnss data file')
    parser.add_argument(
        'ldb', type=str, help='ladybug frame and timestamp file')
    parser.add_argument(
        '-i', '--img_dir', type=str,
        help='directory containing ladybug processed images')
    parser.add_argument(
        '-o', '--output', type=str,
        help='output filename to create')

    args = parser.parse_args()

    if args.img_dir is None:
        args.img_dir = os.path.split(args.ldb)[0]
    if args.output is None:
        ldb_path = os.path.splitext(os.path.split(args.ldb)[1])[0]
        args.output = ldb_path + '_localised.csv'

    run(args)
