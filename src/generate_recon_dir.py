from deeplab.Deeplabv3 import Deeplabv3

import argparse

import pandas as pd
import numpy as np
import geopandas as gpd

import re
import json

import os
from datetime import datetime

from scipy.interpolate import interp1d
from shapely.geometry import Point, Polygon

import shutil
from PIL import Image
import cv2
import sys


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
    heading, pitch, roll = [(lambda d: np.pi*d/180)(d) for d in [heading, pitch, roll]]

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


def localise(gps, ldb):
    gps, ldb = read_files(gps, ldb)
    ldbi = interpolate(ldb, gps)
    append_orientation(ldbi, inplace=True)
    return ldbi


def as_gdf(df):
    gdf = gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(
            df['longitude'],
            df['latitude']),
        crs='epsg:4326')
    return gdf


def expand(df):
    columns = ['FRAME', 'CAMERA TIME', 'latitude', 'longitude', 'altitude', 'heading', 'pitch', 'roll']
    ndf = pd.DataFrame(columns=columns)
    print(ndf)
    geox, geoy = [], []
    for i, row in df.iterrows():
        rotations = row['rotations']
        nrow = row[columns]
        for cam in range(len(rotations)):
            nrow['cam'] = int(cam)
            nrow['rotation'] = rotations[cam]
#             nrow['geometry'] = Point(row.geometry.x, row.geometry.y)
            if type(df) is not gpd.GeoDataFrame:
                geox.append(row['longitude'])
                geoy.append(row['latitude'])
            else:
                geox.append(row.geometry.x)
                geoy.append(row.geometry.y)
            ndf = ndf.append(nrow, ignore_index=True)
    if type(df) is not gpd.GeoDataFrame:
         gdf = gpd.GeoDataFrame(
            data=ndf,
            geometry=gpd.points_from_xy(
                geox, geoy),
            crs='epsg:4326')
    else:
        gdf = gpd.GeoDataFrame(data=ndf, geometry=gpd.points_from_xy(geox, geoy), crs=df.crs)
    return gdf


def find_directions(heading, cam):
    ''' '''
    heading = np.pi * heading / 180.
    def theta(i):
#         return np.pi + (1+2*i)*np.pi/5
        return 2*np.pi - (1+2*i)*np.pi/5.

    th = (theta(cam) + heading)
    if th >= 2*np.pi:
        return th - 2*np.pi
    elif th < 0:
        return th + 2*np.pi
    else:
        return th


def seg(v0, vd=None, t0=None, dist=1, fov=np.pi/2, res=20):

    if vd is None:
        if t0 is None:
            error('no vector or angle')
    else:
        vd = vd / np.linalg.norm(vd)
        t0 = np.arctan2(vd[1], vd[0])

    p = [(v0[0] + dist*np.cos(t), v0[1] + dist*np.sin(t)) for t in np.linspace(t0 - fov/2, t0 + fov/2, res)]
    return Polygon([v0, *p])


def find_views(df, dist = 20, fov=np.pi/2):

    xdf = df.to_crs('epsg:27700')

    def create_seg(geo, heading, cam):
        v0 = np.array([geo.x, geo.y])
        th = find_directions(heading, cam)
        return seg(v0, t0=th, dist=dist, fov=fov)

    df['view'] = gpd.GeoSeries(
        data=xdf.apply(lambda r: create_seg(r.geometry, r.heading, r.cam), axis=1).tolist(),
        crs='epsg:27700').to_crs('epsg:4326')

    return df


def create_circle(centroid, radius=1, crs='epsg:4326', res=100, aspoints=False):
    c = Point(centroid)
    cdf = gpd.GeoDataFrame({'geometry': [c]}, crs=crs).to_crs('epsg:27700')

    points = [(radius*np.sin(t) + cdf.geometry.x[0], radius*np.cos(t) + cdf.geometry.y[0]) for t in np.linspace(0, 2*np.pi, res)]

    poly = Polygon(points)
    out = gpd.GeoDataFrame({'geometry': [Point(p) for p in points] if aspoints else [poly]}, crs='epsg:27700').to_crs(crs)
    return out


def filter_by_view(df, centroid):
    if type(centroid) is not Point:
        centroid = Point(*centroid)

    inview = df.apply(lambda r: centroid.within(r['view']), axis=1)
    return df[inview]


def calculate_focalpoint(polygon):
    ''' '''
    if polygon is None:
        return [-1.5120831368308705, 53.35550826329699]
    else:
        df = pd.read_csv(polygon, header=None)
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(df[1], df[0]),
            crs='epsg:27700').to_crs('epsg:4326')
        _centroid = Polygon(gdf.geometry).centroid
        return [_centroid.x, _centroid.y]


def get_views_by_centroid(gdf, centroid, radius=20, view_dist=40, fov=np.pi/2):

    circ = create_circle(centroid, radius=radius, aspoints=False)

    subset = gdf.overlay(circ, how='intersection')
    full_frame = expand(subset)
    view_frame = find_views(full_frame, dist=view_dist, fov=fov)

    return filter_by_view(view_frame, centroid)


def get_filepaths(df, filedict):
    xdf = df.copy()
    xdf['path'] = df.apply(lambda r: filedict[int(r['cam'])][r['FRAME']], axis=1)
    return xdf


def create_intrinsic(cam):
     return {
        "intrinsicId": str(cam),
        "width": "2048",
        "height": "2464",
        "sensorWidth": "-1",
        "sensorHeight": "-1",
        "serialNumber": str(cam),
        "type": "radial3",
        "initializationMode": "unknown",
        "pxInitialFocalLength": "-1",
        "pxFocalLength": "1244.8954154464859",
        "principalPoint": [
            "1009.5173249704193",
            "1249.8800458307971"
        ],
        "distortionParams": [
            "-0.32180396146710027",
            "0.13839127303452359",
            "-0.032403246994346345"
        ],
            "locked": "0"}


def create_view(row, base_dir=""):
#     index = row['FRAME'] + '_' + str(int(row['cam']))
    index = str(int(row['FRAME'])*10 + int(row['cam']))
    return {
        "viewId": index,#['id']),
        "poseId": index,#['id']),
        "intrinsicId": str(int(row['cam'])),
#         "resectionId": "",
        "path": os.path.join(base_dir, row['path']),
        "width": "2048",
        "height": "2464",
        "metadata": ""}


def build_init_dict(df, base_dir=""):
    """ """
    intrinsics = [create_intrinsic(cam) for cam in range(5)]
    views = []
    poses = []

    for i, row in df.iterrows():
        views.append(create_view(row, base_dir))
        poses.append(create_pose(row))

    camera_init = {
        "version": ["1","0","0"],
        "views": views,
        "intrinsics": intrinsics,
        "poses": poses}

    return camera_init


def generate_cameraInit(df, img_dir, output=None):
    print('generating camera init')
    init_dict = build_init_dict(df, os.path.abspath(img_dir))

    if output is None:
        return json.dumps(init_dict, indent=2)
    else:
        with open(output, 'w') as fid:
            json.dump(init_dict, fid, indent=2)


def create_pose(row):
    index = str(int(row['FRAME'])*10 + int(row['cam']))
    return {
        "poseId": index,#['id']),
        "pose": {
            "transform": {
#                 "rotation":re.findall(r"-?[\d.]+(?:e-?\d+)?", row['rotation']),
                "rotation": [str(v) for v in row['rotation'].ravel()],
                "center": [
                    str(row['local'].x),
                    str(row['local'].y),
                    str(row['altitude'])]
#                 "center": [
#                     str(row['latitude']),
#                     str(row['longitude']),
#                     str(row['altitude'])]
            },
        "locked": "1"}}


def generate_filedict(img_dir):
    filelist = os.listdir(img_dir)
    camdict = [{}, {}, {}, {}, {}]

    def framecam(fn):
        return fn[46:52], fn[56]

    for i, file in enumerate(filelist):
        frame, cam = framecam(file)

        if int(cam) < 5:
            camdict[int(cam)][frame] = file

    return camdict


def generate_local_coords(df, centroid):
    proj = 'epsg:27700'
    df = df.copy()

    if type(df) is not gpd.GeoDataFrame:
        raise TypeError('expected df to be GeoDataFrame')
    local_centroid = gpd.GeoSeries(gpd.points_from_xy([centroid[0]], [centroid[1]]), crs='epsg:4326').to_crs(proj)
    tx, ty = local_centroid.geometry.x, local_centroid.geometry.y

    df['local'] = gpd.GeoSeries(
        data=df.to_crs(proj).translate(-tx, -ty).tolist(),
        index=df.index,
        crs=proj)
    return df


def generate_working_directory(df, img_dir, out_dir):
    ''' copy files create working environment '''
    if os.path.isdir(out_dir):
        raise ValueError('out directory already exists')

    os.makedirs(out_dir)

    new_img_dir = os.path.join(out_dir, 'images')
    os.mkdir(new_img_dir)

    for i, row in df.iterrows():
        im = Image.open(os.path.join(img_dir, row['path']))
        im.transpose(Image.ROTATE_270).save(
            os.path.join(new_img_dir, row['path']))


def encode(x):
    _x = np.array(x).astype('int')
    return (_x[..., 2] << 16) + (_x[..., 1] << 8) + _x[..., 0]


def decode(x):
    return np.stack([x & 0xFF, (x & 0xFF00) >> 8, (x & 0xFF0000) >> 16], axis=-1).astype(np.uint8)


def get_img_paths(path, recursive=True):
    img_paths = []
    dir = os.listdir(path)

    for _path in dir:
        if recursive and os.path.isdir(os.path.join(path, _path)):
            img_paths = img_paths + get_img_paths(
                os.path.join(path, _path), True)
        elif not os.path.isdir(os.path.join(path, _path)):
            _, ext = os.path.splitext(_path)
            if ext in ('.jpg', '.jpeg', '.JPG', '.JPEG', '.exr'):
                img_paths.append(os.path.join(path, _path))
    return img_paths


class DigitMapToBGR:
    """
    The class aims to convert the output from the model
    to the BGR mask image. """
    def __init__(self, palette, digit_map):
        self.digit_map = digit_map
        self.palette = palette

    def digit_to_color(self, h, w, output_mask):
        maximum_channel = self.get_maximum_channel(self.digit_map[h, w])
        color = self.palette[int(maximum_channel)]
        output_mask[h, w] = color
        return output_mask

    def get_maximum_channel(self, channel_vector):
        return list(channel_vector).index(max(list(channel_vector)))

    def __call__(self):
        height, weight, channel = self.digit_map.shape
        output_bgr = np.zeros([height, weight, 3])
        # print(output_bgr.shape)
        for h in range(height):
            for w in range(weight):
                output_bgr = self.digit_to_color(h, w, output_bgr)
        return output_bgr


def get_pallete():
    ''' generate label pallete '''

    label = ['background', 'chimney', 'door', 'window', 'roof', 'wall']

    label_color_dic = {
        i:[int(j_) for j_ in j]
        for i,j in zip(
            label,
            decode(np.linspace(0, encode([255,192,128]), 6).astype('int')))}

    label_value_dic = {'background': 0,
                   'chimney': 3,
                   'door': 5,
                   'window': 4,
                   'roof': 2,
                   'wall': 1}

    pallete = {
        label_value_dic[label]: np.flip(np.array(label_color_dic[label]))
        for label in label_value_dic.keys()}
    return label_color_dic, pallete


    label_color_dic, pallete = get_pallete()

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model_dir = r'G:\binaries\Deeplabv3plus-xception-ce.hdf5'

    img_list = get_img_paths(directory)

    img_shape = (1024, 1024)
    model = Deeplabv3(
        weights=None, input_shape=(*img_shape, 3),
        classes=6, backbone='xception',
        activation='softmax')

    model.load_weights(model_dir)

    for i, path in enumerate(img_list):
        img = cv2.imread(
            path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        input = cv2.resize(
            img, (img.shape[1]//2, img.shape[0]//2))[
                0:img_shape[0], 0:img_shape[1]]

        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB) / 255.0

        print("THE NEXT IS THE LINE THAT USED TO CRASH")
        prediction = model.predict(np.asarray([np.array(input)])) # As of 03 Feb 2022, this is the line that crashes.

        bgr_mask = DigitMapToBGR(
            pallete, digit_map=np.squeeze(prediction, 0))()

        out_path = os.path.join(
            out_dir,
            os.path.splitext(os.path.split(path)[-1])[0] + '.png')
        pad_im = cv2.copyMakeBorder(
            bgr_mask,
            0, 208, 0, 0,
            cv2.BORDER_CONSTANT, value=label_color_dic['background'])

        out_im = cv2.resize(
            pad_im, (2048, 2464),
            interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(out_path, out_im)

        sys.stdout.write('\r%5d/%5d' % (i+1, len(img_list)))
        sys.stdout.flush()


def mask_image(im, mask):
    binmask = np.any(mask, axis=2).astype('int')
    maskdim = im*np.stack(3*[binmask], axis=2)
    out = cv2.cvtColor(maskdim.astype('uint8'), cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = binmask * 255
    return out


def mask_all_images(img_dir, mask_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    img_list = get_img_paths(img_dir)

    if len(img_list) == 0:
        print('no images found')
        return

    for i, path in enumerate(img_list):
        base, filepath = os.path.split(path)

        filename, _ = os.path.splitext(filepath)
        maskpath = os.path.join(mask_dir, filename + '.png')
        outpath = os.path.join(out_dir)

        if os.path.exists(maskpath):
            im = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(maskpath)
            out = mask_image(im, mask)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            cv2.imwrite(os.path.join(outpath, filepath), out)

        sys.stdout.write('\r%5d/%5d' % (i+1, len(img_list)))
        sys.stdout.flush()


def rename_labels(df, label_dir):

    for i, row in df.iterrows():
        fname, _ = os.path.splitext(row['path'])
        path = os.path.join(label_dir, fname + '.png')

        index = str(int(row['FRAME'])*10 + int(row['cam']))
        newname = os.path.join(label_dir, index + '.png')
        os.rename(path, newname)

def autogenerate(args):
    ''' generate working directory for batch processing '''
    args = verify_args(args)

    print("localising and finding views")
    lldb = localise(args.gps, args.ldb)  # localised ladybug frames
    gdf = as_gdf(lldb)  # give it geometry

    centroid = calculate_focalpoint(args.polygon)

    subset = get_views_by_centroid(gdf,
        centroid, args.radius, args.view_dist, args.fov)

    filedict = generate_filedict(args.dir)

    selection = get_filepaths(subset, filedict)

    print("generating working directory")
    generate_working_directory(selection, args.dir, out_dir = args.out)

    print("labelling images")
    label_directory(
        os.path.join(args.out, "images"),
        os.path.join(args.out, "labels")) # As of 03 Feb 2022, this is where it crashes.
    print("\nmasking images")
    mask_all_images(
        os.path.join(args.out, "images"),
        os.path.join(args.out, "labels"),
        os.path.join(args.out, "masked"))

    print("\ncreating cameraInit files")
    local_selection = generate_local_coords(selection, centroid)

    generate_cameraInit(
        local_selection,
        os.path.join(args.out, "images"),
        output=os.path.join(args.out, 'cameraInit.sfm'))

    generate_cameraInit(
        local_selection,
        os.path.join(args.out, "masked"),
        output=os.path.join(args.out, 'cameraInit_label.sfm'))

    print("renaming label data")
    rename_labels(local_selection, os.path.join(args.out, "labels"))

    print("done.")
    return os.path.abspath(args.out)


def generate_argparser():
    ''' generate argparser for command line usage '''
    parser = argparse.ArgumentParser()

    parser.add_argument('gps',
        help='location of imu/gnss file to localise with')
    parser.add_argument('ldb',
        help='ladybug frame information')
    parser.add_argument('dir',
        help='image directory containing ladybug frames')
    parser.add_argument('-o', '--out',
        help='file directory to write reconstruction subset to')
    parser.add_argument('-p', '--polygon',
        help='file containing polygon cordinates with columns x, y [m]')
    parser.add_argument('--radius',
        default=20.0,
        help='radius about polygon to search [m]')
    parser.add_argument('--view_dist',
        default=40.0,
        help='camera view length')
    parser.add_argument('--fov',
        default=np.pi/2,
        help='field of view for ladybug (radians)')

    return parser


def verify_args(args):
    ''' verify and validate arguments and set defaults if invalid '''
    if args is None:
        class Arguments:  # dummy class
            pass
        args = Arguments()

    if not hasattr(args, 'polygon'):
        args.polygon = None

    if not hasattr(args, 'out') or args.out is None:
        wd = 'wd_'+datetime.utcnow().strftime('%Y%m%d%H%M%S')
        args.out = os.path.abspath(wd)
        print('default output directory : %s' % args.out)

    if not hasattr(args, 'gps') or args.gps is None:
        raise ValueError('no valid gps file')
    else:
        if not os.path.isfile(args.gps):
            raise ValueError('could not find gps file')

    if not hasattr(args, 'ldb') or args.ldb is None:
        raise ValueError('no valid ladybug frame file')
    else:
        if not os.path.isfile(args.ldb):
            raise ValueError('could not find ladybug frame file')

    if not hasattr(args, 'dir') or args.dir is None:
        raise ValueError('no image directory provided')
    else:
        if not os.path.isdir(args.dir):
            raise ValueError('could not find image directory')

    if not hasattr(args, 'radius') or args.radius is None:
        args.radius = 20
    if not hasattr(args, 'view_dist') or args.view_dist is None:
        args.view_dist = 40
    if not hasattr(args, 'fov') or args.fov is None:
        args.fov = np.pi/2

    return args

def main(args):
    """ """
    args = verify_args(args)
    try:
        outdir = autogenerate(args)
        print('saved reconstruction directory to \n%s' % outdir)
    except Exception as err:
        print('Error: {}'.format(err))
        print('cleaning up')
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)

if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()

    main(args)
