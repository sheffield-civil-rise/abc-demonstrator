from scale_sfm import load_sfm, load_selection, extract_points
from scale_sfm import get_translation, get_rotation, get_scale, joint_transform

from plotly import graph_objects as go

import open3d as o3d

import os
import numpy as np

import pandas as pd
import geopandas as gpd

from matplotlib import pyplot as plt

import argparse


def calculate_transform(sel_path, sfm_path):
    ''' get transformation matrix '''
    sel, sfm = load_selection(sel_path), load_sfm(sfm_path)

    tru_xyz = extract_points(sel)
    sfm_xyz = extract_points(sfm)

    forward = (
        get_translation(tru_xyz),
        get_rotation(tru_xyz),
        get_scale(tru_xyz))
    backward = (
        get_translation(sfm_xyz),
        get_rotation(sfm_xyz),
        get_scale(sfm_xyz))

    T = joint_transform(forward, backward)

    return T, forward, backward


def main(args):
    ''' '''
    mesh_path = args.file
    sfm_path, sel_path = args.sfm, args.sel

    T, forward, backward = calculate_transform(sel_path, sfm_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.transform(T)

    o3d.visualization.draw_geometries([mesh])

    # plot_overlay(mesh)

    dir, fname = os.path.split(mesh_path)
    name, ext = os.path.splitext(fname)

    path = os.path.join(dir, name + '_transformed' + ext)
    save(path, mesh)


def convert_crs(points):
    gdf = gpd.GeoDataFrame(
        data=points,
        geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
        crs='epsg:27700'
    )
    return gdf.to_crs('epsg:4326')


def save(path, mesh):
    print('saving mesh to {}'.format(path))
    o3d.io.write_triangle_mesh(
        path, mesh,
        write_ascii=True,
        write_triangle_uvs=False,
        compressed=False)


def plot_overlay(mesh):
    pc = np.asarray(mesh.vertices)[::100, :]

    df = convert_crs(pc)
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lon=df.geometry.x,
            lat=df.geometry.y,
            ids=df.index,
            mode='markers',
            marker=dict(size=10, color='red'),
            text='',  # TODO: add opt in args,
        )
    )
    fig.update_layout(
            dragmode='lasso',
            mapbox={
                'center': {
                    'lon': df.geometry.x.mean(),
                    'lat': df.geometry.y.mean()},
                'style': 'open-street-map',  # TODO: add opt in args,
                'zoom': 13})  # TODO: add opt in args
    fig.show()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file", type=str, help="mesh to rescale")
    parser.add_argument(
        "--sfm", type=str, help="location of .sfm file")
    parser.add_argument(
        "-s", "--sel", type=str, help="location of selection file")

    args = parser.parse_args()

    if args.file is None or not os.path.exists(args.file):
        raise ValueError('no valid input mesh')
    if args.sfm is None or not os.path.exists(args.sfm):
        raise ValueError('no valid sfm file')

    if args.sel is None or not os.path.exists(args.sel):
        raise ValueError('no valid sel file')

    main(args)
