import json
import numpy as np

import os
import argparse

import scipy
from sklearn.decomposition import PCA

import open3d as o3d


def get_rotation(points):
    pca = PCA()
    pca.fit(points.T)
    return pca.components_.T


def get_translation(points):
    return np.mean(points, axis=1)


def get_scale(points):
    pca = PCA()
    pca.fit(points.T)
    return pca.explained_variance_


def generate_transform(tru_xyz, sfm_xyz):
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


def to_homogeneous(transform):
    T, R, S = transform

    Th, Rh, Sh = [np.zeros((4, 4)) for _ in range(3)]
    Th[0:4, 0:4] = np.eye(4)
    Th[0, 3] = T[0]
    Th[1, 3] = T[1]
    Th[2, 3] = T[2]

    Rh[0:3, 0:3] = R
    Rh[3, 3] = 1

    Sh[0, 0] = np.sqrt(S[0])
    Sh[1, 1] = np.sqrt(S[1])
    Sh[2, 2] = np.sqrt(S[2])
    Sh[3, 3] = 1.0
    return Th, Rh, Sh


def joint_transform(forward, backward, separate_scale=False):
    Tf, Rf, Sf = to_homogeneous(forward)
    Tb, Rb, Sb = to_homogeneous(backward)

    Rb[0:3, 0:3] = scipy.linalg.inv(Rb[0:3, 0:3])
    Sb = np.diag(1/np.diag(Sb))

    S = Sf@Sb

    _s = 0.5 * (S[0, 0] + S[1, 1])
    S = _s * np.eye(4)
    S[3, 3] = 1.

    Tb[0:3, 3] = -Tb[0:3, 3]
    if separate_scale:
        return Tf@(Rf@(Rb@(Tb))), S
    else:
        return Tf@(Rf@(S@(Rb@(Tb))))


def read_sfm(path):
    ''' read .sfm file and returns dictionary with info '''
    with open(path, 'r') as fid:
        sfm = json.load(fid)
    return sfm


def get_geometries(poses, match=None):
    ''' extract centroid and rotations of sfm poses '''
    rotations = {}
    centers   = {}
    for i, pose in enumerate(poses):
        id = int(pose['poseId'])
        if match is not None:
            if id not in match.keys():
                continue
        centers[id] = np.array(
            [float(x) for x in pose['pose']['transform']['center']])
        rotations[id] = np.reshape(np.array(
            [float(x) for x in pose['pose']['transform']['rotation']]),[3,3])
    return rotations, centers


def calculate_transform(ref, sfm):
    ''' calculate the transformation between relative and reference spaces

        input:
            ref  : reference sfm with true poses (filename)
            sfm  : generated sfm with adjusted poses (filename)
    '''
    reference = read_sfm(ref)
    reconstrc = read_sfm(sfm)

    _, rcn_centers = get_geometries(reconstrc['poses'])
    _, ref_centers = get_geometries(reference['poses'], rcn_centers)

    T, _, _ = generate_transform(
        *[np.vstack(list(c.values())).T for c in [ref_centers, rcn_centers]])

    return T


def load_mesh(path):
    ''' load in mesh '''

    cwd = os.getcwd()
    mesh_base, filename = os.path.split(path)

    os.chdir(mesh_base)  # we change directory to read to avoid textures issues

    mesh = o3d.io.read_triangle_mesh(filename)
    os.chdir(cwd)  # revert working directory
    return mesh


def calculate_geometry(mesh, transform):
    tmesh = mesh.transform(transform)

    V = np.asarray(tmesh.vertices)
    width, height, depth = V.max(axis=0) - V.min(axis=0)
    return width, height, depth


def calculate_height(mesh, transform, ignore_roof=False):
    _, height, _ = calculate_geometry(mesh, transform)
    print("MESH: "+str(mesh))
    print("TRANSFORM: "+str(transform))
    print("RAW HEIGHT: "+str(height))
    if ignore_roof:
        return height
    else:
        return 2 * height / 3

def generate_argparser():
    ''' generate argument parser '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r', '--ref', help='reference SfM with true poses')
    parser.add_argument(
        '-s', '--sfm', help='adapted poses in SfM')
    parser.add_argument(
        '-m', '--mesh', help='mesh of reconstruction to calculate heights')

    return parser


def verify_args(args):
    ''' verify arguments '''
    return args


def main(args):
    T = calculate_transform(args.ref, args.sfm)

    mesh = load_mesh(args.mesh)

    height = calculate_height(mesh, T, ignore_roof=True)

    return height


if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()

    args = verify_args(args)
    height = main(args)

    print(height)
