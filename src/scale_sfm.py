import json

import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import rc_context

import argparse
import os

from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import scipy


def load_sfm(path):
    ''' Load in json sfm file from meshroom '''
    with open(path, 'r') as fid:
        cams = json.load(fid)
    views = pd.DataFrame.from_dict(cams['views'])
    intrs = pd.DataFrame.from_dict(cams['intrinsics'])
    poses = pd.DataFrame.from_dict(cams['poses'])

    tmp = pd.merge(views, intrs, on=['intrinsicId'])
    out = pd.merge(tmp, poses, on=['poseId'])
    return out


def load_selection(path):
    ''' Read in selection file '''
    return pd.read_csv(path)


def extract_points(df, ignore_z=True):
    ''' '''
    if 'geometry' in df.columns:
        x = df['xcoord'].to_numpy()
        y = df['ycoord'].to_numpy()
        z = df['zcoord'].to_numpy() if not ignore_z else np.zeros_like(x)
    else:
        geom = []
        for i in range(3):
            geom.append(np.array([
                float(df.at[j, 'pose']['transform']['center'][i])
                for j in range(len(df))]))
        x, y, z = geom
    return np.vstack([x, y, z])


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


def hx(xyz):
    return np.vstack([np.atleast_2d(xyz).T, [[1]]])


def deproject(xyz, transform, w=True):
    if not w:
        T, R, S = transform
        tmp = scipy.linalg.inv(R) @ np.atleast_2d(xyz - T).T
        return tmp.ravel()/np.sqrt(S)
    else:
        T, R, S = to_homogeneous(transform)

        S = np.diag(1/np.diag(S))

        T[0:3, 3] = -T[0:3, 3]
        return (S@(scipy.linalg.inv(R)@(T@hx(xyz))))[0:3].ravel()


def project(xyz, transform):
    T, R, S = transform

    return T + (R @ np.atleast_2d(xyz * np.sqrt(S)).T).ravel()


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sfm", type=str, help="location of .sfm file")
    parser.add_argument(
        "-s", "--sel", type=str, help="location of selection file")
    args = parser.parse_args()

    if args.sfm is None or not os.path.exists(args.sfm):
        raise ValueError('no valid sfm file')

    if args.sel is None or not os.path.exists(args.sel):
        raise ValueError('no valid sel file')

    sfm_path, sel_path = args.sfm, args.sel

    sel = load_selection(sel_path)
    sfm = load_sfm(sfm_path)

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

    transformed_sfm = np.hstack(
        [np.atleast_2d(
            project(deproject(sfm_xyz[:, i], backward), forward)
        ).T for i in range(sfm_xyz.shape[1])]
    )

    T = joint_transform(forward, backward)
    homogeneous_sfm = np.hstack([
        (T @ np.vstack([np.atleast_2d(sfm_xyz[:, i]).T, [[1]]]))[0:3]
        for i in range(sfm_xyz.shape[1])])
    # to_homogeneous(forward)
    # to_homogeneous(backward)

    forward_ = (forward[0], forward[1], np.ones(3))
    backward_ = (backward[0], backward[1], np.ones(3))
    unscaled_sfm = np.hstack([
        np.hstack(
            [np.atleast_2d(deproject(sfm_xyz[:, i], backward)).T
                for i in range(sfm_xyz.shape[1])])])
    unscaled_tru = np.hstack([
        np.hstack(
            [np.atleast_2d(deproject(tru_xyz[:, i], forward)).T
                for i in range(tru_xyz.shape[1])])])
    #
    # nSt = get_scale(unscaled_tru)
    # nSt[2] = 1.
    # nSs = get_scale(unscaled_tru)

    nT, nS = joint_transform(forward, backward, separate_scale=True)
    print('transl')
    print(nT[0:3, 3])
    print('..')
    print('rotv')
    print(Rotation.from_matrix(nT[0:3, 0:3]).as_rotvec())
    print('euler xyz')
    print(Rotation.from_matrix(nT[0:3, 0:3]).as_euler('xyz', degrees=True))
    print('..')
    print('scale')
    print((np.diag(nS)[0:2], np.mean(np.diag(nS)[0:2])))
    # print(forward[1]@np.diag(np.sqrt(nSt)/np.sqrt(nSs)))

    fig = plt.figure()
    axs = [fig.add_subplot(1, 2, i+1, projection='3d') for i in range(2)]

    axs[0].plot3D(*tru_xyz, '.')
    axs[1].plot3D(*sfm_xyz, '.')
    #
    # # print('plotting ...')
    # # with rc_context(rc={'interactive': True}):
    fig.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    comb = np.hstack([tru_xyz, transformed_sfm])
    cols = ['b' for i in range(tru_xyz.shape[1])] \
        + ['r' for i in range(transformed_sfm.shape[1])]

    ax.scatter3D(*comb, c=cols)
    # # print('plotting ...')
    # # with rc_context(rc={'interactive': True}):

    fig.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    comb = np.hstack([tru_xyz, homogeneous_sfm])
    cols = ['b' for i in range(tru_xyz.shape[1])]\
        + ['r' for i in range(transformed_sfm.shape[1])]
    ax.scatter3D(*comb, c=cols)

    plt.show()
