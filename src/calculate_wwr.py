import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

import os


def encode(x):
    _x = np.array(x).astype('int')
    return (_x[..., 2] << 16) + (_x[..., 1] << 8) + _x[..., 0]


def decode(x):
    return np.stack([x & 0xFF, (x & 0xFF00) >> 8, (x & 0xFF0000) >> 16], axis=-1).astype(np.uint8)


def get_label_dic():
    label = ['background', 'chimney', 'door', 'window', 'roof', 'wall']

    label_color_dic = {
        i:[int(j_) for j_ in j]
        for i,j in zip(
            label,
            decode(np.linspace(0, encode([255,192,128]), 6).astype('int')))}

    return label_color_dic


def calculate_wwr(im):

    label_color_dic = get_label_dic()
    dim = encode(np.asarray(im))
    nbWin = len(dim[dim==encode(label_color_dic['window'])])
    nbWal = len(dim[dim==encode(label_color_dic['wall'])])

    if nbWin > nbWal or nbWal == 0:
        return np.nan
    else:
        return nbWin / nbWal


def ensemble_wwr(label_dir):
    wwrs = []
    for i, path in enumerate(os.listdir(label_dir)):
        im = Image.open(os.path.join(label_dir, path))
        wwr = calculate_wwr(im)
        if wwr < 0.8:
            wwrs.append(wwr)
    return wwrs


def calculate(args):
    ''' '''
    args = verify_args(args)

    wwrs = ensemble_wwr(args.dir)

    wwr = np.nanmedian(wwrs)
    if np.isnan(wwr):
        wwr = 0.4  # default
    return wwr


def verify_args(args):
    ''' '''
    if args is None:
        class Arguments:  # dummy class
            pass
        args = Arguments()

    if not hasattr(args, 'dir') or args.dir is None:
        raise ValueError('cannot calculate wwr without working directory')

    return args


def main(args):
    ''' main body of code '''
    wwr = calculate(args)

    print(wwr)



if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()

    main(args)
