import os
import sys
import argparse

import math
import numpy as np

import cv2


def sw(n):
    return math.ceil(math.log10(n))


def prog(k, N):
    fstr = '\rcopied %'+str(sw(N))+'d/%d files'
    sys.stdout.write(fstr % (k+1, N))
    sys.stdout.flush()


def mask_image(im, mask):
    binmask = np.any(mask, axis=2).astype(np.int)
    maskdim = im*np.stack(3*[binmask], axis=2)
    out = cv2.cvtColor(maskdim.astype('uint8'), cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = binmask * 255
    return out


def main(img_dir, mask_dir, out_dir, recursive=True):

    img_list = get_img_paths(img_dir, recursive=recursive)

    if len(img_list) == 0:
        print('no images found')
        return

    for i, path in enumerate(img_list):
        base, filepath = os.path.split(path)
        _, cam = os.path.split(base)
        filename, _ = os.path.splitext(filepath)
        maskpath = os.path.join(mask_dir, 'rig', cam, filename + '.png')
        outpath = os.path.join(out_dir, 'rig', cam)

        if os.path.exists(maskpath):
            im = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(maskpath)
            out = mask_image(im, mask)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            cv2.imwrite(os.path.join(outpath, filepath), out)
            # break
            prog(i, len(img_list))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--img_dir", type=str, help="location of images to label")

    parser.add_argument(
        "-m", "--mask_dir", type=str, help="location of labelled masks")

    parser.add_argument(
        "-o", "--out_dir", type=str,
        help="output directory for masked images")

    parser.add_argument(
        "-n", "--norecurse", action='store_true',
        help='prevent recursive searching of directories')

    args = parser.parse_args()

    if args.img_dir is None:
        raise ValueError('No image directory supplied')

    if args.mask_dir is None:
        base_path, img_dir = os.path.split(args.img_dir)
        args.mask_dir = os.path.join(base_path, img_dir + '_labels')
        print('No mask directory supplied, looking in ')
        print('   {}'.format(args.mask_dir))

    if args.out_dir is None:
        base_path, img_dir = os.path.split(args.img_dir)
        args.out_dir = os.path.join(base_path, img_dir + '_masked')
        print('No output directory supplied, writing files to ')
        print('   {}'.format(args.out_dir))

    # run main body
    main(args.img_dir, args.mask_dir, args.out_dir, not args.norecurse)
