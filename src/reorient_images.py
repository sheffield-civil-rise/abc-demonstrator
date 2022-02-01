import cv2
import os
import argparse
import sys
import math


def sw(n):
    return math.ceil(math.log10(n))


def prog(k, N):
    fstr = '\rcopied %'+str(sw(N))+'d/%d files'
    sys.stdout.write(fstr % (k+1, N))
    sys.stdout.flush()


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


def rotate(img, _assert=None):
    out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if _assert is not None:
        assert(
            (out.shape[0] == _assert[0]) &
            (out.shape[1] == _assert[1]))

    return out


def main(img_dir, out_dir, recursive=True):
    img_paths = get_img_paths(img_dir, recursive)

    for i, path in enumerate(img_paths):
        img = cv2.imread(
            path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = rotate(img, _assert=(2464, 2048))

        if not recursive:
            out_path = os.path.join(out_dir, os.path.split(path)[-1])
        else:
            head, fname = os.path.split(path)
            _, subdir = os.path.split(head)
            out_path = os.path.join(out_dir, 'rig', subdir, fname)
            _outdir = os.path.split(out_path)[0]
            if not os.path.exists(_outdir):
                os.makedirs(_outdir)
        cv2.imwrite(out_path, img)
        prog(i, len(img_paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", type=str, help="location of images to reorient")
    parser.add_argument(
        "-o", "--out_dir", type=str,
        help="location to save reoriented images")
    parser.add_argument(
        "-n", "--norecurse", action='store_true',
        help='prevent recursive searching of directories')

    args = parser.parse_args()

    if args.img_dir is None:
        raise RuntimeError('No input images provided.')
    else:
        raw_image_dir = os.path.abspath(args.img_dir)

    if args.out_dir is None:
        mask_out_dir = os.path.join(
            raw_image_dir, '..', os.path.split(raw_image_dir)[-1] + '_rotated')
    else:
        mask_out_dir = os.path.abspath(args.out_dir)

    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    search_recurse = not args.norecurse

    main(raw_image_dir, mask_out_dir, search_recurse)
