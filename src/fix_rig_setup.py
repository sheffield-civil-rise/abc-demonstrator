import os
import sys
import math


def re(*args):
    return 'Ladybug-Stream-%s_ColorProcessed_%s_Cam%d_%s' % args


def ex(fn):
    return fn[15:30], fn[46:52], fn[58:]


def np(*args):
    return '%s_%s_%s' % args


def sw(n):
    return math.ceil(math.log10(n))


if __name__ == '__main__':

    rig_dir = "C:\\Users\\kevin\\Documents\\MARVel\\handsworth\\1G\\rig"

    cameras = [0, 1, 2, 3, 4, 5]

    img_list = os.listdir(os.path.join(rig_dir, '0'))  # preallocate jic

    def prog(k, cam):
        total = len(cameras)*len(img_list)
        fstr = '\rrenamed %'+str(sw(total))+'d/%d files'
        sys.stdout.write(fstr % ((k*len(cameras)) + cam + 1, total))
        sys.stdout.flush()

    for k, filename in enumerate(img_list):
        fn, ext = os.path.splitext(filename)
        dt, ix, ui = ex(fn)

        if re(dt, ix, 0, ui) != fn:
            continue

        for cam in cameras:
            opath = os.path.join(rig_dir, str(cam), re(dt, ix, cam, ui) + ext)
            npath = os.path.join(rig_dir, str(cam), np(dt, ix, ui) + ext)

            # print(opath + ' --> ' + npath)
            os.rename(opath, npath)  # rename file
            prog(k, cam)
