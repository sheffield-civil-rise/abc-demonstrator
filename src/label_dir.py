from deeplab.Deeplabv3 import Deeplabv3

import numpy as np

# from PIL import Image
import cv2
import os
import sys
import argparse
import warnings
import math

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_DEBUG = False

label_color_dic = {'background': [0, 0, 0],
                   'chimney': [128, 0, 0],
                   'door': [255, 0, 0],
                   'window': [0, 128, 0],
                   'roof': [255, 255, 0],
                   'wall': [42, 42, 165]}

label_value_dic = {'background': 0,
                   'chimney': 3,
                   'door': 5,
                   'window': 4,
                   'roof': 2,
                   'wall': 1}

pallete = {
    label_value_dic['background']: np.array(label_color_dic['background']),
    label_value_dic['window']: np.array(label_color_dic['window']),
    label_value_dic['door']: np.array(label_color_dic['door']),
    label_value_dic['chimney']: np.array(label_color_dic['chimney']),
    label_value_dic['roof']: np.array(label_color_dic['roof']),
    label_value_dic['wall']: np.array(label_color_dic['wall'])}


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


if __name__ == '__main__':
    ''' main body '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--img_dir", type=str, help="location of images to label")
    parser.add_argument(
        "-o", "--out_dir", type=str,
        help="location of data subset (automatically same as selection file)")
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
            raw_image_dir, '..', os.path.split(raw_image_dir)[-1] + '_labels')
    else:
        mask_out_dir = os.path.abspath(args.out_dir)

    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    search_recurse = not args.norecurse

    model_dir = r'src\deeplab\Deeplabv3plus-xception-ce.hdf5'
    img_list = get_img_paths(raw_image_dir, recursive=search_recurse)

    img_shape = (1024, 1024)
    model = Deeplabv3(
        weights=None, input_shape=(*img_shape, 3),
        classes=6, backbone='xception',
        activation='softmax')

    model.load_weights(model_dir)

    for i, path in enumerate(img_list):
        # img = Image.load(path)
        if _DEBUG:
            img = cv2.imread(
                "src/deeplab/gsv-examples/image_'1522478_gsv_0.jpg")
            input = cv2.resize(
                img, img_shape)[0:img_shape[1], 0:img_shape[0], :]
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB) / 255.0
        else:
            # mm = cv2.imread(
            #     os.path.join(
            #         mask_out_dir,
            #         os.path.splitext(os.path.split(path)[-1])[0] + '.png'))
            # if mm.shape[0] == 2464 and mm.shape[1] == 2048:
            #     continue
            img = cv2.imread(
                path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            input = cv2.resize(
                img, (img.shape[1]//2, img.shape[0]//2))[
                    0:img_shape[0], 0:img_shape[1]]

            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB) / 255.0

        prediction = model.predict(np.asarray([np.array(input)]))

        bgr_mask = DigitMapToBGR(
            pallete, digit_map=np.squeeze(prediction, 0))()

        # print(path)

        if not search_recurse:
            out_path = os.path.join(
                mask_out_dir,
                os.path.splitext(os.path.split(path)[-1])[0] + '.png')
        else:
            head, fname = os.path.split(path)
            _, subdir = os.path.split(head)

            outname = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(mask_out_dir, 'rig', subdir, outname)
            _outdir = os.path.split(out_path)[0]
            if not os.path.exists(_outdir):
                os.makedirs(_outdir)
        #
        # out_path = os.path.join(
        #     mask_out_dir,
        #     os.path.splitext(os.path.split(path)[-1])[0] + '.png')

        # pad_im = cv2.copyMakeBorder(
        #     bgr_mask,
        #     0, img.shape[0]//2 - img_shape[0], 0, 0,
        #     cv2.BORDER_CONSTANT, value=label_color_dic['background'])
        pad_im = cv2.copyMakeBorder(
            bgr_mask,
            0, 208, 0, 0,
            cv2.BORDER_CONSTANT, value=label_color_dic['background'])

        # out_im = cv2.resize(
        #     pad_im, (img.shape[1], img.shape[0]),
        #     interpolation=cv2.INTER_NEAREST)
        out_im = cv2.resize(
            pad_im, (2048, 2464),
            interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(out_path, out_im)

        if _DEBUG:
            break
        else:
            prog(i, len(img_list))
