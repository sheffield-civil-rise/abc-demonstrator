
""" SCHEMA
{
    "version": [
        "1",
        "0",
        "0"
    ],
    "views": [
        {
            "viewId": "",
            "poseId": "",
            "intrinsicsId": "",
            "resectionId": "",
            "path": "",
            "width": "",
            "height": "",
            "metadata": {}
        }
    ],
    "intrinsics":[
        {
            "intrinsicId": "",
            "width": "",
            "height": "",
            "sensorWidth": "",
            "sensorHeight": "",
            "serialNumber": "",
            "type": "",
            "initializationMode": "",
            "pxInitialFocalLength": "",
            "pxFocalLength": "",
            "principalPoint": [
                "",
                ""
            ],
            "distortionParams": [
                "",
                "",
                "",
                ""
            ],
            "locked": ""
        }
    ],
    "poses": [
        {
            "poseId": "",
            "pose": {
                "transform": {
                    "rotation": [
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    ],
                    "center": [
                        "",
                        "",
                        ""
                    ]
                }
            }
        }
    ]
}
"""

import pandas as pd
import os
import re
import json
import argparse

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
    return {
        "viewId": str(row['id']),
        "poseId": str(row['id']),
        "intrinsicsId": str(row['cam']),
        "resectionId": "",
        "path": os.path.join(base_dir, row['image']),
        "width": "2048",
        "height": "2464",
        "metadata": {}}

def create_pose(row):
    return {
        "poseId": str(row['id']),
        "pose": {
            "transform": {
                "rotation":re.findall(r"-?[\d.]+(?:e-?\d+)?", row['rotation']),
                "center": [
                    str(row['latitude']),
                    str(row['longitude']),
                    str(row['altitude'])]}}}

def read_db(path):
    df = pd.read_csv(path)
    return df

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

def main(args):

    df = read_db(args.ldb)
    print('generating camera init')
    init_dict = build_init_dict(df, os.path.abspath(args.img_dir))

    print('writing out')
    with open(args.output, 'wb') as fid:
        json.dump(init_dict, fid, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        args.output = 'cameraInit.sfm'

    main(args)
