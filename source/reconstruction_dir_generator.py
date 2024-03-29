"""
This code defines a class which generates the reconstruction directory.
"""

# Standard imports.
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import sys

# Non-standard imports.
import cv2
import geopandas
import numpy
import pandas
from PIL import Image
from scipy.interpolate import interp1d
from shapely.geometry import Point, Polygon

# Local imports.
from local_configs import (
    make_path_to_gps_data,
    make_path_to_ladybug_gps_data,
    make_path_to_ladybug_images,
    CONFIGS,
    SEMICIRCLE_DEGREES
)
from deeplab.Deeplabv3 import Deeplabv3
from utility_functions import make_label_color_dict

##############
# MAIN CLASS #
##############

@dataclass
class ReconstructionDirGenerator:
    """ The class in question. """
    # Fields.
    path_to_gps_data: str = CONFIGS.paths.path_to_gps_data
    path_to_ladybug_gps_data: str = CONFIGS.paths.path_to_ladybug_gps_data
    path_to_ladybug_images: str = CONFIGS.paths.path_to_ladybug_images
    path_to_polygon: str = CONFIGS.paths.path_to_polygon
    path_to_input_override: str = None
    path_to_output: str = CONFIGS.paths.path_to_output
    path_to_model: str = CONFIGS.paths.path_to_deeplab_binary
    co_ref_sys: str = CONFIGS.general.coordinate_reference_system
    src_co_ref_sys: str = CONFIGS.general.source_coordinate_reference_system
    radius: int = CONFIGS.reconstruction_dir.radius
    view_distance: int = CONFIGS.reconstruction_dir.view_distance
    field_of_view: float = CONFIGS.reconstruction_dir.field_of_view
    output_image_extension: str = \
        CONFIGS.reconstruction_dir.output_image_extension
    number_of_cameras: int = CONFIGS.reconstruction_dir.number_of_cameras
    # Generated fields.
    path_to_output_images: str = None
    path_to_labelled_images: str = None
    path_to_masked_images: str = None
    gps_data: pandas.DataFrame = None
    ladybug_data: pandas.DataFrame = None
    localised_ladybug_data: pandas.DataFrame = None
    geo_data_frame: geopandas.GeoDataFrame = None
    centroid: list = None
    subset: geopandas.GeoDataFrame = None
    file_dict: dict = None
    file_paths: geopandas.GeoDataFrame = None
    local_selection: geopandas.GeoDataFrame = None
    localised_gps_data: pandas.DataFrame = None

    # Class attributes.
    LADYBUG_USECOLS: ClassVar[list] = ["FRAME", "CAMERA TIME"]
    GPS_COLUMNS: ClassVar[list] = [
        "Latitude (deg)",
        "Longitude (deg)",
        "Altitude (m)",
        "Heading (deg)",
        "Pitch (deg)",
        "Roll (deg)"
    ]
    DEFAULT_FOCAL_POINT: ClassVar[list] = [
        -1.5120831368308705,
        53.35550826329699
    ]
    EXPANDED_COLUMNS: ClassVar[list] = [
        "FRAME",
        "CAMERA TIME",
        "latitude",
        "longitude",
        "altitude",
        "heading",
        "pitch",
        "roll"
    ]
    SEGMENT_RESOLUTION: ClassVar[int] = 20
    IMAGE_FILENAME_INDICES: ClassVar[tuple] = (46, 52, 56)
    MAX_CAM_INDEX: ClassVar[int] = 5
    LABEL_VALUE_DICT: ClassVar[dict] = CONFIGS.general.label_value_dict
    LABEL_COLOR_DICT: ClassVar[dict] = make_label_color_dict()
    PALETTE: ClassVar[dict] = None
    IMG_SHAPE: ClassVar[tuple] = (1024, 1024)
    BORDER_BOTTOM: ClassVar[int] = 208
    PADDED_IMG_SHAPE: ClassVar[tuple] = (2048, 2464)
    CAMERA_INIT_FILENAME: ClassVar[str] = "cameraInit.sfm"
    CAMERA_INIT_LABEL_FILENAME: ClassVar[str] = "cameraInit_label.sfm"
    JSON_INDENT: ClassVar[int] = 2
    FRAME_ENCODING_FACTOR: ClassVar[int] = 10
    INTRINSIC_BASE: ClassVar[dict] = {
        "width": "2048",
        "height": "2464",
        "sensorWidth": "-1",
        "sensorHeight": "-1",
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
        "locked": "0"
    }
    CIRCLE_RESOLUTION: ClassVar[int] = \
        CONFIGS.reconstruction_dir.circle_resolution
    CIRCLE_RADIUS: ClassVar[int] = CONFIGS.reconstruction_dir.radius

    def __post_init__(self):
        if self.path_to_input_override:
            self.path_to_gps_data = \
                make_path_to_gps_data(stem=self.path_to_input_override)
            self.path_to_ladybug_gps_data = \
                make_path_to_ladybug_gps_data(stem=self.path_to_input_override)
            self.path_to_ladybug_images = \
                make_path_to_ladybug_images(stem=self.path_to_input_override)
        self.set_palette()

    def set_palette(self):
        """ Initialise this class attribute. """
        self.PALETTE = {
            self.LABEL_VALUE_DICT[label]: \
                numpy.flip(numpy.array(self.LABEL_COLOR_DICT[label]))
            for label in self.LABEL_VALUE_DICT.keys()
        }

    def create_circle(self, aspoints=False):
        """ Return a circle object. """
        centre = Point(self.centroid)
        circle_data_frame = \
            geopandas.GeoDataFrame(
                { "geometry": [centre] },
                crs=self.co_ref_sys
            ).to_crs(self.src_co_ref_sys)
        points = [
            (
                (
                    self.CIRCLE_RADIUS*numpy.sin(angle)+
                    circle_data_frame.geometry.x[0]
                ),
                (
                    self.CIRCLE_RADIUS*numpy.cos(angle)+
                    circle_data_frame.geometry.y[0]
                )
            ) for angle in numpy.linspace(0, 2*numpy.pi, self.CIRCLE_RESOLUTION)
        ]
        polygon = Polygon(points)
        geometry = [Point(point) for point in points] if aspoints else [polygon]
        result = \
            geopandas.GeoDataFrame(
                { "geometry": geometry },
                crs=self.src_co_ref_sys
            ).to_crs(self.co_ref_sys)
        return result

    def load_gps_data(self):
        """ Load the data from the paths. """
        self.gps_data = \
            pandas.read_csv(self.path_to_gps_data, skipinitialspace=True)
        self.ladybug_data = \
            pandas.read_csv(
                self.path_to_ladybug_gps_data,
                skipinitialspace=True,
                usecols=self.LADYBUG_USECOLS
            )
        try:
            self.ladybug_data.drop(
                self.ladybug_data[
                    self.ladybug_data.FRAME.str.contains("ERROR")
                ].index,
                inplace=True
            )
        except KeyError:
            pass

    def interpolate_gps_data(self):
        """ Interpolate the two sets of GPS data. """
        gps_time = "Time (HH:mm:ss.fff)"
        ldb_cols = [column.split()[0].lower() for column in self.GPS_COLUMNS]
        origin = parse_time(self.gps_data[gps_time][0])
        times = \
            numpy.array([
                (lambda seconds: seconds_since(origin, seconds))(
                    parse_time(time_in_seconds)
                )
                for time_in_seconds in self.gps_data[gps_time]
            ])
        times_new = \
            numpy.array([
                (lambda seconds: seconds_since(origin, seconds))(
                    parse_time(time_in_seconds)
                )
                for time_in_seconds in self.ladybug_data["CAMERA TIME"]
            ])
        result = self.ladybug_data.copy()
        for index, col in enumerate(self.GPS_COLUMNS):
            numpified_column = self.gps_data[col].to_numpy()
            f_pos = interp1d(times, numpified_column, kind="linear", copy=False)
            result[ldb_cols[index]] = f_pos(times_new)
        self.localised_gps_data = result

    def append_orientation(self, data_frame, inplace=False):
        """ Append camera orientations to a given data frame. """
        new_data_frame = data_frame.copy()
        new_data_frame["rotations"] = None # Initialise empty column.
        for index, row in data_frame.iterrows():
            rotations = \
                self.build_rotation(row["heading"], row["pitch"], row["roll"])
            new_data_frame.at[index, "rotations"] = rotations
        if inplace: # Append rotations to input dataframe.
            data_frame["rotations"] = new_data_frame["rotations"].values
            return data_frame
        return new_data_frame # Return copy with added rotations.

    def build_rotation(self, heading, pitch, roll):
        """ Build a rotation matrix. """
        heading, pitch, roll = [
            (lambda d: numpy.pi*d/SEMICIRCLE_DEGREES)(d) for d in [
                heading, pitch, roll
            ]
        ]
        cosa, cosb, cosg = numpy.cos(heading), numpy.cos(pitch), numpy.cos(roll)
        sina, sinb, sing = numpy.sin(heading), numpy.sin(pitch), numpy.sin(roll)
        R = \
            numpy.array([
                [cosa*cosb, cosa*sinb*sing-sina*cosg, cosa*sinb*cosg+sina*sing],
                [sina*cosb, sina*sinb*sing+cosa*cosg, sina*sinb*sing-cosa*sing],
                [-sinb, cosb*sing, cosb*cosg]
            ])
        result = [
            R @ (
                lambda i: numpy.array([
                    [
                        numpy.cos(angle_of_camera_with_index(i)),
                        -numpy.sin(angle_of_camera_with_index(i)),
                        0.
                    ],
                    [
                        numpy.sin(angle_of_camera_with_index(i)),
                        numpy.cos(angle_of_camera_with_index(i)),
                        0.
                    ],
                    [0., 0., 1.]
                ])
            )(cam) for cam in range(self.number_of_cameras)
        ]
        return result

    def make_localised_ladybug_gps_data(self):
        """ Localise the Ladybyg GPS data using the GPS CSV file. """
        self.interpolate_gps_data()
        self.append_orientation(self.localised_gps_data, inplace=True)

    def make_geo_data_frame(self):
        """ Add geometry to our localised data. """
        self.geo_data_frame = \
            to_geo_data_frame(self.localised_gps_data, self.co_ref_sys)

    def calculate_focal_point(self):
        """ Calculate the central point from the polygon. """
        if not self.path_to_polygon:
            return self.DEFAULT_FOCAL_POINT
        polygon = pandas.read_csv(self.path_to_polygon, header=None)
        geometry = geopandas.points_from_xy(polygon[1], polygon[0])
        geo_data_frame = \
            geopandas.GeoDataFrame(
                geometry=geometry,
                crs=self.src_co_ref_sys
            ).to_crs(self.co_ref_sys)
        centroid = Polygon(geo_data_frame.geometry).centroid
        result = [centroid.x, centroid.y]
        return result

    def make_centroid(self):
        """ Calculate the focal point, and set the centroid to that value. """
        self.centroid = self.calculate_focal_point()

    def expand_data_frame(self, data_frame):
        """ Expand a given data from to include the expanded columns. """
        new_data_frame = pandas.DataFrame(columns=self.EXPANDED_COLUMNS)
        geox, geoy = [], []
        for _, row in data_frame.iterrows():
            rotations = row["rotations"]
            nrow = row[self.EXPANDED_COLUMNS]
            for cam, _ in enumerate(rotations):
                nrow["cam"] = cam
                nrow["rotation"] = rotations[cam]
                if isinstance(data_frame, geopandas.GeoDataFrame):
                    geox.append(row["longitude"])
                    geoy.append(row["latitude"])
                else:
                    geox.append(row.geometry.x)
                    geoy.append(row.geometry.y)
                new_data_frame = new_data_frame.append(nrow, ignore_index=True)
        local_co_ref_sys = self.co_ref_sys
        if isinstance(data_frame, geopandas.GeoDataFrame):
            local_co_ref_sys = data_frame.crs
        result = \
            geopandas.GeoDataFrame(
                data=new_data_frame,
                geometry=geopandas.points_from_xy(geox, geoy),
                crs=local_co_ref_sys
            )
        return result

    def create_seg(self, geo, heading, cam):
        """ Create a segment polygon. """
        vector = numpy.array([geo.x, geo.y])
        angle0 = find_directions(heading, cam)
        points = [
            (
                vector[0]+self.view_distance*numpy.cos(angle),
                vector[1]+self.view_distance*numpy.sin(angle)
            ) for angle in numpy.linspace(
                angle0-(self.field_of_view/2),
                angle0+(self.field_of_view/2),
                self.SEGMENT_RESOLUTION
            )
        ]
        result = Polygon([vector, *points])
        return result

    def find_views(self, data_frame):
        """ Find views for a given data frame. """
        data_frame_ = data_frame.to_crs(self.src_co_ref_sys)
        data = \
            data_frame_.apply(
                lambda r: self.create_seg(r.geometry, r.heading, r.cam),
                axis=1
            ).tolist()
        data_frame["view"] = \
            geopandas.GeoSeries(
                data=data,
                crs=self.src_co_ref_sys
            ).to_crs(self.co_ref_sys)
        return data_frame

    def get_views_by_centroid(self):
        """ Use the centroid to filter out all but the data we want. """
        circle = self.create_circle()
        subset = self.geo_data_frame.overlay(circle, how="intersection")
        full_frame = self.expand_data_frame(subset)
        view_frame = self.find_views(full_frame)
        result = filter_by_view(view_frame, self.centroid)
        return result

    def select_the_subset(self):
        """ Make a subset of the geo data frame. """
        self.subset = self.get_views_by_centroid()

    def framecam(self, filename):
        """ Extract the necessary data from an image filename. """
        index0 = self.IMAGE_FILENAME_INDICES[0]
        index1 = self.IMAGE_FILENAME_INDICES[1]
        index2 = self.IMAGE_FILENAME_INDICES[2]
        return filename[index0:index1], filename[index2]

    def generate_file_dict(self):
        """ Generate a file dictionary for the image directory. """
        file_list = os.listdir(self.path_to_ladybug_images)
        result = [{}, {}, {}, {}, {}]
        for _, filename in enumerate(file_list):
            frame, cam = self.framecam(filename)
            if int(cam) < self.MAX_CAM_INDEX:
                result[int(cam)][frame] = filename
        self.file_dict = result

    def select_file_paths(self):
        """ Select file paths, given the subset and file dictionary. """
        result = self.subset.copy()
        result["path"] = \
            self.subset.apply(
                lambda r: self.file_dict[int(r['cam'])][r['FRAME']], axis=1
            )
        self.file_paths = result

    def generate_output_directory(self):
        """ Copy the files necessary to create the working environment. """
        self.path_to_output_images = \
            os.path.join(self.path_to_output, "images")
        if os.path.isdir(self.path_to_output):
            raise ReconstructionDirGeneratorError(
                "Output directory at "+self.path_to_output+" already exists."
            )
        os.makedirs(self.path_to_output)
        os.mkdir(self.path_to_output_images)
        for _, row in self.file_paths.iterrows():
            image = \
                Image.open(
                    os.path.join(self.path_to_ladybug_images, row["path"])
                )
            image.transpose(Image.ROTATE_270).save(
                os.path.join(self.path_to_output_images, row["path"])
            )

    def make_model(self):
        """ Return the model object. """
        result = \
            Deeplabv3(
                weights=None,
                input_shape=(*self.IMG_SHAPE, 3),
                classes=len(self.LABEL_VALUE_DICT),
                backbone="xception",
                activation="softmax"
            )
        return result

    def get_img_list(self):
        """ Get a list of the images within the output directory. """
        result = \
            get_img_paths(
                self.path_to_output_images,
                CONFIGS.reconstruction_dir.image_extensions
            )
        return result

    def label_images(self):
        """ Make the directory holding the labelled images, and fill it. """
        self.path_to_labelled_images = \
            os.path.join(self.path_to_output, "labelled")
        if not os.path.exists(self.path_to_labelled_images):
            os.makedirs(self.path_to_labelled_images)
        img_list = self.get_img_list()
        logging.info("Labelling %s images...", len(img_list))
        model = self.make_model()
        model.load_weights(self.path_to_model)
        for _, path in enumerate(img_list):
            img = cv2.imread(
                path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            new_img = \
                cv2.resize(
                    img,
                    (img.shape[1]//2, img.shape[0]//2)
                )[0:self.IMG_SHAPE[0], 0:self.IMG_SHAPE[1]]
            new_img = (
                cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)/
                    CONFIGS.general.max_rgb_channel
            )
            prediction = model.predict(numpy.asarray([numpy.array(new_img)]))
            bgr_mask = \
                DigitMapToBGR(
                    self.PALETTE, digit_map=numpy.squeeze(prediction, 0)
                )()
            out_path = \
                os.path.join(
                    self.path_to_labelled_images,
                    os.path.splitext(os.path.split(path)[-1])[0]+
                    self.output_image_extension
                )
            pad_img = \
                cv2.copyMakeBorder(
                    bgr_mask,
                    0, self.BORDER_BOTTOM, 0, 0,
                    cv2.BORDER_CONSTANT,
                    value=self.LABEL_COLOR_DICT["background"]
                )
            out_img = \
                cv2.resize(
                    pad_img,
                    self.PADDED_IMG_SHAPE,
                    interpolation=cv2.INTER_NEAREST
                )
            cv2.imwrite(out_path, out_img)

    def mask_images(self):
        """ Make the directory holding the masked images, and fill it. """
        self.path_to_masked_images = \
            os.path.join(self.path_to_output, "masked")
        if not os.path.exists(self.path_to_masked_images):
            os.makedirs(self.path_to_masked_images)
        img_list = self.get_img_list()
        logging.info("Masking %s images...", len(img_list))
        for _, path in enumerate(img_list):
            _, file_path = os.path.split(path)
            filename, _ = os.path.splitext(file_path)
            path_to_image_to_mask = \
                os.path.join(
                    self.path_to_labelled_images,
                    filename+self.output_image_extension
                )
            out_path = os.path.join(self.path_to_masked_images, file_path)
            if os.path.exists(path_to_image_to_mask):
                img = \
                    cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                mask = cv2.imread(path_to_image_to_mask)
                out = mask_image(img, mask, CONFIGS.general.max_rgb_channel)
                cv2.imwrite(out_path, out)

    def generate_local_selection(self):
        """ Switch over to local coordinates. """
        result = self.file_paths.copy()
        local_centroid = \
            geopandas.GeoSeries(
                geopandas.points_from_xy(
                    [self.centroid[0]], [self.centroid[1]]
                ),
                crs=self.co_ref_sys
            ).to_crs(self.src_co_ref_sys)
        tx, ty = local_centroid.geometry.x, local_centroid.geometry.y
        data = \
            self.file_paths.to_crs(
                self.src_co_ref_sys
            ).translate(-tx, -ty).tolist()
        result["local"] = \
            geopandas.GeoSeries(
                data=data,
                index=self.file_paths.index,
                crs=self.src_co_ref_sys
            )
        self.local_selection = result

    def create_intrinsic(self, cam):
        """ Return a modified copy of the base intrinsic dictionary. """
        result = self.INTRINSIC_BASE.copy()
        result["intrinsicId"] = str(cam)
        result["serialNumber"] = str(cam)
        return result

    def get_view_or_pose_index(self, row):
        """ Ronseal. """
        result = \
            str(int(row["FRAME"])*self.FRAME_ENCODING_FACTOR+int(row["cam"]))
        return result

    def create_view(self, row, base_path):
        """ A "view" in this context is an image. """
        index = self.get_view_or_pose_index(row)
        result = {
            "viewId": index,
            "poseId": index,
            "intrinsicId": str(int(row["cam"])),
            "path": os.path.join(base_path, row["path"]),
            "width": str(self.PADDED_IMG_SHAPE[0]),
            "height": str(self.PADDED_IMG_SHAPE[1]),
            "metadata": ""
        }
        return result

    def create_pose(self, row):
        """ A "pose" in this context refers to how the camera is positioned. """
        index = self.get_view_or_pose_index(row)
        result = {
            "poseId": index,
            "pose": {
                "transform": {
                    "rotation": [str(v) for v in row["rotation"].ravel()],
                    "center": [
                        str(row["local"].x),
                        str(row["local"].y),
                        str(row["altitude"])
                    ]
                },
                "locked": "1"
            }
        }
        return result

    def build_init_dict(self, base_path):
        """ Build the dictionary to be written to the JSON files. """
        intrinsics = [
            self.create_intrinsic(cam) for cam in range(self.number_of_cameras)
        ]
        views = []
        poses = []
        for _, row in self.local_selection.iterrows():
            views.append(self.create_view(row, base_path))
            poses.append(self.create_pose(row))
        result = {
            "version": ["1", "0", "0"],
            "views": views,
            "intrinsics": intrinsics,
            "poses": poses
        }
        return result

    def create_camera_init_file(
            self,
            output_filename,
            base_path,
            encoding=CONFIGS.general.encoding
        ):
        """ Ronseal. """
        init_dict = self.build_init_dict(base_path)
        path_to_output_file = \
            os.path.join(self.path_to_output, output_filename)
        with open(path_to_output_file, "w", encoding=encoding) as fid:
            json.dump(init_dict, fid, indent=self.JSON_INDENT)

    def rename_labelled_images(self):
        """ Rename the images in the "labelled" directory. """
        for _, row in self.local_selection.iterrows():
            filename, _ = os.path.splitext(row["path"])
            file_path = \
                os.path.join(
                    self.path_to_labelled_images,
                    filename+self.output_image_extension
                )
            index = self.get_view_or_pose_index(row)
            new_name = \
                os.path.join(
                    self.path_to_labelled_images,
                    index+self.output_image_extension
                )
            if not os.path.exists(new_name):
                os.rename(file_path, new_name)

    def generate(self):
        """ Generate the reconstruction directory. """
        logging.info("Generating reconstuction directory...")
        logging.info("Loading GPS data...")
        self.load_gps_data()
        logging.info("Localising Ladybug GPS data...")
        self.make_localised_ladybug_gps_data()
        logging.info("Adding geometry...")
        self.make_geo_data_frame()
        logging.info("Calculating focal point...")
        self.make_centroid()
        logging.info("Selecting subset...")
        self.select_the_subset()
        logging.info("Generating file dictionary...")
        self.generate_file_dict()
        logging.info("Selecting file paths...")
        self.select_file_paths()
        logging.info("Generating output directory...")
        self.generate_output_directory()
        self.label_images()
        self.mask_images()
        logging.info("Generating local selection...")
        self.generate_local_selection()
        logging.info("Creating CameraInit files.")
        self.create_camera_init_file(
            self.CAMERA_INIT_FILENAME,
            self.path_to_output_images
        )
        self.create_camera_init_file(
            self.CAMERA_INIT_LABEL_FILENAME,
            self.path_to_masked_images
        )
        logging.info("Renaming labelled images...")
        self.rename_labelled_images()
        logging.info("Reconstruction directory generated.")

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class ReconstructionDirGeneratorError(Exception):
    """ A custom exception. """

class DigitMapToBGR:
    """ This class converts the output from the model to the BGR mask image. """
    def __init__(self, palette, digit_map):
        self.digit_map = digit_map
        self.palette = palette

    def digit_to_color(self, h, w, output_mask):
        """ Convert a digit at the given coordinates to a colour. """
        maximum_channel = get_maximum_channel(self.digit_map[h, w])
        color = self.palette[int(maximum_channel)]
        output_mask[h, w] = color
        return output_mask

    def __call__(self):
        height, weight, _ = self.digit_map.shape
        output_bgr = numpy.zeros([height, weight, 3])
        for h in range(height):
            for w in range(weight):
                output_bgr = self.digit_to_color(h, w, output_bgr)
        return output_bgr

def get_maximum_channel(channel_vector):
    """ Get the maximum channel in a given channel vector. """
    return list(channel_vector).index(max(list(channel_vector)))

def to_geo_data_frame(data_frame, co_ref_sys):
    """ Add geometry to a given data frame. """
    result = \
        geopandas.GeoDataFrame(
            data=data_frame,
            geometry=geopandas.points_from_xy(
                data_frame["longitude"],
                data_frame["latitude"]),
                crs=co_ref_sys
        )
    return result

def filter_by_view(data_frame, centroid):
    """ Filter a given data frame by a given centroid. """
    if not isinstance(centroid, Point):
        centroid = Point(*centroid)
    inview = data_frame.apply(lambda r: centroid.within(r["view"]), axis=1)
    result = data_frame[inview]
    return result

def parse_time(time_string):
    """ Interpret a string giving a time. """
    return datetime.strptime(time_string, "%H:%M:%S.%f")

def seconds_since(origin, time):
    """ Calculate the seconds between two times. """
    result = (time-origin).total_seconds()
    return result

def angle_of_camera_with_index(index):
    """ Get which way a given camera is pointing, relative to the van. """
    result = 2*numpy.pi-(1+(2*index))*numpy.pi/5.
    return result

def find_directions(heading, cam):
    """ Get which way a given camera is pointing, relative to north. """
    heading = numpy.pi*heading/180.
    th = angle_of_camera_with_index(cam)+heading
    if th >= 2*numpy.pi:
        result = th-2*numpy.pi
    elif th < 0:
        result = th+2*numpy.pi
    else:
        result = th
    return result

def get_img_paths(path_to_dir, image_extensions, recursive=True):
    """ Get the paths within a directory corresponding to image files. """
    result = []
    for path in os.listdir(path_to_dir):
        if os.path.isdir(os.path.join(path_to_dir, path)):
            if recursive:
                result = (
                    result+
                    get_img_paths(
                        os.path.join(path_to_dir, path), image_extensions
                    )
                )
        else:
            _, ext = os.path.splitext(path)
            if ext.lower() in image_extensions:
                result.append(os.path.join(path_to_dir, path))
    return result

def mask_image(img, mask, max_rgb_channel):
    """ Apply a given mask to a given image. """
    binmask = numpy.any(mask, axis=2).astype("int")
    maskdim = img*numpy.stack(3*[binmask], axis=2)
    result = cv2.cvtColor(maskdim.astype("uint8"), cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = binmask*max_rgb_channel
    return result
