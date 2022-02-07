"""
This code defines a class which generates the reconstruction directory.
"""

# Standard imports.
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

# Non-standard imports.
import cv2
import geopandas
import numpy
import pandas
from PIL import Image
from scipy.interpolate import interp1d
from shapely.geometry import Point, Polygon

# Local imports.
import config
from deeplab.Deeplabv3 import Deeplabv3

##############
# MAIN CLASS #
##############

@dataclass
class ReconstructionDirGenerator:
    """ The class in question. """
    # Fields.
    path_to_gps_data: str = config.DEFAULT_PATH_TO_GPS_DATA
    path_to_ladybug_gps_data: str = config.DEFAULT_PATH_TO_LADYBUG_GPS_DATA
    path_to_ladybug_images: str = config.DEFAULT_PATH_TO_LADYBUG_IMAGES
    path_to_polygon: str = config.DEFAULT_PATH_TO_POLYGON
    path_to_output: str = config.DEFAULT_PATH_TO_OUTPUT
    co_ref_sys: str = config.DEFAULT_COORDINATE_REFERENCE_SYSTEM
    src_co_ref_sys: str = config.DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM
    radius: int = config.DEFAULT_RADIUS
    view_distance: int = config.DEFAULT_VIEW_DISTANCE
    field_of_view: float = config.DEFAULT_FIELD_OF_VIEW
    # Generated fields.
    gps_data: pandas.DataFrame = None
    ladybug_data: pandas.DataFrame = None
    localised_ladybug_data: pandas.DataFrame = None
    geo_data_frame: geopandas.GeoDataFrame = None
    centroid: list = None
    subset: geopandas.GeoDataFrame = None
    file_dict: dict = None
    file_paths: geopandas.GeoDataFrame = None

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

    def make_localised_ladybug_gps_data(self):
        """ Localise the Ladybyg GPS data using the GPS CSV file. """
        self.interpolate_gps_data()
        append_orientation(self.localised_gps_data, inplace=True)

    def make_geo_data_frame(self):
        """ Add geometry to our localised data. """
        self.geo_data_frame = \
            to_geo_data_frame(
                self.localised_gps_data,
                co_ref_sys=self.co_ref_sys
            )

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
        for index, row in data_frame.iterrows():
            rotations = row["rotations"]
            nrow = row[columns]
            for cam in range(len(rotations)):
                nrow["cam"] = int(cam)
                nrow["rotation"] = rotations[cam]
                if type(data_frame) is not geopandas.GeoDataFrame:
                    geox.append(row["longitude"])
                    geoy.append(row["latitude"])
                else:
                    geox.append(row.geometry.x)
                    geoy.append(row.geometry.y)
                new_data_frame.append(nrow, ignore_index=True)
        local_co_ref_sys = self.co_ref_sys
        if type(data_frame) is geopandas.GeoDataFrame:
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
                angle0-self.field_of_view/2,
                angle0+self.SEGMENT_RESOLUTION/2, res
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
        circle = \
            create_circle(
                self.centroid,
                radius=self.radius,
                co_ref_sys=self.co_ref_sys,
                src_co_ref_sys=self.src_co_ref_sys,
                aspoints=False
            )
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

    def generate_file_dict(self, path_to):
        """ Generate a file dictionary given a path to a directory. """
        file_list = os.listdir(path_to)
        result = [{}, {}, {}, {}, {}]
        for index, filename in enumerate(file_list):
            frame, cam = self.framecam(filename)
            if int(cam) < self.MAX_CAM_INDEX:
                result[int(cam)][frame] = filename
        self.file_dict = result

    def select_file_paths(self):
        """ Select file paths, given the subset and file dictionary. """
        result = self.subset.copy()
        result["path"] = \
            self.subset.apply(
                lambda r: self.filedict[int(r['cam'])][r['FRAME']], axis=1
            )
        self.file_paths = result

    def generate_output_directory(self):
        """ Copy the files necessary to create the working environment. """
        if os.path.isdir(self.path_to_output):
            raise ReconstructionDirGeneratorError(
                "Output directory at "+self.path_to_output+" already exists."
            )
        os.makedirs(self.path_to_output)
        new_img_dir = os.path.join(self.path_to_output, "images")
        os.mkdir(new_img_dir)
        for index, row in self.file_paths.iterrows():
            image = \
                Image.open(
                    os.path.join(self.path_to_ladybug_images, row["path"])
                )
            image.transpose(Image.ROTATE_270).save(
                os.path.join(new_img_dir, row["path"])
            )

    def generate(self):
        """ Generate the reconstruction directory. """
        print("Loading GPS data...")
        self.load_gps_data()
        print("Localising Ladybug GPS data...")
        self.make_localised_ladybug_gps_data()
        print("Adding geometry...")
        self.make_geo_data_frame()
        print("Calculating focal point...")
        self.make_centroid()
        print("Selecting subset...")
        self.select_subset()
        print("Generating file dictionary...")
        self.generate_file_dict()
        print("Selecting file paths...")
        self.select_file_paths()
        print("Generate output directory...")
        self.generate_output_directory()
        print("THIS IS WHERE THE ORIGINAL SCRIPT CRASHES")
        #print("labelling images")
        #label_directory(
        #    os.path.join(args.out, "images"),
        #    os.path.join(args.out, "labels")
        #)
        #print("\nmasking images")
        #mask_all_images(
        #    os.path.join(args.out, "images"),
        #    os.path.join(args.out, "labels"),
        #    os.path.join(args.out, "masked")
        #)
        #print("\ncreating cameraInit files")
        #local_selection = generate_local_coords(selection, centroid)
        #generate_cameraInit(
        #    local_selection,
        #    os.path.join(args.out, "images"),
        #    output=os.path.join(args.out, 'cameraInit.sfm')
        #)
        #generate_cameraInit(
        #    local_selection,
        #    os.path.join(args.out, "masked"),
        #    output=os.path.join(args.out, 'cameraInit_label.sfm')
        #)
        #print("renaming label data")
        #rename_labels(local_selection, os.path.join(args.out, "labels"))
        #print("done.")
        #return os.path.abspath(args.out)

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class ReconstructionDirGeneratorError(Exception):
    """ A custom exception. """
    pass

def append_orientation(data_frame, inplace=False):
    """ Append camera orientations to a given data frame. """
    new_data_frame = data_frame.copy()
    new_data_frame["rotations"] = None # Initialise empty column.
    for index, row in data_frame.iterrows():
        rotations = build_rotation(row["heading"], row["pitch"], row["roll"])
        new_data_frame.at[index, "rotations"] = rotations
    if inplace: # Append rotations to input dataframe.
        data_frame["rotations"] = ndf["rotations"].values
    else: # return copy with added rotations
        return new_data_frame

def to_geo_data_frame(
        data_frame,
        co_ref_sys=config.DEFAULT_COORDINATE_REFERENCE_SYSTEM
    ):
    """ Add geometry to a given data frame. """
    result = \
        geopandas.GeoDataFrame(
            data=data_frame,
            geometry=geopandas.points_from_xy(
                df['longitude'],
                df['latitude']),
                crs=co_ref_sys
        )
    return result

def create_circle(
        centroid,
        radius=config.DEFAULT_RADIUS,
        co_ref_sys=config.DEFAULT_COORDINATE_REFERENCE_SYSTEM,
        src_co_ref_sys=config.DEFAULT_SOURCE_COORDINATE_REFERENCE_SYSTEM,
        resolution=config.DEFAULT_CIRCLE_RESOLUTION,
        aspoints=False
    ):
    """ Return a circle object. """
    centre = Point(centroid)
    circle_data_frame = \
        geopandas.GeoDataFrame(
            { "geometry": [centre] },
            crs=co_ref_sys
        ).to_crs(src_co_ref_sys)
    points = [
        (
            radius*numpy.sin(angle)+circle_data_frame.geometry.x[0],
            radius*numpy.cos(angle)+circle_data_frame.geometry.y[0]
        ) for angle in numpy.linspace(0, 2*numpy.pi, resolution)
    ]
    polygon = Polygon(points)
    geometry = [Point(point) for point in points] if aspoints else [polygon]
    result = \
        geopandas.GeoDataFrame(
            { "geometry": geometry },
            crs=src_co_ref_sys
        ).to_crs(co_ref_sys)
    return result

def filter_by_view(data_frame, centroid):
    """ Filter a given data frame by a given centroid. """
    if type(centroid) is not Point:
        centroid = Point(*centroid)
    inview = data_frame.apply(lambda r: centroid.within(r["view"]), axis=1)
    result = df[inview]
    return result
