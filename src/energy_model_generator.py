"""
This code defines a class which generates the required Intermediate Data File
(IDF) object.
"""

# Standard imports.
import argparse
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

# Non-standard imports.
import numpy
import pandas
import geopandas
from geomeppy import IDF
from shapely.geometry import Polygon

# Local imports.
from local_configs import CONFIGS

# Local constants.
NORTH, EAST, SOUTH, WEST = 0, 90, 180, 270

##############
# MAIN CLASS #
##############

@dataclass
class EnergyModelGenerator:
    """ The class in question. """
    # Fields.
    id_str: str = "demo"
    path_to_starting_point_idf: str = CONFIGS.paths.path_to_starting_point_idf
    height: float = None
    window_to_wall_ratio: float = None
    path_to_polygon: str = CONFIGS.paths.path_to_polygon
    path_to_idd: str = CONFIGS.paths.path_to_energyplus_input_data_dictionary
    path_to_weather_file: str = CONFIGS.paths.path_to_energyplus_weather_file
    path_to_output_idf: str = CONFIGS.paths.path_to_output_idf
    path_to_output_dir: str = CONFIGS.paths.path_to_energy_model_output_dir
    num_stories: int = 2
    orientation: float = 0.0
    save_idf: bool = True
    uvalues: dict = None
    layers: list = None
    window_shgc: float = CONFIGS.energy_model.window_shgc
    roughness: str = "MediumRough"
    schedules: dict = None
    air_change_per_hour: float = CONFIGS.energy_model.air_change_per_hour
    setpoint_heating: int = CONFIGS.energy_model.setpoint_heating
    setpoint_cooling: int = CONFIGS.energy_model.setpoint_cooling
    densities: dict = None
    boiler_type: str = "CondensingHotWaterBoiler"
    boiler_fuel: str = "NaturalGas"
    boiler_efficiency: float = CONFIGS.energy_model.boiler_efficiency
    src_co_ref_sys: str = CONFIGS.general.source_coordinate_reference_system
    # Generated fields.
    adjusted_height: float = None
    window_to_wall_ratio_dict: dict = None
    polygon: Polygon = None
    idf_obj: IDF = None
    materials: dict = None
    constructions: dict = None
    thermostat: IDF = None
    hot_water_loop: IDF = None
    boiler: IDF = None

    # Class attributes.
    MIN_ADJUSTED_HEIGHT: ClassVar[float] = 7.0
    DEFAULT_FACADE_WWR: ClassVar[int] = 0.2
    OUTPUT_SUFFIX: ClassVar[str] = "C"
    # Schedule keys.
    ACTIVITY_SCHEDULE_KEY: ClassVar[str] = "Activity"
    AVAILABILITY_SCHEDULE_KEY: ClassVar[str] = "Availability"
    EQUIPMENT_SCHEDULE_KEY: ClassVar[str] = "Equipment"
    INFILTRATION_SCHEDULE_KEY: ClassVar[str] = "Infiltration"
    LIGHTING_SCHEDULE_KEY: ClassVar[str] = "Lighting"
    PEOPLE_SCHEDULE_KEY: ClassVar[str] = "People"
    # IDF object keys.
    BOILER_IDF_OBJECT_KEY_STRING: ClassVar[str] = "HVACTEMPLATE:PLANT:BOILER"
    BUILDING_IDF_OBJECT_KEY_STRING: ClassVar[str] = "BUILDING"
    CONSTRUCTION_IDF_OBJECT_KEY_STRING = "CONSTRUCTION"
    EQUIPMENT_IDF_OBJECT_KEY_STRING: ClassVar[str] = "ELECTRICEQUIPMENT"
    GLAZING_IDF_OBJECT_KEY_STRING: ClassVar[str] = \
        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"
    HOT_WATER_LOOP_IDF_OBJECT_KEY_STRING: ClassVar[str] = \
        "HVACTEMPLATE:PLANT:HOTWATERLOOP"
    INFILTRATION_IDF_OBJECT_KEY_STRING: ClassVar[str] = \
        "ZONEINFILTRATION:DESIGNFLOWRATE"
    LIGHTING_IDF_OBJECT_KEY_STRING: ClassVar[str] = "LIGHTS"
    MATERIAL_IDF_OBJECT_KEY_STRING: ClassVar[str] = "MATERIAL:NOMASS"
    PEOPLE_IDF_OBJECT_KEY_STRING: ClassVar[str] = "PEOPLE"
    RADIATORS_IDF_OBJECT_KEY_STRING: ClassVar[str] = \
        "HVACTEMPLATE:ZONE:BASEBOARDHEAT"
    SCHEDULE_IDF_OBJECT_KEY_STRING = "SCHEDULE:COMPACT"
    THERMOSTAT_IDF_OBJECT_KEY_STRING: ClassVar[str] = "HVACTEMPLATE:THERMOSTAT"
    ZONE_IDF_OBJECT_KEY_STRING: ClassVar[str] = "ZONE"
    # Dictionaries and lists.
    DEFAULT_DENSITIES: ClassVar[dict] = None
    DEFAULT_LAYERS: ClassVar[list] = [
        "wall", "roof", "floor", "ceiling", "window"
    ]
    DEFAULT_POLYGON: ClassVar[list] = [(6, 0), (6, 6), (0, 6), (0, 0)]
    DEFAULT_SCHEDULES: ClassVar[dict] = None
    DEFAULT_UVALUES: ClassVar[dict] = {
        "wall": 2.0,
        "roof": 2.0,
        "floor": 2.0,
        "ceiling": 2.0,
        "window": 2.5
    }
    FAILSAFE_WWR_DICT: ClassVar[dict] = {
        NORTH: 0.1, EAST: 0.3, SOUTH: 0.5, WEST: 0.7
    }
    SCHEDULE_KEY_STRINGS: ClassVar[dict] = None
    # IDF object names.
    WINDOW_IDF_OBJECT_NAME: ClassVar[str] = "Window"
    THERMOSTAT_IDF_OBJECT_NAME: ClassVar[str] = "Zone Thermostat"
    HOT_WATER_LOOP_IDF_OBJECT_NAME: ClassVar[str] = \
        "Space heating hot water loop"
    BOILER_IDF_OBJECT_NAME: ClassVar[str] = "Boiler"
    # Method names.
    DEFAULT_OUTDOOR_AIR_METHOD: ClassVar[str] = "Flow/Person"
    DEFAULT_FLOW_RATE_CALCULATION_METHOD: ClassVar[str] = "AirChanges/Hour"
    DEFAULT_POPULATION_CALCULATION_METHOD: ClassVar[str] = "People/Area"
    DEFAULT_DESIGN_LEVEL_CALCULATION_METHOD: ClassVar[str] = "Watts/Area"

    def __post_init__(self):
        self.make_default_schedules()
        self.make_schedule_key_strings()
        self.make_default_densities()
        self.set_adjusted_height()
        self.set_window_to_wall_ratio_dict()
        self.set_uvalues()
        self.set_layers()
        self.set_densities()
        self.read_polygon()

    def make_default_schedules(self):
        """ Create the class attribute. """
        self.DEFAULT_SCHEDULES = {
            self.PEOPLE_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
                "AllOtherDays,Until: 24:00,1;",
            self.ACTIVITY_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,100,For: "+
                "AllOtherDays,Until: 24:00,100;",
            self.LIGHTING_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
                "AllOtherDays,Until: 24:00,1;",
            self.EQUIPMENT_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
                "AllOtherDays,Until: 24:00,1;",
            self.INFILTRATION_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
                "AllOtherDays,Until: 24:00,1;",
            self.AVAILABILITY_SCHEDULE_KEY:
                "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
                "AllOtherDays,Until: 24:00,1;"
        }

    def make_schedule_key_strings(self):
        """ Create the class attribute. """
        self.SCHEDULE_KEY_STRINGS = {
            self.PEOPLE_SCHEDULE_KEY: {
                "name": "People schedule", "type_limits": "Fraction"
            },
            self.ACTIVITY_SCHEDULE_KEY: {
                "name": "Activity level schedule", "type_limits": "Activity"
            },
            self.LIGHTING_SCHEDULE_KEY: {
                "name": "Lighting schedule", "type_limits": "Fraction"
            },
            self.EQUIPMENT_SCHEDULE_KEY: {
                "name": "Equipment schedule", "type_limits": "Fraction"
            },
            self.INFILTRATION_SCHEDULE_KEY: {
                "name": "Infiltration schedule", "type_limits": "Fraction"
            },
            self.AVAILABILITY_SCHEDULE_KEY: {
                "name": "Heating availability schedule",
                "type_limits": "Fraction"
            }
        }

    def make_default_densities(self):
        """ Create the class attribute. """
        self.DEFAULT_DENSITIES = {
            self.EQUIPMENT_SCHEDULE_KEY: 8.0, # Watt/m2
            self.LIGHTING_SCHEDULE_KEY: 8.0, # Watt/m2
            self.PEOPLE_SCHEDULE_KEY: 0.025
        }

    def set_adjusted_height(self):
        """ Make sure the height is at least the minimum. """
        self.adjusted_height = \
            numpy.max([self.MIN_ADJUSTED_HEIGHT, self.height])

    def set_window_to_wall_ratio_dict(self):
        """ Create a dictionary giving the window-to-wall ratio for each
        facade, from a single value. """
        if not self.window_to_wall_ratio_dict:
            if self.window_to_wall_ratio:
                self.window_to_wall_ratio_dict = {
                    NORTH: self.window_to_wall_ratio,
                    EAST: self.DEFAULT_FACADE_WWR,
                    SOUTH: self.window_to_wall_ratio,
                    WEST: self.DEFAULT_FACADE_WWR
                }
            else:
                self.window_to_wall_ratio_dict = \
                    deepcopy(self.FAILSAFE_WWR_DICT)

    def set_uvalues(self):
        """ Set this dictionary. """
        if not self.uvalues:
            self.uvalues = deepcopy(self.DEFAULT_UVALUES)

    def set_layers(self):
        """ Set this list. """
        if not self.layers:
            self.layers = deepcopy(self.DEFAULT_LAYERS)

    def set_densities(self):
        """ Set this dictionary. """
        if not self.densities:
            self.densities = deepcopy(self.DEFAULT_DENSITIES)

    def read_polygon(self):
        """ Read the polygon from a file, if possible. """
        if self.path_to_polygon:
            data_frame = pandas.read_csv(self.path_to_polygon, header=None)
            geometry = geopandas.points_from_xy(data_frame[1], data_frame[0])
            geo_data_frame = \
                geopandas.GeoDataFrame(
                    geometry=geometry,
                    crs=self.src_co_ref_sys
                )
            bounds = Polygon(geo_data_frame.geometry).bounds
            poly = geo_data_frame.geometry.translate(-bounds[0], -bounds[1])
            self.polygon = list(zip(poly.x, poly.y))
        else:
            self.polygon = deepcopy(self.DEFAULT_POLYGON)

    def initialise_idf(self):
        """ Initialise our Intermediate Data Format object. """
        IDF.setiddname(self.path_to_idd)
        self.idf_obj = IDF(self.path_to_starting_point_idf)
        self.idf_obj.epw = self.path_to_weather_file

    def define_geometry(self):
        """ Add initial geometry. """
        # NOTES:
        #     (1) The vertex for coordination can be given clockwise or
        #         anti-clockwise.
        #     (2) The floor plan can be a non-convex surface.
        self.idf_obj.add_block(
            name=self.id_str,
            coordinates=self.polygon,
            height=self.adjusted_height, # Full height of building above ground.
            num_stories=self.num_stories
        )
        # Might be handy for real world coordination: idf.translate_to_origin()
        # It's possible to have more than one block. The intersect_match()
        # method can deal with this.
        # Surface intersect matching to ensure correct boundary conditions.
        self.idf_obj.intersect_match()

    def generate_material_idf(self, material_name, uvalue):
        """ Generate an "inner" IDF object for a given material. """
        if material_name == "window":
            return None  # Create special one for windows.
        result = \
            self.idf_obj.newidfobject(
                self.MATERIAL_IDF_OBJECT_KEY_STRING,
                Name=material_name,
                Roughness=self.roughness,
                Thermal_Resistance=1.0/uvalue
            )
        return result

    def define_materials(self):
        """ Add some information on materials' thermoconductivity. """
        self.materials = {
            name.capitalize():
                self.generate_material_idf(name, uvalue)
                for name, uvalue in self.uvalues.items()
        }
        self.materials[self.WINDOW_IDF_OBJECT_NAME] = \
            self.idf_obj.newidfobject(
                self.GLAZING_IDF_OBJECT_KEY_STRING,
                Name="window",
                UFactor=self.uvalues["window"],
                Solar_Heat_Gain_Coefficient=self.window_shgc
            )

    def generate_construction_idf(self, construction_name):
        """ Generate an "inner" IDF object for a given material. """
        result = \
            self.idf_obj.newidfobject(
                self.CONSTRUCTION_IDF_OBJECT_KEY_STRING,
                Name=construction_name+"_construction",
                Outside_Layer=construction_name
            )
        return result

    def define_constructions(self):
        """ Define what is where, and what it's made of. """
        self.constructions = {
            name.capitalize():
                self.generate_construction_idf(name) for name in self.layers
        }
        for name in self.layers:
            # Assign construction to building surface and sub-surface.
            if name == "window":
                for layer in self.idf_obj.getsubsurfaces(name):
                    layer.Construction_Name = \
                        self.constructions[name.capitalize()].Name
            else:
                for layer in self.idf_obj.getsurfaces(name):
                    layer.Construction_Name = \
                        self.constructions[name.capitalize()].Name

    def define_schedules(self):
        """ Set this dictionary. """
        if not self.schedules:
            self.schedules = deepcopy(self.DEFAULT_SCHEDULES)
        else: # Fill in with defaults if any missing.
            for key, value in self.DEFAULT_SCHEDULES.items():
                if key not in self.schedules:
                    self.schedules[key] = value
        result = {} # Output IDF objects.
        for key in self.SCHEDULE_KEY_STRINGS.keys():
            sub_dict = self.SCHEDULE_KEY_STRINGS[key]
            result[key] = \
                self.idf_obj.newidfobject(
                    self.SCHEDULE_IDF_OBJECT_KEY_STRING,
                    Name=sub_dict["name"],
                    Schedule_Type_Limits_Name=sub_dict["type_limits"],
                    Field_1=self.schedules[key]
                )
        self.schedules = result

    def add_thermostat(self):
        """ Thermostat for HVAC control. """
        self.thermostat = \
            self.idf_obj.newidfobject(
                self.THERMOSTAT_IDF_OBJECT_KEY_STRING,
                Name=self.THERMOSTAT_IDF_OBJECT_NAME,
                Constant_Heating_Setpoint=self.setpoint_heating,
                Constant_Cooling_Setpoint=self.setpoint_cooling
            )

    def add_hot_water_loop(self):
        """ Hot water loop for heating radiators. """
        self.hot_water_loop = \
            self.idf_obj.newidfobject(
                self.HOT_WATER_LOOP_IDF_OBJECT_KEY_STRING,
                Name=self.HOT_WATER_LOOP_IDF_OBJECT_NAME
            )

    def add_boiler(self):
        """ Add a object representing the building's boiler. """
        self.boiler = \
            self.idf_obj.newidfobject(
                self.BOILER_IDF_OBJECT_KEY_STRING,
                Name=self.BOILER_IDF_OBJECT_NAME,
                Boiler_Type=self.boiler_type,
                Efficiency=self.boiler_efficiency,
                Fuel_Type=self.boiler_fuel
            )

    def populate_zones(self):
        """ Populate our model with stuff such as people and lights. """
        for zone in self.idf_obj.idfobjects[self.ZONE_IDF_OBJECT_KEY_STRING]:
            # Add radiator in every zone.
            self.idf_obj.newidfobject(
                self.RADIATORS_IDF_OBJECT_KEY_STRING,
                Zone_Name=zone.Name,
                Template_Thermostat_Name=self.thermostat.Name,
                Baseboard_Heating_Availability_Schedule_Name=\
                    self.schedules[self.AVAILABILITY_SCHEDULE_KEY].Name,
                Outdoor_Air_Method=self.DEFAULT_OUTDOOR_AIR_METHOD,
                Outdoor_Air_Flow_Rate_per_Person=0 # m3/s
            )
            # No additional outdoor air supply apart from the infiltration.
            # Define infiltration for each zone.
            self.idf_obj.newidfobject(
                self.INFILTRATION_IDF_OBJECT_KEY_STRING,
                Name=self.INFILTRATION_SCHEDULE_KEY+": "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=\
                    self.schedules[self.INFILTRATION_SCHEDULE_KEY].Name,
                Design_Flow_Rate_Calculation_Method=\
                    self.DEFAULT_FLOW_RATE_CALCULATION_METHOD,
                Air_Changes_per_Hour=self.air_change_per_hour
            )
            # Define occupants for each zone.
            self.idf_obj.newidfobject(
                self.PEOPLE_IDF_OBJECT_KEY_STRING,
                Name=self.PEOPLE_SCHEDULE_KEY+": "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Number_of_People_Schedule_Name=\
                    self.schedules[self.PEOPLE_SCHEDULE_KEY].Name,
                Number_of_People_Calculation_Method=\
                    self.DEFAULT_POPULATION_CALCULATION_METHOD,
                People_per_Zone_Floor_Area=\
                    self.densities[self.PEOPLE_SCHEDULE_KEY],
                Activity_Level_Schedule_Name=\
                    self.schedules[self.ACTIVITY_SCHEDULE_KEY].Name
            )
            # Define lights for each zone.
            self.idf_obj.newidfobject(
                self.LIGHTING_IDF_OBJECT_KEY_STRING,
                Name=self.LIGHTING_SCHEDULE_KEY+": "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=self.schedules[self.LIGHTING_SCHEDULE_KEY].Name,
                Design_Level_Calculation_Method=\
                    self.DEFAULT_DESIGN_LEVEL_CALCULATION_METHOD,
                Watts_per_Zone_Floor_Area=\
                    self.densities[self.LIGHTING_SCHEDULE_KEY]
            )
            # Define equipment for each zone.
            self.idf_obj.newidfobject(
                self.EQUIPMENT_IDF_OBJECT_KEY_STRING,
                Name=self.EQUIPMENT_SCHEDULE_KEY+": "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=self.schedules[self.EQUIPMENT_SCHEDULE_KEY].Name,
                Design_Level_Calculation_Method=\
                    self.DEFAULT_DESIGN_LEVEL_CALCULATION_METHOD,
                Watts_per_Zone_Floor_Area=\
                    self.densities[self.EQUIPMENT_SCHEDULE_KEY]
            )

    def generate(self):
        """ Generate the model. """
        self.initialise_idf()
        self.define_geometry()
        self.idf_obj.set_wwr(wwr_map=self.window_to_wall_ratio_dict)
        self.idf_obj.idfobjects[
            self.BUILDING_IDF_OBJECT_KEY_STRING
        ][0].North_Axis = self.orientation
        self.define_materials()
        self.define_constructions()
        self.define_schedules()
        self.add_thermostat()
        self.add_hot_water_loop()
        self.add_boiler()
        self.populate_zones()

    def run_simulation(self):
        """ Run the EnergyPlus simulation and save the simulation output to a
        predefined direction. """
        if not os.path.isdir(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
        self.idf_obj.run(
            output_directory=self.path_to_output_dir,
            expandobjects=True,
            output_prefix=self.id_str,
            output_suffix=self.OUTPUT_SUFFIX
        )

    def generate_and_run(self):
        """ Generate our IDF object, and then run it. """
        self.generate()
        if self.save_idf:
            self.idf_obj.save(self.path_to_output_idf)
        self.run_simulation()

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class EnergyModelGeneratorError(Exception):
    """ A custom exception. """

def make_parser():
    """ Return a parser argument. """
    result = \
        argparse.ArgumentParser(
            description="Make and run an EnergyModelGenerator object"
        )
    result.add_argument(
        "--height",
        default=None,
        dest="height",
        help="The height of the building",
        type=float
    )
    result.add_argument(
        "--wwr",
        default=None,
        dest="wwr",
        help="The window-to-wall ratio",
        type=float
    )
    result.add_argument(
        "--path-to-output-idf",
        default=None,
        dest="path_to_output_idf",
        help="The path to the output IDF file",
        type=str
    )
    result.add_argument(
        "--path-to-output-dir",
        default=None,
        dest="path_to_output_dir",
        help="The path to the output directory",
        type=str
    )
    result.add_argument(
        "--path-to-polygon",
        default=None,
        dest="path_to_polygon",
        help="The path to the polygon file",
        type=str
    )
    return result

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    parser = make_parser()
    arguments = parser.parse_args()
    energy_model_generator = \
        EnergyModelGenerator(
            height=arguments.height,
            window_to_wall_ratio=arguments.wwr,
            path_to_output_idf=arguments.path_to_output_idf,
            path_to_output_dir=arguments.path_to_output_dir,
            path_to_polygon=arguments.path_to_polygon
        )
    energy_model_generator.generate_and_run()

if __name__ == "__main__":
    run()
