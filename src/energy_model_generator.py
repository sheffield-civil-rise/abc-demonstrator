"""
This code defines a class which generates the required Intermediate Data File
(IDF) object.
"""

# Standard imports.
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

# Non-standard imports.
import numpy
import pandas
import geopandas
from geomeppy import IDF
from shapely.geometry import Polygon

# Local imports.
import config

# Local constants.
NORTH, EAST, SOUTH, WEST = 0, 90, 180, 270

##############
# MAIN CLASS #
##############

@dataclass
class EnergyModelGenerator:
    """ The class in question. """
    # Fields.
    id_num: int = 0
    path_to_starting_point_idf: str = config.DEFAULT_PATH_TO_STARTING_POINT_IDF
    height: float = None
    window_to_wall_ratio: float = None
    path_to_polygon: str = config.DEFAULT_PATH_TO_POLYGON
    path_to_idd: str = config.DEFAULT_PATH_TO_ENERGYPLUS_INPUT_DATA_DICTIONARY
    path_to_starting_point_idf: str = config.DEFAULT_PATH_TO_STARTING_POINT_IDF
    path_to_weather_file: str = config.DEFAULT_PATH_TO_ENERGYPLUS_WEATHER_FILE
    path_to_output_idf: str = config.DEFAULT_PATH_TO_ENERGYPLUS_OUTPUT_IDF
    path_to_output_dir: str = config.DEFAULT_PATH_TO_ENERGYPLUS_OUTPUT_DIR
    num_stories: int = 2
    orientation: float = 0.0
    save_idf: bool = True
    uvalues: dict = None
    layers: list = None
    window_shgc: float = config.DEFAULT_WINDOW_SHGC # Solar Heat Gain Coeff.
    roughness: str = "MediumRough"
    schedules: dict = None
    air_change_per_hour: float = config.DEFAULT_AIR_CHANGE_PER_HOUR
    setpoint_heating: int = config.DEFAULT_SETPOINT_HEATING
    setpoint_cooling: int = config.DEFAULT_SETPOINT_COOLING
    densities: dict = None
    boiler_type: str = "CondensingHotWaterBoiler"
    boiler_fuel: str = "NaturalGas"
    boiler_efficiency: float = config.DEFAULT_BOILER_EFFICIENCY
    # Generated fields.
    adjusted_height: float = None
    window_to_wall_ratio_dict: dict = None
    idf_obj: IDF = None
    materials: dict = None
    thermostat: IDF = None
    hot_water_loop: IDF = None
    boiler: IDF = None

    # Class attributes.
    MIN_ADJUSTED_HEIGHT: ClassVar[float] = 7.0
    DEFAULT_FACADE_WWR: ClassVar[int] = 0.2
    FAILSAFE_WWR_DICT: ClassVar[dict] = {
        NORTH: 0.1, EAST: 0.3, SOUTH: 0.5, WEST: 0.7
    }
    DEFAULT_UVALUES: ClassVar[dict] = {
        "wall": 2.0,
        "roof": 2.0,
        "floor": 2.0,
        "ceiling": 2.0,
        "window": 2.5
    }
    DEFAULT_LAYERS: ClassVar[list] = [
        "wall", "roof", "floor", "ceiling", "window"
    ]
    DEFAULT_SCHEDULES: ClassVar[dict] = {
        "People":
            "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
            "AllOtherDays,Until: 24:00,1;",
        "Activity":
            "Through: 12/31, For: Weekdays, Until: 24:00,100,For: "+
            "AllOtherDays,Until: 24:00,100;",
        "Lighting":
            "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
            "AllOtherDays,Until: 24:00,1;",
        "Equipment":
            "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
            "AllOtherDays,Until: 24:00,1;",
        "Infiltration":
            "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
            "AllOtherDays,Until: 24:00,1;",
        "Availability":
            "Through: 12/31, For: Weekdays, Until: 24:00,1,For: "+
            "AllOtherDays,Until: 24:00,1;"
    }
    DEFAULT_DENSITIES: ClassVar[dict] = {
        "electric": 8.0, # Watt/m2
        "lighting": 8.0, # Watt/m2
        "people": 0.025
    }
    OUTPUT_SUFFIX: ClassVar[str] = "C"

    def __post_init__(self):
        self.set_adjusted_height()
        self.set_window_to_wall_ratio_dict()
        self.set_uvalues()
        self.set_layers()
        self.set_densities()

    def set_adjusted_height(self):
        # TODO: Ask about why we do this.
        self.adjusted_height = \
            numpy.max(self.MIN_ADJUSTED_HEIGHT, self.height*(2/3))

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
            name=self.id_num,
            coordinates=self.path_to_polygon,
            height=self.height, # Total height of building above ground level.
            num_stories=self.num_stories
        )
        # Might be handy for real world coordination: idf.translate_to_origin()
        # It's possible to have more than one block. The intersect_match()
        # method can deal with this.
        # Surface intersect matching to ensure correct boundary conditions.
        self.idf_obj.intersect_match()

    def generate_material_idf(self, material_name, uvalue):
        """ Generate an "inner" IDF object for a given material. """
        if name == "window":
            return None  # Create special one for windows.
        result = \
            self.idf_obj.newidfobject(
                "MATERIAL:NOMASS",
                Name=material_name,
                Roughness=self.roughness,
                Thermal_Resistance=1.0/uvalue
            )
        return result

    def define_materials(self):
        # TODO: Ask about filling in this docstring.
        self.materials = {
            k.capitalize():
                self.generate_material_idf(name, uvalue)
                for name, uvalue in self.uvalues.items()
        }
        self.materials["Window"] = \
            self.idf_obj.newidfobject(
                "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
                Name="window",
                UFactor=self.uvalues["window"],
                Solar_Heat_Gain_Coefficient=shgc
            )

    def generate_construction_idf(self, construction_name):
        """ Generate an "inner" IDF object for a given material. """
        result = \
            self.idf_obj.newidfobject(
                "CONSTRUCTION",
                Name="%s_construction"%construction_name,
                Outside_Layer=construction_name
            )
        return result

    def define_construction(self):
        # TODO: Ask about filling in this docstring.
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
        result["People"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="People schedule",
                Schedule_Type_Limits_Name="Fraction",
                Field_1=self.schedules["People"]
            )
        result["Activity"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Activity level schedule",
                Schedule_Type_Limits_Name="Activity",
                Field_1=self.schedules["Activity"]
            )
        result["Lighting"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Lighting schedule",
                Schedule_Type_Limits_Name="Fraction",
                Field_1=self.schedules["Lighting"]
            )
        result["Equipment"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Equipment schedule",
                Schedule_Type_Limits_Name="Fraction",
                Field_1=self.schedules["Equipment"]
            )
        result["Infiltration"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Infiltration schedule",
                Schedule_Type_Limits_Name="Fraction",
                Field_1=self.schedules["Infiltration"]
            )
        result["Availability"] = \
            self.idf_obj.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Heating availability schedule",
                Schedule_Type_Limits_Name="Fraction",
                Field_1=self.schedules["Availability"]
            )
        return result

    def add_thermostat(self):
        """ Thermostat for HVAC control. """
        self.thermostat = \
            self.idf_obj.newidfobject(
                "HVACTEMPLATE:THERMOSTAT",
                Name="Zone Thermostat",
                Constant_Heating_Setpoint=self.setpoint_heating,
                Constant_Cooling_Setpoint=self.setpoint_cooling
            )

    def add_hot_water_loop(self):
        """ Hot water loop for heating radiators. """
        self.hot_water_loop = \
            self.idf_obj.newidfobject(
                "HVACTEMPLATE:PLANT:HOTWATERLOOP",
                Name="Space heating hot water loop"
            )

    def add_boiler(self):
        """ Add a object representing the building's boiler. """
        self.boiler = \
            self.idf_obj.newidfobject(
                "HVACTEMPLATE:PLANT:BOILER",
                Name="Boiler",
                Boiler_Type=self.boiler_type,
                Efficiency=self.boiler_efficiency,
                Fuel_Type=self.boiler_fuel
            )

    def populate_zones(self):
        # TODO: Ask about filling in this docstring.
        for zone in self.idf_obj.idfobjects["ZONE"]:
            # Add radiator in every zone.
            self.idf_obj.newidfobject(
                "HVACTEMPLATE:ZONE:BASEBOARDHEAT",
                Zone_Name=zone.Name,
                Template_Thermostat_Name=self.thermostat.Name,
                Baseboard_Heating_Availability_Schedule_Name=\
                    self.schedules["Availability"].Name,
                Outdoor_Air_Method="Flow/Person",
                Outdoor_Air_Flow_Rate_per_Person=0 # m3/s
            )
            # No additional outdoor air supply apart from the infiltration.
            # Define infiltration for each zone.
            self.idf_obj.newidfobject(
                "ZONEINFILTRATION:DESIGNFLOWRATE",
                Name="Infiltration: "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=self.schedules["Infiltration"].Name,
                Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
                Air_Changes_per_Hour=self.air_change_per_hour
            )
            # Define occupants for each zone.
            self.idf_obj.newidfobject(
                "PEOPLE",
                Name="People: "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Number_of_People_Schedule_Name=self.schedules["People"].Name,
                Number_of_People_Calculation_Method="People/Area",
                People_per_Zone_Floor_Area=self.densities["people"],
                Activity_Level_Schedule_Name=self.schedules["Activity"].Name
            )
            # Define lights for each zone.
            self.idf_obj.newidfobject(
                "LIGHTS",
                Name="Light: "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=self.schedules["Lighting"].Name,
                Design_Level_Calculation_Method="Watts/Area",
                Watts_per_Zone_Floor_Area=self.densities["lighting"]
            )
            # Define equipment for each zone.
            self.idf_obj.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="Electricequipment: "+zone.Name,
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name=self.schedules['Equipment'].Name,
                Design_Level_Calculation_Method="Watts/Area",
                Watts_per_Zone_Floor_Area=self.densities["electric"]
            )

    def generate(self):
        """ Generate the model. """
        self.initialise_idf()
        self.define_geometry()
        self.idf_obj.set_wwr(wwr_map=self.window_to_wall_ratio_dict)
        self.idf_obj.idfobjects["BUILDING"][0].North_Axis = self.orientation
        self.define_materials()
        self.define_schedules()
        self.add_thermostat()
        self.add_hot_water_loop()
        self.add_boiler()

    def run_simulation(self):
        """ Run the EnergyPlus simulation and save the simulation output to a
        predefined direction. """
        if not os.path.isdir(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
        self.idf_obj.run(
            output_directory=self.path_to_output_dir,
            expandobjects=True,
            output_prefix=self.id_num,
            output_suffix=self.OUTPUT_SUFFIX
        )

    def generate_and_run(self):
        """ Generate our IDF object, and then run it. """
        self.generate()
        if self.save_idf:
            self.idf_obj.save(self.path_to_idf_output)
        self.run_simulation()

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class EnergyModelGeneratorError(Exception):
    """ A custom exception. """
    pass
