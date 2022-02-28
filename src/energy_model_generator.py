"""
This code defines a class which generates the required Intermediate Data File
(IDF) object.
"""

# Standard imports.
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
from config import get_configs

# Local constants.
NORTH, EAST, SOUTH, WEST = 0, 90, 180, 270
CONFIGS = get_configs()

##############
# MAIN CLASS #
##############

@dataclass
class EnergyModelGenerator:
    """ The class in question. """
    # Fields.
    id_str: str = "demo"
    path_to_starting_point_idf: str = \
        CONFIGS.energy_model.path_to_starting_point_idf
    height: float = None
    window_to_wall_ratio: float = None
    path_to_polygon: str = CONFIGS.general.path_to_polygon
    path_to_idd: str = \
        CONFIGS.energy_model.path_to_energyplus_input_data_dictionary
    path_to_weather_file: str = \
        CONFIGS.energy_model.path_to_energyplus_weather_file
    path_to_output_idf: str = CONFIGS.energy_model.path_to_output_idf
    path_to_output_dir: str = \
        CONFIGS.energy_model.path_to_energy_model_output_dir
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
    DEFAULT_POLYGON: ClassVar[list] = [(6, 0), (6, 6), (0, 6), (0, 0)]

    def __post_init__(self):
        self.set_adjusted_height()
        self.set_window_to_wall_ratio_dict()
        self.set_uvalues()
        self.set_layers()
        self.set_densities()
        self.read_polygon()

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
                "MATERIAL:NOMASS",
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
        self.materials["Window"] = \
            self.idf_obj.newidfobject(
                "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
                Name="window",
                UFactor=self.uvalues["window"],
                Solar_Heat_Gain_Coefficient=self.window_shgc
            )

    def generate_construction_idf(self, construction_name):
        """ Generate an "inner" IDF object for a given material. """
        result = \
            self.idf_obj.newidfobject(
                "CONSTRUCTION",
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
        self.schedules = result

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
        """ Populate our model with stuff such as people and lights. """
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
