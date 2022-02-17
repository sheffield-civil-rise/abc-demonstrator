# information about this specific python package can be find in https://geomeppy.readthedocs.io/en/latest/?
from geomeppy import IDF

import os
import argparse
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon

class config:
    """ this is a dummy config class to be pulled out later """
    LOCATION_IDD = r'C:\EnergyPlusV9-5-0\Energy+.idd'
    LOCATION_EPW = r'C:\EnergyPlusV9-5-0\WeatherData\GBR_Finningley.033600_IWEC.epw'

def generate_initial_idf(init, epw):
    """ """

    IDF.setiddname(config.LOCATION_IDD)  # location of Energy+.idd file

    idf = IDF(init)  # location of the starting_point.idf

    idf.epw = epw  # Give the location of weather epw file

    return idf


def define_geometry(idf, id, polygon, height, nbfloors=2):
    """ add initial geometry """
    # NOTE:the vertex for coordination can be given clockwise or anti-clockwise
    # NOTE:the floor plan can be a non-convex surface
    idf.add_block(
        name = id,
        coordinates = polygon,
        height = height,  # total height of the building above ground level
        num_stories = nbfloors)

    # idf.translate_to_origin()  # might be handy for real world coordination
    # it's possible to have more than 1 block
    #   intersect match can deal with this

    # surface intersect matching to ensure correct boundary conditions
    idf.intersect_match()


def set_window_wall_ratio(idf, wwr=None):
    """ set the WWR for the building """
    if wwr is None:
        #      N        E         S         W
        wwr = {0: 0.1, 90: 0.3, 180: 0.5, 270: 0.7}
    idf.set_wwr(wwr_map=wwr)


def rotate_building(idf, orientation):
    """ Rotate the building to the real orientation onsite """
    idf.idfobjects["BUILDING"][0].North_Axis = orientation


def define_materials(idf, uvalues=None, window_shgc=None, roughness=None):
    """ Define materials, currently defuel all roughness to "MediumRough",
    this can be change if needed
    """
    if roughness is None:
        roughness = "MediumRough"

    if uvalues is None:
        uvalues = {
            "wall": 2.0,
            "roof": 2.0,
            "floor": 2.0,
            "ceiling": 2.0,
            "window": 2.5}

    if window_shgc is None:
        window_shgc = 0.5

    def generate(name, uvalue):
        """ sub method to generate material idf """
        if name == 'window':
            return None  #  create special one for windows
        return idf.newidfobject("MATERIAL:NOMASS",
            Name = name,
            Roughness = roughness,
            Thermal_Resistance = 1.0/uvalue)

    materials = {
        k.capitalize(): generate(k, v) for k,v in uvalues.items()}

    materials['Window'] = define_window_materials(
        idf, uvalues['window'], window_shgc)

    return materials


def define_window_materials(idf, uvalue=None, shgc=None):
    """ define window materials """

    if uvalue is None:
        uvalue = 2.5
    if shgc is None:
        shgc = 0.5

    material = idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
        Name = 'window',
        UFactor = uvalue,
        Solar_Heat_Gain_Coefficient = shgc)
    return material


def define_construction(idf, layers=None):
    """ Define constructions """

    def generate(name):
        """ sub method to generate construction idf """
        return idf.newidfobject("CONSTRUCTION",
            Name = "%s_construction" % name,
            Outside_Layer = name)

    if layers is None:
        layers = ["wall", "roof", "floor", "ceiling", "window"]
    elif type(layers) is not list:
        layers = [layers]

    constructions = {name.capitalize(): generate(name) for name in layers}

    for name in layers:
        # assign construction to building surface and subsurface
        if name is 'window':
            for layer in idf.getsubsurfaces(name):
                layer.Construction_Name = constructions[name.capitalize()].Name
        else:
            for layer in idf.getsurfaces(name):
                layer.Construction_Name = constructions[name.capitalize()].Name

    return constructions


def default_schedules():
    #  Detail schedule definitions
    #  [refers to the energyplus I/O document for further information]
    return {
        "People": 'Through: 12/31, For: Weekdays, Until: 24:00,1,For: AllOtherDays,Until: 24:00,1;',
        "Activity": 'Through: 12/31, For: Weekdays, Until: 24:00,100,For: AllOtherDays,Until: 24:00,100;',
        "Lighting": 'Through: 12/31, For: Weekdays, Until: 24:00,1,For: AllOtherDays,Until: 24:00,1;',
        "Equipment": 'Through: 12/31, For: Weekdays, Until: 24:00,1,For: AllOtherDays,Until: 24:00,1;',
        "Infiltration": 'Through: 12/31, For: Weekdays, Until: 24:00,1,For: AllOtherDays,Until: 24:00,1;',
        "Availability": 'Through: 12/31, For: Weekdays, Until: 24:00,1,For: AllOtherDays,Until: 24:00,1;'}


def define_schedules(idf, schedules=None):
    """ schedules definition """

    if schedules is None:  # use all defaults
        schedules = default_schedules()
    else:  # fill in with defaults if any missing
        for k,v in default_schedules.items():
            if k not in schedules:
                schedules[k] = v

    idf_schedules = {}  # output idf objects

    idf_schedules["People"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name = "People schedule",
        Schedule_Type_Limits_Name = "Fraction",
        Field_1 = schedules["People"])

    idf_schedules["Activity"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name = "Activity level schedule",
        Schedule_Type_Limits_Name = "Activity",
        Field_1 = schedules["Activity"])

    idf_schedules["Lighting"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name = "Lighting schedule",
        Schedule_Type_Limits_Name = "Fraction",
        Field_1 = schedules["Lighting"])

    idf_schedules["Equipment"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name="Equipment schedule",
        Schedule_Type_Limits_Name="Fraction",
        Field_1=schedules["Equipment"])

    idf_schedules["Infiltration"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name = "Infiltration schedule",
        Schedule_Type_Limits_Name = "Fraction",
        Field_1 = schedules["Infiltration"])

    idf_schedules["Availability"] = idf.newidfobject("SCHEDULE:COMPACT",
        Name = "Heating availability schedule",
        Schedule_Type_Limits_Name = "Fraction",
        Field_1 = schedules["Availability"])

    return idf_schedules


def add_thermostat(idf, setpoint_heating=None, setpoint_cooling=None):
    """ Thermostat for HVAC control """

    if setpoint_heating is None:
        setpoint_heating = 18  # degrees C
    if setpoint_cooling is None:
        setpoint_cooling = 26  # degrees C

    thermostat = idf.newidfobject("HVACTEMPLATE:THERMOSTAT",
        Name = "Zone Thermostat",
        Constant_Heating_Setpoint = setpoint_heating,
        Constant_Cooling_Setpoint = setpoint_cooling)

    return thermostat


def add_heating_hot_water_loop(idf):
    """ hot water loop to serve the space heating radiator """
    hot_water_loop = idf.newidfobject("HVACTEMPLATE:PLANT:HOTWATERLOOP",
        Name = "Space heating hot water loop")
    return hot_water_loop


def add_boiler(idf, type=None, fuel=None, efficiency=None):
    """ Boiler to serve the space heating system-natural gas condensing boiler

        inputs:
            idf         idf object
            type        type of boiler [default CondensingHotWaterBoiler]
            fuel        fuel type for boiler [default NaturalGas]
            efficiency  boiler efficency [default 0.8]
    """

    if type is None:
        type = "CondensingHotWaterBoiler"
    if fuel is None:
        fuel = "NaturalGas"
    if efficiency is None:
        efficiency = 0.8

    boiler = idf.newidfobject("HVACTEMPLATE:PLANT:BOILER",
        Name = "Boiler",
        Boiler_Type = type,
        Efficiency = efficiency,
        Fuel_Type = fuel)

    return boiler


def populate_zones(idf, idf_schedules, thermostat,
    densities = None, ach = None):
    """ populate zones """
    # Density definitions
    if densities is None:
        densities = {
            'electric': 8.0,  # Watt/m2
            'lighting': 8.0,  # Watt/m2
            'people': 0.025}  # person/m2

    if ach is None:
        ach = 0.5  # infiltration air change per hour rate

    for zone in idf.idfobjects["ZONE"]:
        # add radiator in every zone
        idf.newidfobject("HVACTEMPLATE:ZONE:BASEBOARDHEAT",
            Zone_Name = zone.Name,
            Template_Thermostat_Name = thermostat.Name,
            Baseboard_Heating_Availability_Schedule_Name= idf_schedules['Availability'].Name,
            Outdoor_Air_Method = "Flow/Person",
            Outdoor_Air_Flow_Rate_per_Person = 0)  # m3/s
            # no additional outdoor air supply apart from the infiltration

        # define infiltration in every zone
        idf.newidfobject("ZONEINFILTRATION:DESIGNFLOWRATE",
            Name='Infiltration: ' + zone.Name,
            Zone_or_ZoneList_Name = zone.Name,
            Schedule_Name = idf_schedules['Infiltration'].Name,
            Design_Flow_Rate_Calculation_Method = "AirChanges/Hour",
            Air_Changes_per_Hour = ach)

        # define occupants in every zone
        idf.newidfobject("PEOPLE",
            Name = 'People: ' + zone.Name,
            Zone_or_ZoneList_Name = zone.Name,
            Number_of_People_Schedule_Name = idf_schedules['People'].Name,
            Number_of_People_Calculation_Method = "People/Area",
            People_per_Zone_Floor_Area = densities['people'],
            Activity_Level_Schedule_Name = idf_schedules['Activity'].Name)

        # define lights in every zone
        idf.newidfobject("LIGHTS",
            Name='Light: ' + zone.Name,
            Zone_or_ZoneList_Name=zone.Name,
            Schedule_Name = idf_schedules['Lighting'].Name,
            Design_Level_Calculation_Method = "Watts/Area",
            Watts_per_Zone_Floor_Area=densities['lighting'])

        # define equipment in every zone
        idf.newidfobject("ELECTRICEQUIPMENT",
            Name = 'Electricequipment: ' + zone.Name,
            Zone_or_ZoneList_Name = zone.Name,
            Schedule_Name = idf_schedules['Equipment'].Name,
            Design_Level_Calculation_Method = "Watts/Area",
            Watts_per_Zone_Floor_Area=densities['electric'])


def save_to_file(idf, output):
    """ Write out idf file in a user defined location for manual checking if needed """
    idf.save(output)


def run_simulation(idf, id, output):
    """ Run the energyplus simulation and save the simulation output to a predefined direction """

    if not os.path.isdir(output):
        os.makedirs(output)

    buildingID = 'test box'  # to do: bring this out
    idf.run(
        output_directory = output,
        expandobjects = True,
        output_prefix = id,
        output_suffix = 'C')


def autogenerate(args):
    """ autogenerate idf from input arguments """
    args = verify_args(args)

    idf = generate_initial_idf(args.init, args.epw)

    define_geometry(idf, args.id, args.polygon, args.height, args.nbfloors)

    set_window_wall_ratio(idf, args.wwr)

    rotate_building(idf, args.orientation)

    materials = define_materials(idf, args.uvalues, args.shgc, args.roughness)

    constructions = define_construction(idf, args.layers)

    schedules = define_schedules(idf, args.schedules)

    thermostat = add_thermostat(idf, *args.setpoint)

    hot_water_loop = add_heating_hot_water_loop(idf)

    boiler = add_boiler(idf,
        args.boiler['type'], args.boiler['fuel'], args.boiler['efficiency'])

    populate_zones(idf, schedules, thermostat, args.densities, args.ach)

    return idf


def read_polygon(polygon=None):
    if polygon is None:
        return [(6, 0), (6, 6), (0, 6), (0, 0)]
    else:
        df = pd.read_csv(polygon, header=None)
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(df[1], df[0]),
            crs='epsg:27700')
        bounds = Polygon(gdf.geometry).bounds
        poly = gdf.geometry.translate(-bounds[0], -bounds[1])
        return list(zip(poly.x, poly.y))


def generate_argparser():
    ''' generate argparser for command line usage '''
    parser = argparse.ArgumentParser()

    parser.add_argument('id', type=str, help='building id')
    parser.add_argument('init', type=str,
        help='location of initial idf file')
    parser.add_argument('--height', type=float,
        help = 'building height')
    parser.add_argument('--nbfloors', type=int,
        help = 'number of floors', default=2)
    parser.add_argument('--wwr', nargs=1, type=float,
        help = 'window to wall ratio (single value)')
    parser.add_argument('--wwrs', nargs=4, type=float,
        help='window to wall ratios of each facade')
    parser.add_argument('--epw', type=str,
        help='location of epw weather file')
    parser.add_argument('--orientation', type=float,
        help = 'orientation of shape')
    parser.add_argument('-p', '--polygon', type=str,
        help = 'location of polygon coordinates (csv)')
    parser.add_argument('-o', '--output', type=str, nargs='?',
        help = 'location to write output',
        default = '%ID%_autogenerated.idf')
    parser.add_argument('--nosave', action='store_true',
        help = 'print idf to console output only')
    parser.add_argument('-d', '--outdir', type=str,
        help = 'simulate energy model and save here')

    return parser


def verify_args(args):
    ''' verify and validate arguments and set defaults if invalid '''


    if args is None:
        class Arguments:  # dummy class
            pass
        args = Arguments()

    if not hasattr(args, 'id') or args.id is None:
        args.id = 'test_building'
    elif type(args.id) is not str:
        args.id = str(id)

    if not hasattr(args, 'height') or args.height is None:
        args.height = 6.0

    if not hasattr(args, 'nbfloors') or args.nbfloors is None:
        args.nbfloors = 2

    if not hasattr(args, 'polygon') or args.polygon is None:
        args.polygon = [(6, 0), (6, 6), (0, 6), (0, 0)]
        #[(10, 0), (10, 0), (0, 10), (0, 0)]
    elif type(args.polygon) is str:
        args.polygon = read_polygon(args.polygon)

    if not hasattr(args, 'init') or args.init is None:
        args.init = 'starting_point.idf'

    if not hasattr(args, 'epw') or args.epw is None:
        args.epw = config.LOCATION_EPW

    if not hasattr(args, 'wwr') or args.wwr is None:
        args.wwr = None # use default
    elif type(args.wwr) is list:
        if len(args.wwr) == 1:
            args.wwr = {0: args.wwr[0], 90: 0.2, 180: args.wwr[0], 270: 0.2}
        else:
            args.wwr = {k: v for k, v in zip([0, 90, 180, 270], args.wwr)}
    elif type(args.wwr) is float:
        args.wwr = {0: args.wwr, 90: 0.2, 180: args.wwr, 270: 0.2}

    if not hasattr(args, 'orientation') or args.orientation is None:
        args.orientation = 0.0

    if not hasattr(args, 'output') or args.output is None:
        args.output = '%s_autogenerated.idf' % args.id
    elif args.output == '%ID%_autogenerated.idf':
        args.output = '%s_autogenerated.idf' % args.id

    if not hasattr(args, 'outdir') or args.outdir is None:
        args.simulate = False
    else:
        args.simulate = True

    if not hasattr(args, 'nosave') or args.nosave is None:
        args.nosave = False


    ### defaults (set as None do defer to default)
    args.layers = None

    args.uvalues = None
    args.shgc = None
    args.roughness = None

    args.schedules = None

    args.setpoint = [None, None]  # heating, cooling

    args.boiler = {
        'type': None, 'fuel': None, 'efficiency': None}

    args.densities = None
    args.ach = None

    return args


def main(args):
    """ """
    args = verify_args(args)

    idf = autogenerate(args)

    if not args.nosave:
        save_to_file(idf, args.output)
    else:
        print(idf)

    if args.simulate:
        run_simulation(idf, args.id, args.outdir)


if __name__ == '__main__':
    parser = generate_argparser()
    args = parser.parse_args()

    main(args)
