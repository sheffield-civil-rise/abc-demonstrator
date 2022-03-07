"""
This code defines a class, "Amy's IDF", which holds the anticipated changes to
the EnergyModelGenerator class.
"""

# Local imports.
from energy_model_generator import EnergyModelGenerator

##############
# MAIN CLASS #
##############

class AddNewPreciseWindow(EnergyModelGenerator):
    """ The class in question. """
    # Class attributes.
    AUTOCALCULATE_KEY_STRING = "autocalculate"
    GLOBAL_GEOMETRY_RULES_KEY_STRING = "GLOBALGEOMETRYRULES"
    OUTSIDE_KEY_STRING = "outdoors"
    WALL_KEY_STRING = "wall"
    WINDOW_OBJECT_KEY_STRING = "FENESTRATIONSURFACE:DETAILED"

    def add_windows(self, coords):
        # Storey as string object.
        try:
            global_geometry_rules = \
                self.idf_obj.idfobjects[
                    self.GLOBAL_GEOMETRY_RULES_KEY_STRING
                ][0]  # Type: Optional[Idf_MSequence]
        except IndexError:
            global_geometry_rules = None
        # Find external walls.
        external_walls = \
            list(filter(
                lambda x: (
                    x.Outside_Boundary_Condition.lower() == 
                        self.OUTSIDE_KEY_STRING,
                )
                self.idf_obj.getsurfaces(self.WALL_KEY_STRING)
            ))
        for wall in external_walls:
            right_wall_index = get_right_wall_index(wall, coords)
            if right_wall_index:
                window_name = wall.Name+"_window"
                window = \
                    self.idf_obj.newidfobject(
                        self.WINDOW_OBJECT_KEY_STRING,
                        Name=window_name,
                        Surface_Type=self.WINDOW_IDF_OBJECT_NAME,
                        Building_Surface_Name=wall.Name,
                        View_Factor_to_Ground=self.AUTOCALCULATE_KEY_STRING
                        # ^ From surface angle. ^
                    )
                window.setcoords(coords, ggr)

def get_right_wall_index(wall, coords):
    result =
        (
            (
                (
                    wall.Vertex_1_Xcoordinate == wall.Vertex_2_Xcoordinate ==
                    wall.Vertex_3_Xcoordinate == wall.Vertex_4_Xcoordinate ==
                    coords[0][0] == coords[1][0] == coords[2][0] ==c oords[3][0]
                ) and (
                    max(
                        wall.Vertex_1_Ycoordinate, wall.Vertex_2_Ycoordinate,
                        wall.Vertex_3_Ycoordinate, wall.Vertex_4_Ycoordinate
                    ) >= max(
                        coords[0][1], coords[1][1], coords[2][1], coords[3][1]
                    ) >= min(
                        coords[0][1], coords[1][1], coords[2][1], coords[3][1]
                    ) >= min(
                        wall.Vertex_1_Ycoordinate, wall.Vertex_2_Ycoordinate,
                        wall.Vertex_3_Ycoordinate, wall.Vertex_4_Ycoordinate
                    )
                ) and (
                    max(
                        wall.Vertex_1_Zcoordinate, wall.Vertex_2_Zcoordinate,
                        wall.Vertex_3_Zcoordinate, wall.Vertex_4_Zcoordinate
                    ) >= max(
                        coords[0][2], coords[1][2], coords[2][2], coords[3][2]
                    ) >= min(
                        coords[0][2], coords[1][2], coords[2][2], coords[3][2]
                    ) >= min(
                        wall.Vertex_1_Zcoordinate, wall.Vertex_2_Zcoordinate,
                        wall.Vertex_3_Zcoordinate, wall.Vertex_4_Zcoordinate
                    )
                )
            ) or (
                (
                    wall.Vertex_1_Ycoordinate == wall.Vertex_2_Ycoordinate ==
                    wall.Vertex_3_Ycoordinate == wall.Vertex_4_Ycoordinate ==
                    coords[0][1] == coords[1][1] == coords[2][1] == coords[3][1]
                ) and (
                    max(
                        wall.Vertex_1_Xcoordinate, wall.Vertex_2_Xcoordinate,
                        wall.Vertex_3_Xcoordinate, wall.Vertex_4_Xcoordinate
                    ) >= max(
                        coords[0][0], coords[1][0], coords[2][0], coords[3][0]
                    ) >= min(
                        coords[0][0], coords[1][0], coords[2][0], coords[3][0]
                    ) >= min(
                        wall.Vertex_1_Xcoordinate, wall.Vertex_2_Xcoordinate,
                        wall.Vertex_3_Xcoordinate, wall.Vertex_4_Xcoordinate
                    )
                ) and (
                    max(
                        wall.Vertex_1_Zcoordinate, wall.Vertex_2_Zcoordinate,
                        wall.Vertex_3_Zcoordinate, wall.Vertex_4_Zcoordinate
                    ) >= max(
                        coords[0][2], coords[1][2], coords[2][2], coords[3][2]
                    ) >= min(
                        coords[0][2], coords[1][2], coords[2][2], coords[3][2]
                    ) >= min(
                        wall.Vertex_1_Zcoordinate, wall.Vertex_2_Zcoordinate,
                        wall.Vertex_3_Zcoordinate, wall.Vertex_4_Zcoordinate
                    )
                )
            )
        )
    return result
