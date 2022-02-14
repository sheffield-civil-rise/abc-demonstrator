"""
This code defines a class which runs some of the more time-consuming processes.
"""

# Standard imports.
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

# Non-standard imports.
import meshroom
from meshroom import multiview
from meshroom.nodes.aliceVision.CameraInit import readSfMData
from meshroom.core.graph import Graph
from meshroom.core.taskManager import TaskManager

# Local imports.
import config
from custom_pipeline import build_graph

##############
# MAIN CLASS #
##############

@dataclass
class BatchProcessor:
    """ The class in question. """
    # Fields.
    search_recursively: bool = True
    path_to_output_images: str = None
    pipeline: str = "custom"
    node_output: list = field(default_factory=list) # TODO: Ask about renaming this.
    path_to_cache: str = os.path.join(config.DEFAULT_PATH_TO_OUTPUT, "cache")
    paths_to_structure_from_motion_files: list = field(default_factory=list)
    path_to_labelled_images: str = None
    # Generated fields.
    has_searched_for_images: bool = False
    flag_two_way: bool = False

    # Class attributes.
    SWITCH_NODE: ClassVar[dict] = {
        "photogrammetry": multiview.photogrammetry,
        "panoramahdr": multiview.panoramaHdr,
        "panoramafisheyehdr": multiview.panoramaFisheyeHdr,
        "custom": build_graph
    }

    def __post_init__(self):
        meshroom.setupEnvironment()
        self.auto_initialise_path_to_cache()

    if args.cameraInit_mask is not None:
        if args.cameraInit is not None:
            print('running twoway pipeline')
            run(
                input, 'custom',
                cache=os.path.abspath(cache),
                init=[args.cameraInit_mask, args.cameraInit],
                label_dir=args.labels)
        else:
            print('running only masked cameraInit')
            run(
                input, 'custom',
                cache=os.path.abspath(cache),
                init=args.cameraInit_mask,
                label_dir=args.labels)
    else:
        print('running single pipeline')
        run(
            input, 'custom',
            cache=os.path.abspath(cache),
            init=args.cameraInit,
            label_dir=args.labels)

    def auto_initialise_path_to_cache(self):
        """ Set this field, if it hasn't been set already. """
        if not self.path_to_cache:
            cache_id = datetime.now().strftime("%Y%m%d%H%M%S")
            self.path_to_cache = os.path.join("cache", cache_id)
        if not os.path.exists(self.path_to_cache):
            os.makedirs(self.path_to_cache)

    def run(self):
        """ Run the batch processes in question. """
        views, intrinsics = [], []  # todo: autocalculate
        files_by_type = multiview.FilesByType()
        if len(self.paths_to_structure_from_motion_files) >= 2:
            views_0, intrinsics_0 = check_and_read_sfm(self.paths_to_structure_from_motion_files[0])
            views_1, intrinsics_1 = check_and_read_sfm(self.paths_to_structure_from_motion_files[1])
        elif len(self.paths_to_structure_from_motion_files) == 1:
            views, intrinsics = check_and_read_sfm(self.paths_to_structure_from_motion_files[0])
        else:
            print(search_recursively)
            files_by_type.extend(
                multiview.findFilesByTypeInFolder(
                    input, recursive=search_recursively
                )
            )
            if not files_by_type.images:
                print("Input unable to complete.")
                sys.exit(1)
        graph = Graph(name=pipeline)
        with multiview.GraphModification(graph):
            try:
                switchNode[pipeline.lower()](
                    inputViewpoints=views,
                    inputIntrinsics=intrinsics,
                    output=self.node_output,
                    graph=graph,
                    init=self.paths_to_structure_from_motion_files,
                    label_dir=self.path_to_labelled_images
                )
            except KeyError:
                graph.load(pipeline)
            if self.flag_two_way:
                camera_inits = graph.nodesOfType("CameraInit")
                camera_inits[0].viewpoints.resetValue()
                camera_inits[0].viewpoints.extend(views_0)
                camera_inits[0].intrinsics.resetValue()
                camera_inits[0].intrinsics.extend(intrinsics_0)
                camera_inits[1].viewpoints.resetValue()
                camera_inits[1].viewpoints.extend(views_1)
                camera_inits[1].intrinsics.resetValue()
                camera_inits[1].intrinsics.extend(intrinsics_1)
            else:
                camera_init = getOnlyNodeOfType(graph, "CameraInit")
                camera_init.viewpoints.resetValue()
                camera_init.viewpoints.extend(views)
                camera_init.intrinsics.resetValue()
                camera_init.intrinsics.extend(intrinsics)
            if not graph.canComputeLeaves:
                raise BatchProcessorError(
                    "Graph cannot be computed. Check compatibility"
                )
            if self.node_output:
                publish = get_only_node_of_type(graph, "Publish")
                publish.output.value = self.node_output
            if files_by_type.images:
                views, intrinsics = \
                    camera_init.nodeDesc.buildIntrinsics(
                        cameraInit, files_by_type.images
                    )
            if self.paths_to_structure_from_motion_files:
                camera_init.viewpoints.value = views
                camera_init.intrinsics.value = intrinsics
            graph.cacheDir = cache if cache else meshroom.core.defaultCacheFolder
        to_nodes = None
        task_manager = TaskManager()
        task_manager.compute(graph, toNodes=to_nodes)
        return (lambda: taskManager._thread.isRunning())

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class BatchProcessorError(Exception):
    """ A custom exception. """
    pass

def get_only_node_of_type(graph, node_type):
    """ Helper function to get a node of WnodeTypeW in the graph and raise an
    exception if there are none or multiple candidates. """
    nodes = graph.nodesOfType(node_type)
    if len(nodes) != 1:
        raise BatchProcessorError(
            "Required a pipeline graph with exactly one '{}' node. "+
            "Found {} such nodes."
            .format(node_type, len(nodes))
        )
    return nodes[0]

def is_structure_from_motion_file(path_to):
    """ Detect whether a given file is the StructureFromMotion file. """
    if not path_to or (len(path_to) < 1) or not os.path.isfile(path_to):
        return False
    if os.path.splitext(path_to)[-1] in (".json", ".sfm"):
        return True
    return False

def check_and_read_sfm(path_to):
    """ Raise an exception if the input is not a structure from motion file;
    otherwise read it. """
    if not is_structure_from_motion_file(path_to):
        raise BatchProcessorError(
            "File at "+path_to+" is not a structure from motion file."
        )
    return readSfMData(path_to)
