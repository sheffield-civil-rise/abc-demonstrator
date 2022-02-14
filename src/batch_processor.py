"""
This code defines a class which runs some of the more time-consuming processes.
"""

# Standard imports.
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, ClassVar

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
    paths_to_init_files: list = field(default_factory=list) # I.e. SFM files.
    path_to_labelled_images: str = None
    # Generated fields.
    has_searched_for_images: bool = False
    two_way: bool = False
    single_track: bool = False
    views: list = field(default_factory=list)
    intrinsics: list = field(default_factory=list)
    files_by_type: list = None
    graph: Graph = None
    function_to_run: Callable = None

    # Class attributes.
    SWITCH_NODE: ClassVar[dict] = {
        "photogrammetry": multiview.photogrammetry,
        "panoramahdr": multiview.panoramaHdr,
        "panoramafisheyehdr": multiview.panoramaFisheyeHdr,
        "custom": build_graph
    }
    INIT_NODE_TYPE: ClassVar[str] = "CameraInit"

    def __post_init__(self):
        meshroom.setupEnvironment()
        self.auto_initialise_path_to_cache()
        self.discern_pipeline()
        self.files_by_type = multiview.FilesByType()
        self.get_views_and_instrinsics()
        self.make_graph()
        self.make_function_to_run()

    def auto_initialise_path_to_cache(self):
        """ Set this field, if it hasn't been set already. """
        if not self.path_to_cache:
            cache_id = datetime.now().strftime("%Y%m%d%H%M%S")
            self.path_to_cache = os.path.join("cache", cache_id)
        if not os.path.exists(self.path_to_cache):
            os.makedirs(self.path_to_cache)

    def discern_pipeline(self):
        """ Find out what kind of pipleline we're running. """
        if len(self.paths_to_init_files) == 2:
            print("Running two-way pipeline...")
            self.two_way = True
        elif len(self.paths_to_init_files) == 1:
            print("Running only masked cameraInit...") # TODO: Ask about rephrasing this.
        else:
            print("Running single pipeline...")
            self.single_track = True

    def get_views_and_instrinsics(self):
        """ Fill in these fields. """
        if self.two_way:
            self.views = [None, None]
            self.intrinsics = [None, None]
            self.views[0], self.intrinsics[0] = \
                check_and_read_sfm(self.paths_to_init_files[0])
            self.views[1], self.intrinsics[1] = \
                check_and_read_sfm(self.paths_to_init_files[1])
        elif self.single_track:
            self.files_by_type.extend(
                multiview.findFilesByTypeInFolder(
                    self.path_to_output_images,
                    recursive=self.search_recursively
                )
            )
            if not self.files_by_type.images:
                print("Input unable to complete.")
                sys.exit(1)
        else:
            self.views, self.intrinsics = \
                check_and_read_sfm(self.paths_to_init_files[0])

    def make_graph(self):
        """ Make our graph field. """
        self.graph = Graph(name=self.pipeline)
        with multiview.GraphModification(self.graph):
            try:
                self.SWITCH_NODE[self.pipeline.lower()](
                    inputViewpoints=self.views,
                    inputIntrinsics=self.intrinsics,
                    output=self.node_output,
                    graph=self.graph,
                    init=self.paths_to_init_files,
                    label_dir=self.path_to_labelled_images
                )
            except KeyError:
                self.graph.load(self.pipeline)
            if self.two_way:
                print(len(self.views))
                camera_inits = self.graph.nodesOfType(self.INIT_NODE_TYPE)
                camera_inits[0].viewpoints.resetValue()
                camera_inits[0].viewpoints.extend(self.views[0])
                camera_inits[0].intrinsics.resetValue()
                camera_inits[0].intrinsics.extend(self.intrinsics[0])
                camera_inits[1].viewpoints.resetValue()
                camera_inits[1].viewpoints.extend(self.views[1])
                camera_inits[1].intrinsics.resetValue()
                camera_inits[1].intrinsics.extend(self.intrinsics[1])
            else:
                camera_init = getOnlyNodeOfType(self.graph, self.INIT_NODE_TYPE)
                camera_init.viewpoints.resetValue()
                camera_init.viewpoints.extend(self.views)
                camera_init.intrinsics.resetValue()
                camera_init.intrinsics.extend(self.intrinsics)
            if not self.graph.canComputeLeaves:
                raise BatchProcessorError(
                    "Graph cannot be computed. Check compatibility."
                )
            if self.node_output:
                publish = get_only_node_of_type(graph, "Publish")
                publish.output.value = self.node_output
            if self.files_by_type.images:
                self.views, self.intrinsics = \
                    camera_init.nodeDesc.buildIntrinsics(
                        cameraInit, self.files_by_type.images
                    )
            self.graph.cacheDir = (
                self.path_to_cache
                    if self.path_to_cache
                    else meshroom.core.defaultCacheFolder
            )

    def make_function_to_run(self):
        """ Return the batch processing function. """
        task_manager = TaskManager()
        task_manager.compute(self.graph, toNodes=None)
        self.function_to_run = (lambda: taskManager._thread.isRunning())

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
