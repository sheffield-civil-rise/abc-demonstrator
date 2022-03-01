"""
This code defines a class which runs some of the more time-consuming processes.
"""

# Standard imports.
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import ClassVar

# Non-standard imports.
import meshroom
from meshroom import multiview
from meshroom.nodes.aliceVision.CameraInit import readSfMData
from meshroom.core.graph import Graph
from meshroom.core.taskManager import TaskManager

# Local imports.
from config import get_configs
from custom_pipeline import build_graph

# Local constants.
CONFIGS = get_configs()

#####################
# SPECIAL FUNCTIONS #
#####################

def mute():
    """ Suppress standard output. """
    sys.stdout = open(os.devnull, "w")

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
    publisher_output: list = field(default_factory=list)
    path_to_cache: str = os.path.join(CONFIGS.general.path_to_output, "cache")
    paths_to_init_files: list = field(default_factory=list) # I.e. SFM files.
    path_to_labelled_images: str = None
    timeout: int = CONFIGS.batch_process.timeout
    # Generated fields.
    has_searched_for_images: bool = False
    two_way: bool = False
    single_track: bool = False
    views: list = field(default_factory=list)
    intrinsics: list = field(default_factory=list)
    files_by_type: multiview.FilesByType = None
    pool: Pool = None

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
        self.get_views_and_instrinsics()

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
            print("Running only masked cameraInit...")
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

    def start(self):
        """ Start the process. """
        args = [
            self.pipeline,
            self.views,
            self.intrinsics,
            self.publisher_output,
            self.paths_to_init_files,
            self.path_to_labelled_images,
            self.two_way,
            self.path_to_cache
        ]
        self.pool = Pool(initializer=mute)
        async_result = self.pool.map_async(make_graph_and_process, args)
        async_result.get(timeout=self.timeout)

################################
# HELPER CLASSES AND FUNCTIONS #
################################

class BatchProcessorError(Exception):
    """ A custom exception. """

def get_only_node_of_type(graph, node_type):
    """ Helper function to get a node of WnodeTypeW in the graph and raise an
    exception if there are none or multiple candidates. """
    nodes = graph.nodesOfType(node_type)
    if len(nodes) != 1:
        raise BatchProcessorError(
            "Required a pipeline graph with exactly one "+str(node_type)+" "+
            "node. Found "+str(len(nodes))+" such nodes."
        )
    return nodes[0]

def is_structure_from_motion_file(path_to):
    """ Detect whether a given file is the StructureFromMotion file. """
    if path_to and (len(path_to) >= 1) and os.path.isfile(path_to):
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

def make_graph(
        pipeline,
        views,
        intrinsics,
        publisher_output,
        paths_to_init_files,
        path_to_labelled_images,
        two_way,
        path_to_cache
    ):
    """ Make a graph, which the task manager will then process. """
    result = Graph(name=pipeline)
    file_by_type = multiview.FilesByType()
    with multiview.GraphModification(result):
        try:
            BatchProcessor.SWITCH_NODE[pipeline.lower()](
                input_viewpoints=views,
                input_intrinsics=intrinsics,
                output=publisher_output,
                graph=result,
                init=paths_to_init_files,
                label_dir=path_to_labelled_images
            )
        except KeyError:
            result.load(pipeline)
        if two_way:
            camera_inits = result.nodesOfType(BatchProcessor.INIT_NODE_TYPE)
            camera_inits[0].viewpoints.resetValue()
            camera_inits[0].viewpoints.extend(views[0])
            camera_inits[0].intrinsics.resetValue()
            camera_inits[0].intrinsics.extend(intrinsics[0])
            camera_inits[1].viewpoints.resetValue()
            camera_inits[1].viewpoints.extend(views[1])
            camera_inits[1].intrinsics.resetValue()
            camera_inits[1].intrinsics.extend(intrinsics[1])
        else:
            camera_init = \
                get_only_node_of_type(result, BatchProcessor.INIT_NODE_TYPE)
            camera_init.viewpoints.resetValue()
            camera_init.viewpoints.extend(views)
            camera_init.intrinsics.resetValue()
            camera_init.intrinsics.extend(intrinsics)
        if not result.canComputeLeaves:
            raise BatchProcessorError(
                "Graph cannot be computed. Check compatibility."
            )
        if publisher_output:
            publish = get_only_node_of_type(result, "Publish")
            publish.output.value = self.publisher_output
        if files_by_type.images:
            views, intrinsics = \
                camera_init.nodeDesc.buildIntrinsics(
                    camera_init, files_by_type.images
                )
        result.cacheDir = (
            path_to_cache
                if path_to_cache
                else meshroom.core.defaultCacheFolder
        )
        return result

def make_graph_and_process(
        pipeline,
        views,
        intrinsics,
        publisher_output,
        paths_to_init_files,
        path_to_labelled_images,
        two_way,
        path_to_cache
    ):
    """ Create the graph and the task manager object, start the latter, and
    keep us in a loop until it's finished. """
    graph = \
        make_graph(
            pipeline,
            views,
            intrinsics,
            publisher_output,
            paths_to_init_files,
            path_to_labelled_images,
            two_way,
            path_to_cache
        )
    task_manager = TaskManager()
    task_manager.compute(graph, toNodes=None)
    while task_manager._thread.isRunning():
        pass
