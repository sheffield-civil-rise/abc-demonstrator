#!/usr/bin/env python

from meshroom import multiview
from meshroom.nodes.aliceVision.CameraInit import readSfMData
from meshroom.core.graph import Graph
from meshroom.core.taskManager import TaskManager

from custom_pipeline import build_graph

import argparse

import os
import sys
import datetime

import meshroom
meshroom.setupEnvironment()

switchNode = {
    'photogrammetry':      multiview.photogrammetry,
    'panoramahdr':        multiview.panoramaHdr,
    'panoramafisheyehdr': multiview.panoramaFisheyeHdr,
    # 'cameratracking':     multiview.cameraTracking,
    'custom':             build_graph
}  # maps name to initial function

search_recursively = False


def getOnlyNodeOfType(g, nodeType):
    """ Helper function to get a node of 'nodeType' in the graph 'g' and raise
        if no or multiple candidates. """
    nodes = g.nodesOfType(nodeType)
    if len(nodes) != 1:
        raise RuntimeError(
            "require a pipeline graph with exactly one '{}' node, {} found."
            .format(nodeType, len(nodes)))
    return nodes[0]


def issfm(name):
    """ Detect if file is StructureFromMotion file """
    if not name or len(name) < 1 or not os.path.isfile(name):
        return False
    if os.path.splitext(name)[-1] in ('.json', '.sfm'):
        return True
    else:
        return False


def run(input, pipeline, output=[], cache=[], save=[], init=None, label_dir=None):
    ''' Execute '''

    views, intrinsics = [], []  # todo: autocalculate
    filesByType = multiview.FilesByType()

    hasSearchedForImages = False

    flagTwoWay = False
    if init is not None:
        if type(init) is list and len(init) > 1:
            views_0, intrinsics_0 = readSfMData(init[0])
            views_1, intrinsics_1 = readSfMData(init[1])
            flagTwoWay = True
        elif issfm(init):
            views, intrinsics = readSfMData(init)
        else:
            raise ValueException('what is init ? \n%s' % str(init))
    else:
        print(search_recursively)
        filesByType.extend(
            multiview.findFilesByTypeInFolder(
                input, recursive=search_recursively))
        hasSearchedForImages = True

        if hasSearchedForImages and not filesByType.images:
            print("Input unable to complete")
            sys.exit(-1)

    graph = Graph(name=pipeline)

    print("views: "+str(len(views_0))+", "+str(len(views_1)))
    print("intrinsics: "+str(len(intrinsics_0))+", "+str(len(intrinsics_1)))
    print("output: "+str(output))
    print("graph: "+str(graph))
    print("init: "+str(init))
    print("label_dir: "+str(label_dir))
    with multiview.GraphModification(graph):
        try:
            switchNode[pipeline.lower()](
                inputViewpoints=views, inputIntrinsics=intrinsics,
                output=output, graph=graph, init=init, label_dir=label_dir)
        except KeyError:
            graph.load(pipeline)

        # if init is None:
        if flagTwoWay:
            cameraInits = graph.nodesOfType('CameraInit')
            cameraInits[0].viewpoints.resetValue()
            cameraInits[0].viewpoints.extend(views_0)
            # add intrinsics
            cameraInits[0].intrinsics.resetValue()
            cameraInits[0].intrinsics.extend(intrinsics_0)

            cameraInits[1].viewpoints.resetValue()
            cameraInits[1].viewpoints.extend(views_1)
            # add intrinsics
            cameraInits[1].intrinsics.resetValue()
            cameraInits[1].intrinsics.extend(intrinsics_1)
        else:
            cameraInit = getOnlyNodeOfType(graph, 'CameraInit')
            # add views
            cameraInit.viewpoints.resetValue()
            cameraInit.viewpoints.extend(views)
            # add intrinsics
            cameraInit.intrinsics.resetValue()
            cameraInit.intrinsics.extend(intrinsics)

        if not graph.canComputeLeaves:
            raise RuntimeError("Graph cannot be computed. Check compatibility")

        if output:
            publish = getOnlyNodeOfType(graph, 'Publish')
            publish.output.value = output

        if filesByType.images:
            views, intrinsics = cameraInit.nodeDesc.buildIntrinsics(
                cameraInit, filesByType.images)
        if init is None:
            cameraInit.viewpoints.value = views
            cameraInit.intrinsics.value = intrinsics
        #
        # print('===========')
        # print(cache) 
        # print('==========')

        graph.cacheDir = cache if cache else meshroom.core.defaultCacheFolder

        if save:
            graph.save(save, setupProjectFile=not bool(cache))
            print('file saved successfully: "{}"'.format(save))

    # This will compute all graph but could be editted to check intermidiaries
    toNodes = None

    taskManager = TaskManager()
    taskManager.compute(graph, toNodes=toNodes)

    print("GRAPH NODES: "+str(len(graph._nodes)))

    return (lambda: taskManager._thread.isRunning())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir', help='image directory')
    parser.add_argument(
        '-c', '--cameraInit', help='cameraInit file for known poses')
    parser.add_argument(
        '-l', '--cameraInit_mask', help='cameraInit file for masked images')
    parser.add_argument(
        '-d', '--labels', help='directory of labels')

    parser.add_argument(
        '--cache', help='cache for writing out')

    args = parser.parse_args()

    input = os.path.abspath(args.dir)
    search_recursively = True

    if args.cache is None:
        id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        cache = os.path.join('cache', id)

    if not os.path.exists(cache):
        os.makedirs(cache)

    # run(input, 'photogrammetry', cache=os.path.abspath(cache))

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
