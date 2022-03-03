"""
This code defines some functions which help to create a pipeline for our
BatchProcessor object.
"""

# Non-standard imports.
from meshroom.core.graph import Graph, GraphModification

# Local imports.
from config import get_configs

# Local constants.
CONFIGS = get_configs()
SFM_TYPES = ("StructureFromMotion", "SfMAlignment", "SfMTransfer")

#############
# FUNCTIONS #
#############

def default_sfm_pipeline(graph):
    """ Instantiate SfM pipeline for photogrammetry pipeline. Based on default
    photogrammetry pipeline in meshroom. """
    camera_init = graph.addNewNode("CameraInit")
    feature_extraction = graph.addNewNode(
        "FeatureExtraction", input=camera_init.output,
        forceCpuExtraction=False)
    image_matching = graph.addNewNode(
        "ImageMatching", input=feature_extraction.input,
        featuresFolders=[feature_extraction.output],
        tree=CONFIGS.batch_process.path_to_vocab_tree)
    feature_matching = graph.addNewNode(
        "FeatureMatching", input=image_matching.input,
        featuresFolders=image_matching.featuresFolders,
        imagePairsList=image_matching.output,
        describerTypes=feature_extraction.describerTypes)
    structure_from_motion = graph.addNewNode(
        "StructureFromMotion", input=feature_matching.input,
        featuresFolders=feature_matching.featuresFolders,
        matchesFolders=[feature_matching.output],
        describerTypes=feature_matching.describerTypes)
    result = [
        camera_init,
        feature_extraction,
        image_matching,
        feature_matching,
        structure_from_motion
    ]
    return result

def two_way_sfm_pipeline(graph, camera_init=None):
    """ Instantiate SfM pipeline for photogrammetry pipeline. Uses custom
    defined cameraInit file. """
    out = []
    camera_init = graph.addNewNode("CameraInit")
    out.append(camera_init)
    _camera_init = graph.addNewNode("CameraInit")
    out.append(_camera_init)
    feature_extraction = \
        graph.addNewNode(
            "FeatureExtraction",
            input=camera_init.output,
            forceCpuExtraction=False
        )
    _feature_extraction = \
        graph.addNewNode(
            "FeatureExtraction",
            input= _camera_init.output,
            forceCpuExtraction=False
        )
    out.append(feature_extraction)
    out.append(_feature_extraction)
    image_matching = \
        graph.addNewNode(
            "ImageMatching",
            input=feature_extraction.input,
            featuresFolders=[feature_extraction.output],
            tree=CONFIGS.batch_process.path_to_vocab_tree
        )
    _image_matching = \
        graph.addNewNode(
            "ImageMatching",
            input=_feature_extraction.input,
            featuresFolders=[_feature_extraction.output],
            tree=CONFIGS.batch_process.path_to_vocab_tree
        )
    out.append(image_matching)
    out.append(_image_matching)
    feature_matching = \
        graph.addNewNode(
            "FeatureMatching",
            input=image_matching.input,
            featuresFolders=image_matching.featuresFolders,
            imagePairsList=image_matching.output,
            describerTypes=feature_extraction.describerTypes,
            matchFromKnownCameraPoses=True
        )
    _feature_matching = \
        graph.addNewNode(
            "FeatureMatching",
            input=_image_matching.input,
            featuresFolders=_image_matching.featuresFolders,
            imagePairsList=_image_matching.output,
            describerTypes=_feature_extraction.describerTypes,
            matchFromKnownCameraPoses=True
        )
    out.append(feature_matching)
    out.append(_feature_matching)
    structure_from_motion = \
        graph.addNewNode(
            "StructureFromMotion", input=feature_matching.input,
            featuresFolders=feature_matching.featuresFolders,
            matchesFolders=[feature_matching.output],
            describerTypes=feature_matching.describerTypes,
            useRigConstraint=False,
            lockScenePreviouslyReconstructed=True,
            lockAllIntrinsics=True
        )
    _structure_from_motion = \
        graph.addNewNode(
            "StructureFromMotion", input=_feature_matching.input,
            featuresFolders=_feature_matching.featuresFolders,
            matchesFolders=[_feature_matching.output],
            describerTypes=_feature_matching.describerTypes,
            useRigConstraint=False,
            lockScenePreviouslyReconstructed=True,
            lockAllIntrinsics=True
        )
    out.append(structure_from_motion)
    out.append(_structure_from_motion)
    sfm_alignment = \
        graph.addNewNode(
            "SfMTransfer",
            input=structure_from_motion.output,
            reference=_structure_from_motion.output,
            method="from_viewid",
            transferPoses=True,
            transferIntrinsics=True
        )
    out.append(sfm_alignment)
    return out

def custom_sfm_pipeline(graph, camera_init=None):
    """ Instnatiate SfM pipeline for photogrammetry pipeline. Uses custom
    defined cameraInit file. """
    out = []
    _camera_init = graph.addNewNode("CameraInit")
    out.append(_camera_init)
    feature_extraction = \
        graph.addNewNode(
            "FeatureExtraction",
            input= _camera_init.output,
            forceCpuExtraction=False
        )
    out.append(feature_extraction)
    image_matching = \
        graph.addNewNode(
            "ImageMatching",
            input=feature_extraction.input,
            featuresFolders=[feature_extraction.output],
            tree=CONFIGS.batch_process.path_to_vocab_tree
        )
    out.append(image_matching)
    feature_matching = \
        graph.addNewNode(
            "FeatureMatching",
            input=image_matching.input,
            featuresFolders=image_matching.featuresFolders,
            imagePairsList=image_matching.output,
            describerTypes=feature_extraction.describerTypes,
            matchFromKnownCameraPoses=True
        )
    out.append(feature_matching)
    node_input = \
        camera_init if camera_init is not None else feature_matching.input
    structure_from_motion = \
        graph.addNewNode(
            "StructureFromMotion",
            input=node_input,
            featuresFolders=feature_matching.featuresFolders,
            matchesFolders=[feature_matching.output],
            describerTypes=feature_matching.describerTypes,
            useRigConstraint=False,
            lockScenePreviouslyReconstructed=True,
            lockAllIntrinsics=True
        )
    out.append(structure_from_motion)
    return out

def sfm_pipeline(graph, init=None):
    """ Instantiate custom pipeline graph for photogrammetry. Currently uses
    default. """
    if isinstance(init, list) and (len(init) > 1):
        return two_way_sfm_pipeline(graph, init)
    return custom_sfm_pipeline(graph, init)

def default_mvs_pipeline(graph, sfm=None):
    """ Instantiate SfM pipeline for photogrammetry pipeline. Based on default
    photogrammetry pipeline in meshroom. """
    if sfm and not sfm.nodeType in SFM_TYPES:
        raise ValueError(
            "Invalid node type. Expected SfM; got "+str(sfm.nodeType)+"."
        )
    prepare_dense_scene = \
        graph.addNewNode(
            "PrepareDenseScene", input=sfm.output if sfm else ""
        )
    depth_map = \
        graph.addNewNode(
            "DepthMap",
            input=prepare_dense_scene.input,
            imagesFolder=prepare_dense_scene.output
        )
    depth_map_filter = \
        graph.addNewNode(
            "DepthMapFilter",
            input=depth_map.input,
            depthMapsFolder=depth_map.output
        )
    meshing = \
        graph.addNewNode(
            "Meshing",
            input=depth_map_filter.input,
            depthMapsFolder=depth_map_filter.output
        )
    mesh_filtering = \
        graph.addNewNode(
            "MeshFiltering", inputMesh=meshing.outputMesh
        )
    texturing = \
        graph.addNewNode(
            "Texturing",
            input=meshing.output,
            imagesFolder=depth_map.imagesFolder,
            inputMesh=mesh_filtering.outputMesh
        )
    result = [
        prepare_dense_scene,
        depth_map,
        depth_map_filter,
        meshing,
        mesh_filtering,
        texturing
    ]
    return result

def custom_mvs_pipeline(graph, sfm=None, label_dir=None):
    """ Instantiate SfM pipeline for photogrammetry pipeline. Based on default
    photogrammetry pipeline in meshroom. """
    if sfm and not sfm.nodeType in [
        "StructureFromMotion", "SfMAlignment", "SfMTransfer"]:
        raise ValueError(
            "Invalid node type. Expected SfM, got "+str(sfm.nodeType)+"."
        )
    prepare_dense_scene = \
        graph.addNewNode(
            "PrepareDenseScene",
            input=sfm.output if sfm else "",
            outputFileType="png"
        )
    depth_map = \
        graph.addNewNode(
            "DepthMap",
            input=prepare_dense_scene.input,
            imagesFolder=prepare_dense_scene.output
        )
    depth_map_filter = \
        graph.addNewNode(
            "DepthMapFilter",
            input=depth_map.input,
            depthMapsFolder=depth_map.output
        )
    meshing = \
        graph.addNewNode(
            "Meshing",
            input=depth_map_filter.input,
            depthMapsFolder=depth_map_filter.output)
    mesh_filtering = \
        graph.addNewNode(
            "MeshFiltering", inputMesh=meshing.outputMesh
        )
    texturing = \
        graph.addNewNode(
            "Texturing",
            input=meshing.output,
            imagesFolder=depth_map.imagesFolder,
            downscale=2,
            inputMesh=mesh_filtering.outputMesh
        )
    labelling = \
        graph.addNewNode(
            "Texturing",
            input=meshing.output,
            imagesFolder=label_dir,
            downscale=4,
            inputMesh=mesh_filtering.outputMesh
        )
    result = [
        prepare_dense_scene,
        depth_map,
        depth_map_filter,
        meshing,
        mesh_filtering,
        texturing,
        labelling
    ]
    return result

def mvs_pipeline(graph, sfm=None, label_dir=None):
    """ Instantiate custom pipeline graph for photogrammetry. Currently uses
    default. """
    if not label_dir:
        return default_mvs_pipeline(graph, sfm)
    return custom_mvs_pipeline(graph, sfm, label_dir)

def build_graph_old(
        input_images=[], input_viewpoints=[],
        input_intrinsics=[], output="",
        graph=None, init=None, label_dir=None):
    """ Custom photogrammetry graph """
    if graph is None:
        graph = Graph("Custom photogrammetry")
    with GraphModification(graph):
        if init is None:
            sfm_nodes = sfm_pipeline(graph)
            camera_init = sfm_nodes[0]
            camera_init.viewpoints.extend([{"path": img} for img in input_images])
            camera_init.viewpoints.extend(input_viewpoints)
            camera_init.intrinsics.extend(input_intrinsics)
        else:
            sfm_nodes = sfm_pipeline(graph, init)
            if type(init) is list and len(init) > 1:
                pass
            else:
                camera_init = sfm_nodes[0]
                camera_init.viewpoints.extend([{"path": img} for img in input_images])
                camera_init.viewpoints.extend(input_viewpoints)
                camera_init.intrinsics.extend(input_intrinsics)

        mvs_nodes = mvs_pipeline(graph, sfm_nodes[-1], label_dir)
        if output:
            texturing = mvs_nodes[-1]
            graph.addNewNode(
                "Publish", output=output,
                inputFiles=[
                    texturing.outputMesh,
                    texturing.outputMaterial,
                    texturing.outputTextures])
    return graph

def build_graph(
        input_images=None, input_viewpoints=None,
        input_intrinsics=None, output="",
        graph=None, init=None, label_dir=None):
    """ Custom photogrammetry graph """
    if input_images is None:
        input_images = []
    if input_viewpoints is None:
        input_viewpoints = []
    if input_intrinsics is None:
        input_intrinsics = []
    if graph is None:
        graph = Graph("Custom photogrammetry")
    with GraphModification(graph):
        if init is None:
            sfm_nodes = sfm_pipeline(graph)
            camera_init = sfm_nodes[0]
            camera_init.viewpoints.extend(
                [{"path": img} for img in input_images]
            )
            camera_init.viewpoints.extend(input_viewpoints)
            camera_init.intrinsics.extend(input_intrinsics)
        else:
            sfm_nodes = sfm_pipeline(graph, init)
            if type(init) is list and len(init) > 1:
                pass
            else:
                camera_init = sfm_nodes[0]
                camera_init.viewpoints.extend(
                    [{"path": img} for img in input_images]
                )
                camera_init.viewpoints.extend(input_viewpoints)
                camera_init.intrinsics.extend(input_intrinsics)

        mvs_nodes = mvs_pipeline(graph, sfm_nodes[-1], label_dir)
        if output:
            texturing = mvs_nodes[-1]
            graph.addNewNode(
                "Publish", output=output,
                inputFiles=[
                    texturing.outputMesh,
                    texturing.outputMaterial,
                    texturing.outputTextures])
    return graph

def initialise_list_as_necessary(to_initialise):
    """ Return an empty list if the input is None, otherwise return the
    input. """
    if to_initialise is None:
        return []
    return to_initialise
