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

#############
# FUNCTIONS #
#############

def defaultSfmPipeline(graph):
    """
    Instantiate SfM pipeline for photogrammetry pipeline.
    Based on default photogrammetry pipeline in meshroom.
    """
    cameraInit = graph.addNewNode("CameraInit")
    featureExtraction = graph.addNewNode(
        "FeatureExtraction", input=cameraInit.output,
        forceCpuExtraction=False)
    imageMatching = graph.addNewNode(
        "ImageMatching", input=featureExtraction.input,
        featuresFolders=[featureExtraction.output],
        tree=CONFIGS.batch_process.path_to_vocab_tree)
    featureMatching = graph.addNewNode(
        "FeatureMatching", input=imageMatching.input,
        featuresFolders=imageMatching.featuresFolders,
        imagePairsList=imageMatching.output,
        describerTypes=featureExtraction.describerTypes)
    structureFromMotion = graph.addNewNode(
        "StructureFromMotion", input=featureMatching.input,
        featuresFolders=featureMatching.featuresFolders,
        matchesFolders=[featureMatching.output],
        describerTypes=featureMatching.describerTypes)
    return [
        cameraInit,
        featureExtraction,
        imageMatching,
        featureMatching,
        structureFromMotion]

def twowaySfmPipeline(graph, cameraInit=None):
    """
    Instantiate SfM pipeline for photogrammetry pipeline.
    Uses custom defined cameraInit file.
    """
    out = []
    cameraInit = graph.addNewNode("CameraInit")
    out.append(cameraInit)
    _cameraInit = graph.addNewNode("CameraInit")
    out.append(_cameraInit)
    featureExtraction = graph.addNewNode(
            "FeatureExtraction",
            input=cameraInit.output,
            forceCpuExtraction=False)
    _featureExtraction = graph.addNewNode(
            "FeatureExtraction",
            input= _cameraInit.output,
            forceCpuExtraction=False)
    out.append(featureExtraction)
    out.append(_featureExtraction)

    imageMatching = graph.addNewNode(
        "ImageMatching",
        input=featureExtraction.input,
        featuresFolders=[featureExtraction.output],
        tree=CONFIGS.batch_process.path_to_vocab_tree)
    _imageMatching = graph.addNewNode(
        "ImageMatching",
        input=_featureExtraction.input,
        featuresFolders=[_featureExtraction.output],
        tree=CONFIGS.batch_process.path_to_vocab_tree)
    out.append(imageMatching)
    out.append(_imageMatching)

    featureMatching = graph.addNewNode(
        "FeatureMatching", input=imageMatching.input,
        featuresFolders=imageMatching.featuresFolders,
        imagePairsList=imageMatching.output,
        describerTypes=featureExtraction.describerTypes,
        matchFromKnownCameraPoses=True)
    _featureMatching = graph.addNewNode(
        "FeatureMatching", input=_imageMatching.input,
        featuresFolders=_imageMatching.featuresFolders,
        imagePairsList=_imageMatching.output,
        describerTypes=_featureExtraction.describerTypes,
        matchFromKnownCameraPoses=True)
    out.append(featureMatching)
    out.append(_featureMatching)

    structureFromMotion = graph.addNewNode(
        "StructureFromMotion", input=featureMatching.input,
        featuresFolders=featureMatching.featuresFolders,
        matchesFolders=[featureMatching.output],
        describerTypes=featureMatching.describerTypes,
        useRigConstraint=False,
        lockScenePreviouslyReconstructed=True,
        lockAllIntrinsics=True)

    _structureFromMotion = graph.addNewNode(
        "StructureFromMotion", input=_featureMatching.input,
        featuresFolders=_featureMatching.featuresFolders,
        matchesFolders=[_featureMatching.output],
        describerTypes=_featureMatching.describerTypes,
        useRigConstraint=False,
        lockScenePreviouslyReconstructed=True,
        lockAllIntrinsics=True)
    out.append(structureFromMotion)
    out.append(_structureFromMotion)

    sfMAlignment = graph.addNewNode(
        "SfMTransfer", input=structureFromMotion.output,
        reference=_structureFromMotion.output,
        method="from_viewid",
        transferPoses=True,
        transferIntrinsics=True)
    out.append(sfMAlignment)
    return out

def customSfmPipeline(graph, cameraInit=None):
    """
    Instnatiate SfM pipeline for photogrammetry pipeline.
    Uses custom defined cameraInit file.
    """
    out = []
    # if cameraInit is None:
    _cameraInit = graph.addNewNode("CameraInit")
    out.append(_cameraInit)
    featureExtraction = graph.addNewNode(
            "FeatureExtraction",
            # input=cameraInit if cameraInit is not None else _cameraInit.output,
            input= _cameraInit.output,
            forceCpuExtraction=False)
    out.append(featureExtraction)
    imageMatching = graph.addNewNode(
        "ImageMatching",
        input=featureExtraction.input,
        featuresFolders=[featureExtraction.output],
        tree=CONFIGS.batch_process.path_to_vocab_tree)
    out.append(imageMatching)
    featureMatching = graph.addNewNode(
        "FeatureMatching", input=imageMatching.input,
        featuresFolders=imageMatching.featuresFolders,
        imagePairsList=imageMatching.output,
        describerTypes=featureExtraction.describerTypes,
        matchFromKnownCameraPoses=True)
    out.append(featureMatching)
    structureFromMotion = graph.addNewNode(
        "StructureFromMotion", input=cameraInit if cameraInit is not None else featureMatching.input,
        featuresFolders=featureMatching.featuresFolders,
        matchesFolders=[featureMatching.output],
        describerTypes=featureMatching.describerTypes,
        useRigConstraint=False,
        lockScenePreviouslyReconstructed=True,
        lockAllIntrinsics=True)
    out.append(structureFromMotion)
    return out

def sfmPipeline(graph, init=None):
    """
    Instantiate custom pipeline graph for photogrammetry.
    Currently uses default.
    """
    if (type(init) is list) and len(init) > 1:
        return twowaySfmPipeline(graph, init)
    else:
        return customSfmPipeline(graph, init)

def defaultMvsPipeline(graph, sfm=None):
    """
    Instantiate SfM pipeline for photogrammetry pipeline.
    Based on default photogrammetry pipeline in meshroom.
    """
    if sfm and not sfm.nodeType in [
        "StructureFromMotion", "SfMAlignment", "SfMTransfer"]:
        raise ValueError(
            "Invalid node type. Expected SfM, got {}".
            format(sfm.nodeType))
    prepareDenseScene = graph.addNewNode(
        "PrepareDenseScene", input=sfm.output if sfm else "")
    depthMap = graph.addNewNode(
        "DepthMap", input=prepareDenseScene.input,
        imagesFolder=prepareDenseScene.output)
    depthMapFilter = graph.addNewNode(
        "DepthMapFilter", input=depthMap.input,
        depthMapsFolder=depthMap.output)
    meshing = graph.addNewNode(
        "Meshing", input=depthMapFilter.input,
        depthMapsFolder=depthMapFilter.output)
    meshFiltering = graph.addNewNode(
        "MeshFiltering", inputMesh=meshing.outputMesh)
    texturing = graph.addNewNode(
        "Texturing", input=meshing.output,
        imagesFolder=depthMap.imagesFolder,
        inputMesh=meshFiltering.outputMesh)
    return [
        prepareDenseScene,
        depthMap,
        depthMapFilter,
        meshing,
        meshFiltering,
        texturing]

def customMvsPipeline(graph, sfm=None, label_dir=None):
    """
    Instantiate SfM pipeline for photogrammetry pipeline.
    Based on default photogrammetry pipeline in meshroom.
    """
    if sfm and not sfm.nodeType in [
        "StructureFromMotion", "SfMAlignment", "SfMTransfer"]:
        raise ValueError(
            "Invalid node type. Expected SfM, got {}".
            format(sfm.nodeType))
    prepareDenseScene = graph.addNewNode(
        "PrepareDenseScene", input=sfm.output if sfm else "",
        outputFileType="png")
    depthMap = graph.addNewNode(
        "DepthMap", input=prepareDenseScene.input,
        imagesFolder=prepareDenseScene.output)
    depthMapFilter = graph.addNewNode(
        "DepthMapFilter", input=depthMap.input,
        depthMapsFolder=depthMap.output)
    meshing = graph.addNewNode(
        "Meshing", input=depthMapFilter.input,
        depthMapsFolder=depthMapFilter.output)
    meshFiltering = graph.addNewNode(
        "MeshFiltering", inputMesh=meshing.outputMesh)
    texturing = graph.addNewNode(
        "Texturing", input=meshing.output,
        imagesFolder=depthMap.imagesFolder,
        downscale=2,
        inputMesh=meshFiltering.outputMesh)
    labelling = graph.addNewNode(
        "Texturing", input=meshing.output,
        imagesFolder=label_dir,
        downscale=4,
        # unwrapMethod="ABF",
        # fillHoles=True,
        inputMesh=meshFiltering.outputMesh)
    return [
        prepareDenseScene,
        depthMap,
        depthMapFilter,
        meshing,
        meshFiltering,
        texturing,
        labelling]

def mvsPipeline(graph, sfm=None, label_dir=None):
    """
    Instantiate custom pipeline graph for photogrammetry.
    Currently uses default.
    """
    if label_dir is None:
        return defaultMvsPipeline(graph, sfm)
    else:
        return customMvsPipeline(graph, sfm, label_dir)


def build_graph(
        inputImages=[], inputViewpoints=[],
        inputIntrinsics=[], output="",
        graph=None, init=None, label_dir=None):
    """ Custom photogrammetry graph """
    if graph is None:
        graph = Graph("Custom photogrammetry")
    with GraphModification(graph):
        if init is None:
            sfmNodes = sfmPipeline(graph)
            cameraInit = sfmNodes[0]
            cameraInit.viewpoints.extend([{"path": img} for img in inputImages])
            cameraInit.viewpoints.extend(inputViewpoints)
            cameraInit.intrinsics.extend(inputIntrinsics)
        else:
            sfmNodes = sfmPipeline(graph, init)
            if type(init) is list and len(init) > 1:
                pass
            else:
                cameraInit = sfmNodes[0]
                cameraInit.viewpoints.extend([{"path": img} for img in inputImages])
                cameraInit.viewpoints.extend(inputViewpoints)
                cameraInit.intrinsics.extend(inputIntrinsics)

        mvsNodes = mvsPipeline(graph, sfmNodes[-1], label_dir)
        if output:
            texturing = mvsNodes[-1]
            graph.addNewNode(
                "Publish", output=output,
                inputFiles=[
                    texturing.outputMesh,
                    texturing.outputMaterial,
                    texturing.outputTextures])
    return graph
