# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import slicer
from slicer.ScriptedLoadableModule import *
import vtk
import sys
import argparse
import os


def main(
    input_brain_mask: str,
    tumor_mask: str,
    input_ref_vol: str,
    output_model: str,
    output_brain_mask: str,
    output_tumor_mask: str
) -> None:
    """
    This script is executed within the 3D Slicer (5.6.2) environment to process and edit brain and tumor segmentations 
    for the biomechanical simulations.

    It performs the following tasks:
    1. Loads a brain segmentation predicted by SynthSeg, a tumor segmentation, and the original brain scan volume.
    2. Merges tumor segments into the brain segmentation to overwrite any malformed regions.
    3. Hides unwanted anatomical segments (e.g., the cerebellum).
    4. Applies logical and morphological editing operations (union, smoothing, cleanup) using the Segment Editor.
    5. Exports a surface model of the brain as an STL file.
    6. Refines the tumor segmentation by removing irrelevant parts (e.g., edema) and outputs a cleaned tumor labelmap.

    Args:
        input_brain_mask (str): Path to SynthSeg's predicted brain segmentation.
        input_tumor_mask (str): Path to the tumor segmentation.
        input_ref_vol (str): Path to the reference brain scan.
        output_model (str): Path to save the generated surface brain model (.stl).
        output_brain_mask (str): Path to save the edited brain segmentation (.nrrd).
        output_tumor_mask (str): Path to save the edited tumor segmentation (.nrrd).

    Usage:
        python <this_script.py> --edit-mask <input_brain_mask> <input_tumor_mask> <input_ref_vol> <output_model> <output_brain_mask> <output_tumor_mask>

    Requires:
        - 3D Slicer with access to Segment Editor and ModelMaker.
        - Pre-existing brain and tumor segmentations.

    Note:
        This script minimizes the main Slicer window during execution and exits the application upon completion.
    """
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    os.makedirs(os.path.dirname(output_brain_mask), exist_ok=True)
    os.makedirs(os.path.dirname(output_tumor_mask), exist_ok=True)

    # Load SynthSeg predicted brain segmentation, tumor segmentation, and reference brain volume
    segmentationNode = slicer.util.loadSegmentation(input_brain_mask)
    tumorNode = slicer.util.loadSegmentation(tumor_mask)
    referenceVolumeNode = slicer.util.loadVolume(input_ref_vol)

    # Copy the tumor segmentation to the brain segmentation to make sure any SynthSeg error caused by the presence of the tumor is filled
    segmentationNode.GetSegmentation().CopySegmentFromSegmentation(tumorNode.GetSegmentation(), tumorNode.GetSegmentation().GetNthSegmentID(0), False)
    segmentationNode.GetSegmentation().CopySegmentFromSegmentation(tumorNode.GetSegmentation(), tumorNode.GetSegmentation().GetNthSegmentID(1), False)
    segmentationNode.GetSegmentation().CopySegmentFromSegmentation(tumorNode.GetSegmentation(), tumorNode.GetSegmentation().GetNthSegmentID(2), False)
    
    # The brain segmentation consists of ~60 different segments representing different areas of the brain
    # These segment numbers refer to the cerebellum, which is hidden (as the authors did in the original paper)
    # This was assessed manually in 3D Slicer, and the segment numbers noted
    segments_to_hide = [f"Segment_{i}" for i in [7,8,15,46,47]]

    # Disable visibility of these segment IDs
    segmentation = segmentationNode.GetSegmentation()
    num_segments = segmentation.GetNumberOfSegments()
    for i in range(num_segments):
        segment_id = segmentation.GetNthSegmentID(i)
        if segment_id in segments_to_hide:
            segmentationNode.GetDisplayNode().SetSegmentVisibility(segment_id, False)

    # Initializes the segmentation editor node and widget in the scene
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(segmentEditorNode)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

    # Set parameters that will make so that the editor only changes areas inside of the segments
    # and is allowed to overwrite the active segment
    segmentEditorNode.SetMaskMode(2)
    segmentEditorNode.SetOverwriteMode(0)

    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(referenceVolumeNode)

    # Activates the "Logical operators" effect in the segmentation editor
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    logical_operators = segmentEditorWidget.activeEffect()
    # Fills in all the segments, i.e., makes the union of all segments
    logical_operators.setParameter("Operation", "FILL") ###
    logical_operators.setParameter("BypassMasking", 0)
    logical_operators.self().onApply()

    # Removes all the segments except the first one, which was the one that was overwritten with the union of segments
    # Iterate backwards to avoid index shifting
    for i in range(num_segments-1,0,-1):  
        segment_id = segmentation.GetNthSegmentID(i)
        segmentation.RemoveSegment(segment_id)

    # Now the edits only change the areas outside the segments
    segmentEditorNode.SetMaskMode(4)

    # Activates the "Smoothing" effect in the segmentation editor
    segmentEditorWidget.setActiveEffectByName("Smoothing")
    smoothing = segmentEditorWidget.activeEffect()

    # Removes extrusions smaller than a certain kernel size
    smoothing.setParameter("SmoothingMethod", "MORPHOLOGICAL_OPENING")
    smoothing.setParameter("KernelSizeMm", 5.0)
    smoothing.self().onApply()

    # Fills holes smaller than a certain kernel size
    smoothing.setParameter("SmoothingMethod", "MORPHOLOGICAL_CLOSING")
    smoothing.setParameter("KernelSizeMm", 10.0)
    smoothing.self().onApply()

    # Removes extrusions smaller than a certain kernel size
    smoothing.setParameter("SmoothingMethod", "MORPHOLOGICAL_OPENING")
    smoothing.setParameter("KernelSizeMm", 20.0)
    smoothing.self().onApply()

    # Two steps of removing extrusions is performed: the first step removes small extrusions that can affect the subsequent step of filling in holes,
    # i.e., if the first step is not performed, when the holes are filled, a lot of wrong space will be filled due to wrong extrusions being present
    # The second step does a final extrusion removal to smooth out the contour

    # Makes the contour more smooth // not really necessary
    #smoothing.setParameter("SmoothingMethod", "MEDIAN")
    #smoothing.setParameter("KernelSizeMm", 5.0)
    #smoothing.self().onApply()

    # Export the segmentation to a labelmap node with the reference volume affine matrix
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, referenceVolumeNode)

    # Set up ModelMaker to generate the surface brain model
    modelHierarchyNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelHierarchyNode")
    mm_parameters = {
        "InputVolume": labelmapVolumeNode.GetID(),
        "ModelSceneFile": modelHierarchyNode.GetID(),
        "Smooth": 10.0,
        "FilterType": "Laplacian",
        "Decimate": 0,
        "SplitNormals": False,
        "PointNormals": False,
        "Pad": True
    }

    # Run ModelMaker
    modelmaker = slicer.modules.modelmaker
    slicer.cli.runSync(modelmaker, None, mm_parameters)

    # Get model node and save it to an output path
    modelNode = slicer.util.getNodesByClass("vtkMRMLModelNode")[-1]
    slicer.util.saveNode(modelNode, output_model, {"useCompression": 1})

    # Save the brain segmentation labelmap volume
    slicer.util.saveNode(labelmapVolumeNode, output_brain_mask, {"useCompression": 1})

    # Removes the edema from the tumor segmentation, as it is not useful for the biomechanical simulation
    # Only the tumor mass will be simulated ...
    segmentation = tumorNode.GetSegmentation()
    segmentation.RemoveSegment(segmentation.GetNthSegmentID(1))

    segmentEditorNode.SetMaskMode(2)
    segmentEditorNode.SetOverwriteMode(0)

    segmentEditorWidget.setSegmentationNode(tumorNode)
    segmentEditorWidget.setSourceVolumeNode(referenceVolumeNode)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    logical_operators = segmentEditorWidget.activeEffect()
    # Like the very first step done to the brain segmentation, but now we are getting the union of the tumor segments (necrotic and non-enhancing segments)
    logical_operators.setParameter("Operation", "FILL")
    logical_operators.setParameter("BypassMasking", 0)
    logical_operators.self().onApply()

    # Exports to labelmap volume and saves the tumor segmentation
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(tumorNode, labelmapVolumeNode, referenceVolumeNode)
    slicer.util.saveNode(labelmapVolumeNode, output_tumor_mask, {"useCompression": 1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs=6, help="Takes an input brain segmentation and the respective reference volume, \
                        and outputs an edited brain mask, tumor mask labelmap, and generated brain surface model. \
                        Format: <input_brain_mask>, <input_tumor_mask>, <input_reference_volume>, <output_model_path>, <output_brain_mask_path>, <output_tumor_seg_path>")
    args = parser.parse_args()

    assert os.path.exists(args.inputs[0]), f"Input brain segmentation not found: {args.inputs[0]}"
    assert os.path.exists(args.inputs[1]), f"Input tumor segmentation not found: {args.inputs[1]}"
    assert os.path.exists(args.inputs[2]), f"Input reference volume not found: {args.inputs[2]}"
    assert args.inputs[3].endswith(".stl"), f"Output surface model must be in .stl format: {args.inputs[3]}"
    assert args.inputs[4].endswith(".nrrd"), f"Output brain segmentation must be in .nrrd format: {args.inputs[4]}"
    assert args.inputs[5].endswith(".nrrd"), f"Output tumor segmentation must be in .nrrd format: {args.inputs[5]}"
    
    # Minimizes Slicer on launch
    slicer.util.mainWindow().showMinimized()

    slicer.app.processEvents()

    main(*args.inputs)

    # Clears the scene and closes Slicer
    slicer.mrmlScene.Clear()
    sys.exit()
