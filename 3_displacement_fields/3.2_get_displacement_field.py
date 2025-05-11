# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# April 2025


import os
import numpy as np
import subprocess
from natsort import natsorted
from tqdm import tqdm
import SimpleITK as sitk
import glob
import time
import datetime
import argparse
import gc
import json


def calculate_transform(init_coords_path: str, final_coords_path: str, output_path: str, tf_cli_path: str) -> None:
    """
    Calculates a transformation between two sets of coordinates using the ScatteredTransform CLI extension of 3D Slicer,
    and saves the resulting transform to an output file.
    
    Args:
        init_coords_path (str): Path to the TXT or CSV file containing the initial coordinates.
        final_coords_path (str): Path to the TXT or CSV file containing the displaced coordinates.
        output_path (str): Path where the output transform file should be saved to.
        tf_cli_path (str): Path to the ScatteredTransform CLI binary.
    """
    init_coords = np.genfromtxt(init_coords_path, delimiter=",")
    final_coords = np.genfromtxt(final_coords_path, delimiter=",")
    
    # Get the min and max coordinates from both sets to create a bounding box for the transform
    min_init = np.min(init_coords, axis=0)
    max_init = np.max(init_coords, axis=0)
    min_final = np.min(final_coords, axis=0)
    max_final = np.max(final_coords, axis=0)

    min_coords = np.array([])
    max_coords = np.array([])
    
    for mii, mif in zip(min_init, min_final):
        min_coords = np.append(min_coords, min(mii, mif))
    for mai, maf in zip(max_init, max_final):
        max_coords = np.append(max_coords, max(mai, maf))

    # Add a small margin to avoid issues with the images not fitting the transform boundary box
    min_coords -= 1
    max_coords += 1

    # ScatteredTransform CLI command
    cmd = [
        tf_cli_path,
        "--initialPointsFile", init_coords_path,
        "--displacedPointsFile", final_coords_path,
        "--bsplineTransformFile", output_path,
        "--transformSpaceDimension", "3D",
        "--intendedUse", "ITK",
        "--invertTransform",
        "--useLinearApproximation",
        "--splineGridSpacing", "100,100,100",
        "--minCoordinates", f"{','.join([str(v) for v in min_coords])}",
        "--maxCoordinates", f"{','.join([str(v) for v in max_coords])}",
        "--tolerance", "0.1",
        "--minGridSpacing", "0.1",
        "--maxNumLevels", "10"
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    

def get_displacement_field(transform_path: str, reference_path: str, output_path: str, nifti: bool = False) -> None:
    """
    Computes the displacement field from a given transform and reference image, then saves the displacement field 
    in (both NIfTI and) NumPy format(s) as a (3, D, H, W) tensor.
    
    Args:
        transform_path (str): Path to the transform file (.h5).
        reference_path (str): Path to the reference image used for computing the displacement field.
        output_path (str): Directory path where the displacement field should be saved.
    """
    transform = sitk.ReadTransform(transform_path)
    reference_image = sitk.ReadImage(reference_path)

    disp_field = sitk.TransformToDisplacementField(transform, 
                                                   sitk.sitkVectorFloat32, 
                                                   reference_image.GetSize(), 
                                                   reference_image.GetOrigin(), 
                                                   reference_image.GetSpacing(), 
                                                   reference_image.GetDirection())
    
    # Save as .nii.gz if requested
    #if nifti:
    #    nii_path = os.path.join(output_path, f"disp_field_{case_num}.nii.gz")
    #    sitk.WriteImage(disp_field, nii_path, True)

    # Convert to NumPy and save as .npz
    disp_array = sitk.GetArrayFromImage(disp_field)  # (D, H, W, 3)
    disp_array = np.moveaxis(disp_array, -1, 0) # (3, D, H, W) for PyTorch (?)
    np.savez_compressed(output_path, field=disp_array)

    # Clean up memory because these transforms can be large
    del transform, reference_image, disp_field, disp_array
    gc.collect()
    gc.collect()


def main(input_path: str, tf_cli_path: str, nifti: bool = False) -> None:
    """
    Processes all cases in the specified input directory, calculating the transform from a set of initial and final coordinates, 
    and extracting the corresponding 4D displacement field tensor.
    
    Args:
        input_path (str): Path to a directory containing case folders with simulation subfolders where the coordinate files exist.
        nifti (bool, optional): If True, also saves the displacement field in NIfTI format. Defaults to False.
        tf_cli_path (str, optional): Path to the ScatteredTransform CLI binary.
    """
    intputs = natsorted(os.listdir(input_path))
    for case in tqdm(intputs):
        # Specific formatting for file naming purposes depending on the dataset (ReMIND or UPENN-GBM)
        case_num = case.split("-")[-1]
        if case.startswith("UPENN"):
            case_num = case_num[:-3]
        
        case_path = os.path.join(input_path, case)
        simulation_paths = glob.glob(os.path.join(case_path, "simulation*"))
        for sim_path in natsorted(simulation_paths):
                sim = os.path.split(sim_path)[1].replace("simulation", "")
                tf_output_path = os.path.join(sim_path, f"transform_{case_num}_{sim}.h5")

                tqdm.write(f"Processing case '{case}', simulation #{sim}...")
                if not os.path.exists(tf_output_path):
                    init_coords_path = os.path.join(sim_path, "init_coord.txt")
                    final_coords_path = os.path.join(sim_path, "final_coord.txt")

                    tqdm.write(f"\tCalculating transform. This may take a few minutes...")
                    start = time.time()
                    calculate_transform(init_coords_path, final_coords_path, tf_output_path, tf_cli_path)
                    tqdm.write(f"\tTransform calculated in:  {datetime.timedelta(seconds=time.time()-start)}")
                else:
                    tqdm.write(f"\tTransform for this case already exists. Skipping.")

                # Search for the reference image in the case directory
                reference_path = glob.glob(os.path.join(case_path, f"resampled_*T*{case_num}.nrrd"))
                if len(reference_path) == 0:
                    reference_path = glob.glob(os.path.join(case_path, f"*T*{case_num}.nrrd"))

                tqdm.write("\tExtracting displacement field...")
                npz_out_path = os.path.join(sim_path, f"disp_field_{case_num}_{sim}.npz")
                if os.path.exists(npz_out_path):
                    tqdm.write(f"\tDisplacement field for this case already exists. Skipping.")
                    continue   

                start = time.time() 
                get_displacement_field(tf_output_path, reference_path[0], npz_out_path, nifti)
                tqdm.write(f"\tDisplacement field saved in: {round(time.time() - start, 2)}s.")
        tqdm.write(f"Case '{case}' processed.\n")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config_file = json.load(f)
    tf_cli_path = config_file["scatteredtransform_cli_path"]

    parser = argparse.ArgumentParser(description="Calculate the transform and the displacement field from coordinate files in batch.")
    parser.add_argument("input_path", type=str, help="Path to the input directory of case folders containing simulation subfolders with initial and displaced coordinates.")
    #parser.add_argument("--nifti", action="store_true", help="Also saves the displacement field in NIfTI format.")
    args = parser.parse_args()

    assert os.path.exists(args.input_path), "Input path does not exist."
    assert os.path.isdir(args.input_path), f"{args.input_path} is not a valid directory."
    assert len([subd for d in os.listdir(args.input_path) for subd in os.listdir(os.path.join(args.input_path, d))]) > 0, "Input path is empty."

    assert os.path.exists(tf_cli_path), "ScatteredTransform CLI binary path does not exist."

    main(args.input_path, tf_cli_path)#, args.nifti)
