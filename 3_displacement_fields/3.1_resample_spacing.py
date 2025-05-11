# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# April 2025

import os
import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import argparse
from natsort import natsorted


def resample_spacing(input_path: str, output_path: str, new_spacing: list[float] = [1.0, 1.0, 1.0], interpolator: int = sitk.sitkLinear, \
                     verbose: bool = False) -> None:
    """
    Resamples the spacing of a medical image to the specified new spacing, saving the result to the given output file.
    
    Args:
        input (str): Path to a directory containing case folders with images in NIfTI or NRRD files.
        output (str): Path to save the resampled image.
        new_spacing (list of float, optional): Desired spacing for the output image. Defaults to [1.0, 1.0, 1.0].
        interpolator (int, optional): Interpolator to be used during resampling. Default to sitk.sitkLinear.
        verbose (bool, optional): If True, prints additional information about the resampling process. Defaults to False.
    """
    np.set_printoptions(precision=3)
    
    input_img = os.path.split(input_path)[1]
    if verbose:
        tqdm.write(f"Resampling spacing for image '{input_img}'...")
    
    input_image = sitk.ReadImage(input_path)
    
    original_spacing = input_image.GetSpacing()

    # Skip if the image already has the desired spacing
    if original_spacing == new_spacing:
        if verbose:
            tqdm.write(f"\tImage '{input_img}' already has the desired spacing. Skipping resampling.\n")
        return

    original_size = input_image.GetSize()
    original_origin = input_image.GetOrigin()
    original_direction = input_image.GetDirection()
    if verbose:
        tqdm.write(f"\tOriginal spacing: {np.array(original_spacing)}", )
        tqdm.write(f"\tOriginal size: {np.array(original_size)}", )

    # Calculate new size based on the new spacing and original size
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    if verbose:
        tqdm.write(f"\tNew spacing: {new_spacing}")
        tqdm.write(f"\tNew size: {new_size}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputPixelType(input_image.GetPixelID())
    resampler.SetInterpolator(interpolator)

    resampled_image = resampler.Execute(input_image)

    sitk.WriteImage(resampled_image, output_path, useCompression=True)
    if verbose:
        tqdm.write(f"Resampled image saved to '{output_path}'\n")


def main(input_path: str, new_spacing: list[float] = [1.0, 1.0, 1.0], interpolator: int = sitk.sitkLinear, \
         verbose: bool = False, nifti: bool = False, nrrd: bool = True, remind: bool = True) -> None:
    """
    Processes all medical images in the specified input path and resamples them to the desired spacing.
    
    Args:
        input_path (str): Path to a directory containing case folders with images in NIfTI or NRRD files.
        new_spacing (list of float): Desired spacing for the output images. Defaults to [1.0, 1.0, 1.0].
        interpolator (int): Interpolator to be used during resampling. Default is sitk.sitkLinear.
        verbose (bool, optional): If True, prints additional information about the processing. Defaults to False.
        nifti (bool, optional): If True, only processes NIfTI images. Defaults to False.
        nrrd (bool, optional): If True, only processes NRRD images. Defaults to False.
        remind (bool, optional): If True, only processes ReMIND images. Defaults to True.
    """
    inputs = natsorted(os.listdir(input_path))
    for case in tqdm(inputs):
        if remind:
            # Skip if the case is not a ReMIND case
            if not case.startswith("ReMIND"):
                continue
        nrrd_images = glob.glob(os.path.join(input_path, case, "*.nrrd"))
        nifti_images = glob.glob(os.path.join(input_path, case, "*.nii*"))
        # Select the images to process based on the provided flags. If no flags are set, process both NRRD and NIfTI images.
        images = nrrd_images + nifti_images if nifti and nrrd \
            else (nrrd_images if nrrd \
                  else (nifti_images if nifti \
                        else nrrd_images + nifti_images))
        for image in images:
            out_image = os.path.split(image)[1]
            # Skip if the image is already resampled
            if "resampled" in out_image:
                continue
            output = os.path.join(input_path, case, "resampled_" + out_image)
            if os.path.exists(output):
                if verbose:
                    tqdm.write(f"Image '{out_image}' has already been resampled. Skipping.\n")
                continue
            # Check if the image is a segmentation or mask and set the interpolator accordingly
            if (("seg" in out_image) or ("mask" in out_image)) and (interpolator != sitk.sitkNearestNeighbor):
                if verbose:
                    tqdm.write(f"Image '{out_image}' seems to be a segmentation. Image will be resampled with a nearest neighbor interpolation mode.\n")
                resample_spacing(image, output, new_spacing, sitk.sitkNearestNeighbor, verbose)
            else:
                resample_spacing(image, output, new_spacing, interpolator, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample images (NRRD or NIfTI) to any provided voxel spacing in batch.")
    parser.add_argument("input_path", type=str, help="Path to the input directory containing case folders with images.")
    parser.add_argument("--spacing", "-s", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Desired spacing for the output image (default: [1.0 1.0 1.0]).")
    parser.add_argument("--interpolator", "-i", type=str, default="bspline", choices=["linear", "nearest", "bspline"], help="Interpolation method to use (default: bspline).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enables verbose output.")
    parser.add_argument("--nifti", "--nii", action="store_true", help="Indicates to only resample NIfTI images.")
    parser.add_argument("--nrrd", action="store_true", help="Indicates to only resample NRRD images.")
    parser.add_argument("--remind", action="store_true", help="Indicates to only resample the ReMIND cases.")
    args = parser.parse_args()
    
    assert os.path.exists(args.input_path), "Input path does not exist."
    assert os.path.isdir(args.input_path), f"{args.input_path} is not a valid directory."
    assert len([dir for dir in os.listdir(args.input_path)]) > 0, "Input path is empty."

    assert len(args.spacing) == 3, "Spacing must be a list of three floats or integers."
    assert all(isinstance(i, (int, float)) for i in args.spacing), "Spacing must be a list of three floats or integers."
    assert all(i > 0 for i in args.spacing), "Spacing values must be positive."

    if args.interpolator == "linear":
        interpolator = sitk.sitkLinear
    elif args.interpolator == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    elif args.interpolator == "bspline":
        interpolator = sitk.sitkBSpline

    main(args.input_path, args.spacing, interpolator, args.verbose, args.nifti, args.nrrd, args.remind)
