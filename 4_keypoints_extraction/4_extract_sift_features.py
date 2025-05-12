# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# April 2025


import os
import glob
import subprocess
from tqdm import tqdm
from natsort import natsorted
import argparse
#import json


def main(input_path: str, world: bool = False, sift_path: str = "./3DSIFT-Rank/build/featExtract/featExtract") -> None:
    """
    Extracts SIFT-Ranked features from NIfTI images in a given directory.
    For each case in the input directory, the most relevant NIfTI image (prioritizing T1ce, and then T2) 
    is identified and 3DSIFT-Rank is ran to compute and save keypoints.

    Args:
        input_path (str): Path to a directory containing case folders with images in NIfTI format.
        world (bool, optional): If True, features are extracted in world coordinates; otherwise, voxel space is used. Defaults to False.
        sift_path (str, optional): Path to the SIFT feature extraction binary. Defaults to "./3DSIFT-Rank/build/featExtract/featExtract".
    """
    for case in tqdm(natsorted(os.listdir(input_path))):
        if world:
            tqdm.write(f"Generating keypoints for case '{case}' (using world space)...")
        else:
            tqdm.write(f"Generating keypoints for case '{case}' (using voxel space)...")

        case_path = os.path.join(input_path, case)
        # Search for the image files in the case directory
        # First look for T1GD from UPENN-GBM, then T1_postcontrast from ReMIND, then T2 from both datasets, 
        # and finally any other NIfTI file available in the case directory.
        image_query = [
        "*T1GD*.nii.gz",
        "*T1_postcontrast*.nii.gz",
        "*T2*.nii.gz",
        "*.nii.gz"
        ]
        for im in image_query:
            image = glob.glob(os.path.join(case_path, im))
            if image:
                image = image[0]
                break
        image_basename = os.path.basename(image)
        if case.startswith("ReMIND"):
            filename = "_".join(image_basename.split("_")[2:])[:-7]
        else:
            filename = image_basename.split(".")[0]

        out_dir = os.path.join(case_path, "keypoints")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{filename}{'_w' if world else ''}.key")
        if os.path.exists(out):
            tqdm.write(f"\tKeypoints have already been saved to '{out}'. Skipping.\n")
            continue
        
        # Format the command to run the SIFT feature extraction binary depending on the provided flags
        cmd = [sift_path]
        if world:
            cmd.append("-w")
        cmd.extend([image, out])

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        tqdm.write(f"\tKeypoints saved to '{out}'\n")


if __name__ == "__main__":
    #with open("config.json", "r") as f:
    #    config_file = json.load(f)
    #sift_path = config_file["sift_path"]

    parser = argparse.ArgumentParser(description="Extracts SIFT features from NIfTI images.")
    parser.add_argument("input_path", type=str, help="Path to the input directory containing NIfTI images.")
    parser.add_argument("--world", "-w", action="store_true", help="Use world space for SIFT feature extraction (default is voxel space).")
    args = parser.parse_args()

    assert os.path.exists(args.input_path), "Input path does not exist."
    assert os.path.isdir(args.input_path), f"{args.input_path} is not a valid directory."
    assert len([dir for dir in os.listdir(args.input_path)]) > 0, "Input path is empty."

    #assert os.path.exists(sift_path), "SIFT feature extractor binary path does not exist."

    main(args.input_path, args.world)
