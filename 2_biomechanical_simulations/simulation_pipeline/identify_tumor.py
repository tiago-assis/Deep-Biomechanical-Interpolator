#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:31:01 2021

@author: yue
"""

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import os
import time
import numpy as np
import mvloader.nrrd as mvnrrd
import nrrd
import meshio
import numpy as np
import sys
import multiprocessing as mp
from typing import Tuple, List


def main(tumor_mask_file_path: str, brain_mask_file_path: str) -> None:
    """
    Main function to process tumor and brain segmentation files, identifying the tumor nodes
    in the brain volumetric mesh and generating output files with indices of related nodes for later use
    in the simulation pipeline.

    Args:
        tumor_mask_file_path (str): Path to the tumor segmentation file (.nrrd).
        brain_mask_file_path (str): Path to the brain segmentation file (.nrrd).
    """
    tumor_mask, t_header = trim_tumor(tumor_mask_file_path, brain_mask_file_path)
    print("\nIdentifying the tumor nodes in the brain mesh...")
    identify_tumor(tumor_mask_file_path, tumor_mask, t_header)


def trim_tumor(tumor_mask_file_path: str, brain_mask_file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Trims any tumor segmentation that may sit outside the brain segmentation.
    Saves the modified brain mask as a temporary file for later use.

    Args:
        tumor_mask_file_path (str): Path to the tumor segmentation file (.nrrd).
        brain_mask_file_path (str): Path to the brain segmentation file (.nrrd).

    Returns:
        Tuple[np.ndarray, dict]: The trimmed tumor segmentation and its header information.
    """
    tumor_mask, t_header = nrrd.read(tumor_mask_file_path)
    brain_mask, b_header = nrrd.read(brain_mask_file_path)
    tumor_mask[brain_mask == 0] = 0
    nrrd.write("./5_memberships/brain_mask_temp.nrrd", brain_mask, b_header)
    return tumor_mask, t_header

def identify_tumor(tumor_mask_file_path: str, tumor_mask: np.ndarray, t_header: dict) -> None:
    """
    Identifies the nodes in the brain mesh that belong to the tumor and generates related output files
    including the mesh element indices of the tumor and the coordinates of the tumor boundary nodes.

    Args:
        tumor_mask_file_path (str): Path to the tumor segmentation file (.nrrd).
        tumor_mask (np.ndarray): The tumor segmentation array.
        t_header (dict): Header data for the tumor segmentation.
    """
    start = time.time()

    # Load tumor segmentation data
    tumor_info = mvnrrd.open_image(tumor_mask_file_path, verbose=False) # .nrrd file
    nrrd.write("./2_tumor_idxs/tumor_mask_temp.seg.nrrd", tumor_mask, t_header) # to be used later

    # Get affine matrix to transform world coordinates to voxel indices
    voxels2world = tumor_info.src_transformation
    world2voxels = np.linalg.inv(voxels2world)

    # Load brain volume mesh and get the brain nodes
    brainmesh = meshio.read("./1_brain_meshes/brain_tetravolume.inp") # .inp file
    brain_nodes = brainmesh.points
    tetras = brainmesh.cells_dict["tetra"]


    # Add an ID number to the brain node coordinates
    ids_nodes = np.arange(1, brain_nodes.shape[0] + 1).reshape(-1, 1)
    brain_nodes_new = np.hstack((ids_nodes, brain_nodes))

    # Add an ID number to the brain tetrahedral point coordinates
    ids_tetras = np.arange(1, tetras.shape[0] + 1).reshape(-1, 1)
    brain_tetras_new = np.hstack((ids_tetras, tetras + 1))


    # Append a 1 to the brain node coordinates (homogeneous coordinate system)
    coords = brain_nodes_new[:, 1:]  # shape (N, 3)
    ones = np.ones((coords.shape[0], 1), dtype=coords.dtype)
    coords_h = np.hstack((coords, ones))  # shape (N, 4)

    # Convert world coordinates to voxel indices (rounding to the closest integer)
    voxel_idxs = (world2voxels[:3] @ coords_h.T).T  # shape (N, 3)
    voxel_idxs = voxel_idxs.astype(int)

    # Get the values at each index of the tumor mask and select only the brain nodes that are within the tumor
    mask_values = tumor_mask[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]]
    tumour_nodes = brain_nodes_new[mask_values != 0]


    # Generate formatted file with tumor nodes IDs (can be used in abaqus inp file)
    tumour_idx = tumour_nodes[:,0].astype(int)

    with open("./2_tumor_idxs/tumour_idx.txt", "w") as f:
        for t_idx in range(0, tumour_idx.shape[0]):
            if (t_idx + 1) % 16 == 0:
                s = int(tumour_idx[t_idx])
                f.write("%d\n" %s)
            else:
                s = int(tumour_idx[t_idx])
                f.write("%d," %s)        

    print(f"\tThere are {tumour_idx.shape[0]} tumor mesh nodes.")


    tumour_ele_idx = set()
    all_element_idxs = brain_tetras_new[:, 0].astype(int)
    non_tumour_element_idx_set = set(all_element_idxs)

    for idx, t_idx in enumerate(tumour_idx):
        #if idx % 50 == 0:
        #    print("tumour_idx comes to", idx, "/", tumour_idx.shape[0], ".")
        tetra_indices = brain_tetras_new[:, 1:5]
        matching_tetras = np.any(tetra_indices == t_idx, axis=1)
        matching_tetra_ids = brain_tetras_new[matching_tetras, 0]
        tumour_ele_idx.update(matching_tetra_ids) 

    non_tumour_element_idx_set = non_tumour_element_idx_set - tumour_ele_idx

    tumour_ele_idx = np.array(list(tumour_ele_idx), dtype=int)
    non_tumour_element_idx_set = np.array(list(non_tumour_element_idx_set), dtype=int)
    print(f"\tThere are {tumour_ele_idx.shape[0]} tumor mesh elements.")


    with open("./2_tumor_idxs/tumour_ele_idx.txt", "w") as f:
        for t_idx in range(0, tumour_ele_idx.shape[0]):
            if (t_idx + 1) % 16 == 0:
                s = int(tumour_ele_idx[t_idx])
                f.write("%d\n" %s)
            else:
                s = int(tumour_ele_idx[t_idx])
                f.write("%d," %s)  


    # Precompute the indices for tumour and non-tumour elements
    tumour_ele_indices = tumour_ele_idx - 1
    non_tumour_ele_indices = non_tumour_element_idx_set - 1

    # Precompute all non-tumour tetrahedral nodes
    non_tum_tetra_nodes = brain_tetras_new[non_tumour_ele_indices, 1:5]

    # Define number of CPU cores for each job
    num_processes = mp.cpu_count()

    # Split tumour indices into chunks
    chunk_size = len(tumour_ele_indices) // num_processes
    chunks = [tumour_ele_indices[i:i + chunk_size] for i in range(0, len(tumour_ele_indices), chunk_size)]

    # Use multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_tumour_chunk, [(chunk, brain_tetras_new, non_tum_tetra_nodes) for chunk in chunks])

    # Flatten the results and remove duplicates
    sharing_nodes_id = np.unique(np.concatenate([item for sublist in results for item in sublist]))

    # Save the sharing nodes to a file
    with open("./2_tumor_idxs/sharing_nodes.txt", "w") as f:
        for t_idx, node_id in enumerate(sharing_nodes_id):
            if (t_idx + 1) % 16 == 0:
                f.write(f"{node_id}\n")
            else:
                f.write(f"{node_id},")

    # Get sharing nodes coordinates for every ID
    sharing_nodes_coordinate = np.zeros([sharing_nodes_id.shape[0],4])
    sharing_nodes_coordinate = np.hstack((sharing_nodes_id.reshape(-1, 1), brain_nodes[sharing_nodes_id.astype(int) - 1, :]))

    np.savetxt("./2_tumor_idxs/sharing_nodes_coordinate.txt", sharing_nodes_coordinate, fmt = "%d, %.8f, %.8f, %.8f", newline = '\n')


    end = time.time()
    print(f'\tTotal time for tumor nodes identification and processing: {round(end - start, 2)}s')

def process_tumour_chunk(chunk: np.ndarray, brain_tetras: np.ndarray, non_tumor_tetra_nodes: np.ndarray) -> List[np.ndarray]:
    """
    Processes a chunk of tumor elements to find the sharing nodes between the tumor and non-tumor mesh elements.

    Args:
        chunk (np.ndarray): A chunk of tumor element indices to process.
        brain_tetras (np.ndarray): Array of tetrahedral elements.
        non_tumor_tetra_nodes (np.ndarray): Precomputed non-tumor tetrahedral nodes.

    Returns:
        List[np.ndarray]: A list of shared nodes for each tumor element in the chunk.
    """
    sharing_nodes_list = []
    for tum_ele_id in chunk:
        # Get the tetrahedral nodes for the current tumour element
        tumor_tetra_nodes = brain_tetras[tum_ele_id, 1:5]
        # Find shared nodes using vectorized operations
        shared_nodes = np.intersect1d(tumor_tetra_nodes, non_tumor_tetra_nodes)
        sharing_nodes_list.append(shared_nodes)
    return sharing_nodes_list


if __name__ == '__main__':
    brain_mask_file_path, tumor_mask_file_path = sys.argv[1], sys.argv[2]
    assert os.path.exists(tumor_mask_file_path), "The tumor segmentation file does not exist."
    assert os.path.exists(brain_mask_file_path), "The brain segmentation file does not exist."
    assert tumor_mask_file_path.lower().endswith(".nrrd"), "The tumor segmentation file must be in .nrrd format."
    assert brain_mask_file_path.lower().endswith(".nrrd"), "The brain segmentation file must be in .nrrd format."
    main(tumor_mask_file_path, brain_mask_file_path)
    