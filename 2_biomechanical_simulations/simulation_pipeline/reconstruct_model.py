#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:13:14 2021

@author: yue
"""

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import time
import meshio
import numpy as np


def reconstruct_model_without_tumor() -> None:
    """
    Reconstructs the brain mesh model after removing tumor elements and nodes, simulating the brain state after resection.
    Outputs updated mesh files, along with new node mappings.
    """
    print("\nReconstructing brain mesh without tumor...")
    start = time.time()

    brainmesh = meshio.read("./1_brain_meshes/brain_tetravolume.inp") # .inp file
    brain_nodes = brainmesh.points
    tetras = brainmesh.cells_dict["tetra"]

    # Add an ID number to the brain node coordinates
    ids_nodes = np.arange(1, brain_nodes.shape[0] + 1).reshape(-1, 1)
    brain_nodes_new = np.hstack((ids_nodes, brain_nodes))

    # Add an ID number to the brain tetrahedral point coordinates
    ids_tetras = np.arange(1, tetras.shape[0] + 1).reshape(-1, 1)
    brain_tetras_new = np.hstack((ids_tetras, tetras + 1))

    # Extract tumour element ID
    with open("./2_tumor_idxs/tumour_ele_idx.txt", "r") as f:
        data = f.read().replace("\n", ",")
    data = list(data.split(","))
    data.remove('')
    tumour_ele_idx = np.array(data)

    # Get tumour element nodes info
    tumour_ele_indices = tumour_ele_idx.astype(int) - 1
    tumour_ele_nodes = brain_tetras_new[tumour_ele_indices]
    # Extract unique node IDs from tumour elements
    tumour_ele_nodes2 = np.unique(tumour_ele_nodes[:, 1:5].flatten())

    # Extract sharing nodes coordinate
    sharing_nodes_coordinate = np.genfromtxt("./2_tumor_idxs/sharing_nodes_coordinate.txt", delimiter = ',')
    sharing_nodes_id = sharing_nodes_coordinate[:, 0].astype(int)
    final_tumour_nodes_id = np.setdiff1d(tumour_ele_nodes2, sharing_nodes_id)

    # Tumour nodes coordinate
    tumour_node_ids = final_tumour_nodes_id.astype(int)
    tumour_nodes_coordinate = np.hstack((tumour_node_ids.reshape(-1, 1), brain_nodes[tumour_node_ids - 1]))

    # Tumour element info
    tumour_element_ids = tumour_ele_idx.astype(int) - 1
    tumour_element = brain_tetras_new[tumour_element_ids]

    # Create new nodes by excluding tumour node IDs
    mask_nodes = ~np.isin(brain_nodes_new[:, 0], tumour_nodes_coordinate[:, 0])
    brain_nodes_reconstructed = brain_nodes_new[mask_nodes]

    # Create new tetras by excluding tumour element IDs
    mask_tetras = ~np.isin(brain_tetras_new[:, 0], tumour_element[:, 0])
    brain_tetras_reconstructed = brain_tetras_new[mask_tetras]

    # For .vtu/.inp file output
    re_brain_nodes = brain_nodes_reconstructed[:, 1:]


    # Generate the output array without tumor
    # Initialize arrays
    brain_nodes_no_tumour = np.hstack((np.arange(1, re_brain_nodes.shape[0] + 1).reshape(-1, 1), re_brain_nodes))
    brain_tetra_no_tumour = np.zeros([brain_tetras_reconstructed.shape[0], 5], dtype=int)

    # Mapping of node coordinates to new indices
    re_brain_nodes_dict = {tuple(row): idx + 1 for idx, row in enumerate(re_brain_nodes)}

    # Assign new node IDs
    brain_nodes_pair = np.zeros([brain_nodes_new.shape[0], 2], dtype=int)
    brain_nodes_pair[:, 0] = np.arange(1, brain_nodes_new.shape[0] + 1)
    brain_nodes_pair[:, 1] = [re_brain_nodes_dict.get(tuple(row[1:]), 0) for row in brain_nodes_new]
    brain_tetras_reconstructed_no_idx = brain_tetras_reconstructed[:, 1:]
    brain_tetra_no_tumour[:, 0] = np.arange(1, brain_tetras_reconstructed.shape[0] + 1)
    brain_tetra_no_tumour[:, 1:] = np.vectorize(lambda x: brain_nodes_pair[x - 1, 1])(brain_tetras_reconstructed_no_idx)

    # Save reconstructed files and meshes
    np.savetxt('./3_reconstructed_idxs/brain_tetra_new_re.txt', brain_tetra_no_tumour, fmt="%d, %d, %d, %d, %d", newline='\n')
    np.savetxt("./3_reconstructed_idxs/brain_nodes_new_re.txt", brain_nodes_no_tumour, fmt="%d, %.8f, %.8f, %.8f", newline='\n')

    re_tetras = brain_tetra_no_tumour[:,1:]-1
    # Convert to .vtu format
    meshio.write_points_cells("./3_reconstructed_brain_models/re_brain.vtu", re_brain_nodes, [("tetra",re_tetras)])

    # Convert to .inp format
    re_brain_vtu = meshio.read("./3_reconstructed_brain_models/re_brain.vtu")
    meshio.write("./3_reconstructed_brain_models/re_brain.inp", re_brain_vtu)


    # Get tumor boundary coordinates to update the sharing nodes IDs
    sharing_nodes_coordinate_re = np.zeros((sharing_nodes_coordinate.shape[0], 5))
    sharing_nodes_coordinate_re[:, 4] = sharing_nodes_coordinate[:, 0]
    sharing_nodes_coordinate_re[:, 1:4] = sharing_nodes_coordinate[:, 1:4]
    coord_to_index = {tuple(np.round(row[1:], 8)): idx + 1 for idx, row in enumerate(brain_nodes_reconstructed)}

    # Assign new node IDs to sharing nodes based on coordinate matches
    for i, row in enumerate(sharing_nodes_coordinate[:, 1:4]):
        key = tuple(np.round(row, 8))
        sharing_nodes_coordinate_re[i, 0] = coord_to_index.get(key, 0)

    np.savetxt('./3_reconstructed_idxs/sharing_nodes_re.txt', sharing_nodes_coordinate_re, fmt = "%d, %.8f, %.8f, %.8f, %d", newline = '\n')


    end = time.time()
    print(f'\tModel reconstructed in: {round(end - start,2)}s')


if __name__ == '__main__':
    reconstruct_model_without_tumor()
