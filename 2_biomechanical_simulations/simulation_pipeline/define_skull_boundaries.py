#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:29:47 2021

@author: yue
"""

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import time
import meshio
import numpy as np


def main() -> None:
    """
    Processes both the original and reconstructed brain mesh volumes by combining them with the skull surface mesh. 
    Outputs integrated models with skull boundaries and contact node sets.
    """
    brain_volume = "./1_brain_meshes/brain_tetravolume.inp"
    reconstructed_brain_volume = "./3_reconstructed_brain_models/re_brain.vtu"
    skull = "./1_brain_meshes/brain_triangsurface.msh"

    print("\nDefining skull-brain interface and generating final brain mesh files...")
    combine_skull_brain(brain_volume, skull)
    combine_skull_brain(reconstructed_brain_volume, skull)


def combine_skull_brain(volume_file_path: str, surface_file_path: str) -> None:
    """
    Combines a brain volume mesh with a skull surface mesh into a single
    tetrahedral model. Appends skull surface triangles and contact node set representing the skull-brain interface.

    Args:
        volume_file_path (str): Path to the brain volume mesh (.inp or .vtu).
        surface_file_path (str): Path to the skull surface mesh (.msh).
    """
    start = time.time()

    # Check if dealing with the original or reconstructed mesh
    reconstruction = volume_file_path.endswith('re_brain.vtu')

    # Load meshes
    brain_volume = meshio.read(volume_file_path) # .inp file or .vtu file (for reconstructed file)
    skull = meshio.read(surface_file_path) # .msh file

    brain_nodes = brain_volume.points
    brain_tetra = brain_volume.cells_dict["tetra"] 

    skull_nodes = skull.points
    skull_tris = skull.cells_dict["triangle"]

    # Change normals from outwards to inwards
    skull_tris = skull_tris[:, [0, 2, 1]]

    brain_volume_nodes_length = brain_nodes.shape[0]

    # Merge nodes and scale
    final_nodes = np.vstack((brain_nodes, skull_nodes)) / 1000
    final_nodes2 = final_nodes.astype(object)

    # Save final reconstructed tetrahedral mesh with skull boundaries
    output_brain_path = "./4_final_brain_models/brain_final_re.inp" if reconstruction else "./4_final_brain_models/brain_final.inp"
    meshio.write_points_cells(output_brain_path, final_nodes2, [("tetra", brain_tetra)])

    # Offset skull triangle indices and format for output
    skull_tris_offset = skull_tris + brain_volume_nodes_length + 1
    element_ids = np.arange(1, skull_tris.shape[0] + 1) + brain_tetra.shape[0]
    skull_tris_file = np.column_stack((element_ids, skull_tris_offset))

    # Write triangle connectivity
    output_skull_path = "./4_skull_idxs/skull_re.txt" if reconstruction else "./4_skull_idxs/skull.txt"
    np.savetxt(output_skull_path, skull_tris_file, fmt='%d, %d, %d, %d', newline='\n')  

    # Load triangle text content
    with open(output_skull_path, "r") as file:
        content = file.read()

    # Identify brain nodes in contact with skull nodes 
    scaled_skull_nodes = skull_nodes / 1000
    final_node_set = {tuple(row) for row in final_nodes2}
    contact = [i + 1 for i, node in enumerate(scaled_skull_nodes) if tuple(node) in final_node_set]
    # Stringify for output file
    contact_str = ",".join(map(str, sorted(contact)))

    # Append surface elements and contact set to .inp file
    with open(output_brain_path, "a+") as file:
        file.write("*Element, type=S3, ELSET=skull\n")
        file.write(content)
        file.write("*Nset, nset=contact\n")
        file.write(contact_str)
    

    end = time.time()
    print(f'\tInterface defined in: {round(end - start, 2)}s')


if __name__ == '__main__':
    main()
