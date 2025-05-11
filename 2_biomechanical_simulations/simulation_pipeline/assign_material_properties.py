#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:01:24 2021

@author: yue
"""

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import time
import numpy as np
import mvloader.nrrd as mvnrrd
import nrrd
import skfuzzy as fuzz
import sys
import os
from multiprocessing import Pool, cpu_count


def main(brain_image_file_path: str) -> None:
    """
    Runs the fuzzy classification of the brain image and the assignment
    of material properties to each integration point.

    Args:
        brain_image_file_path (str): Path to the brain image file (.nrrd).
    """
    print("\nPerforming fuzzy classification of brain voxels...")
    fuzzy_classification(brain_image_file_path)
    integration_points_file_path = "./mtled_outputs/integration_points.txt"
    reconstructed_integration_points_file_path = "./mtled_outputs/integration_points_re.txt"
    material_properties_out_path = "./5_material_properties/material_properties.txt"
    recounstructed_material_properties_out_path = "./5_material_properties/material_properties_re.txt"
    print("\nAssigning material properties to the brain meshes...")
    assign_material_properties(integration_points_file_path, material_properties_out_path)
    assign_material_properties(reconstructed_integration_points_file_path, recounstructed_material_properties_out_path)


def fuzzy_classification(brain_image_file_path: str) -> None:
    """
    Performs fuzzy c-means clustering on the brain image to assign membership values to brain tissue
    types (e.g., ventricle, parenchyma, and tumor).

    Args:
        brain_image_file_path (str): Path to the brain image file (.nrrd).
    """
    start = time.time()

    # Read input files
    brain_mask, b_header = nrrd.read("./5_memberships/brain_mask_temp.nrrd")
    brain_img, _ = nrrd.read(brain_image_file_path)
    tumor_mask, t_header = nrrd.read("./2_tumor_idxs/tumor_mask_temp.seg.nrrd")

    # Clean up temporary files
    os.remove("./5_memberships/brain_mask_temp.nrrd")
    os.remove("./2_tumor_idxs/tumor_mask_temp.seg.nrrd")

    # Save a stripped version of the brain image (masked by the brain segmentation)
    brain_img_masked = np.where(brain_mask, brain_img, 0)
    nrrd.write("./5_memberships/brain_img_masked.nrrd", brain_img_masked, b_header)

    # Extract voxels for clustering
    voxel_intensities = brain_img[brain_mask > 0].reshape(1, -1)

    # Fuzzy C-means clustering
    ncenters = 2
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(voxel_intensities, ncenters, 2, error=0.005, maxiter=10000, init=None)

    # Allocate and assign memberships
    membership_1 = np.zeros_like(brain_mask, dtype=u.dtype)
    membership_1[brain_mask > 0] = u[0]
    membership_1[tumor_mask > 0] = 0
    nrrd.write("./5_memberships/membership_1.nrrd", membership_1, b_header) # ventricle?

    membership_2 = np.zeros_like(brain_mask, dtype=u.dtype)
    membership_2[brain_mask > 0] = u[1]
    membership_2[tumor_mask > 0] = 0
    nrrd.write("./5_memberships/membership_2.nrrd", membership_2, b_header) # parenchyma?

    tumor_membership = np.where(tumor_mask > 0, 1, 0)
    nrrd.write("./5_memberships/tumor_membership.nrrd", tumor_membership, t_header)
    

    end = time.time()
    print(f"\tFuzzy classification done in: {round(end - start, 2)}s")

def assign_material_properties(integration_points_file_path: str, material_properties_out_file: str) -> None:
    """
    Assigns material properties (shear modulus, alpha, Poisson ratio, density) to a set of integration points by considering
    the surrounding voxels' memberships (ventricle, parenchyma, tumor) from the fuzzy classification.

    Args:
        integration_points_file_path (str): Path to the file containing the integration points (.txt).
        material_properties_out_file (str): Path to the output file where the computed material properties will be saved (.txt).
    """
    start = time.time()

    # Reads brain memberships for material assignment
    tumour_membership, _ = nrrd.read("./5_memberships/tumor_membership.nrrd")
    m1 = mvnrrd.open_image("./5_memberships/membership_1.nrrd", verbose=False)
    m2 = mvnrrd.open_image("./5_memberships/membership_2.nrrd", verbose=False)

    # Determines which membership corresponds to the ventricle or parenchyma (they may not always be membership_1 or membership_2...)
    # From manual visualization, the parenchyma membership has brighter voxels, 
    # thus the result of the sum of its voxels should be bigger than that of the ventricle membership
    if np.sum(m1.src_data) > np.sum(m2.src_data):
        ventricle_membership = m2.src_data
        voxels2world = m2.src_transformation
    else:
        ventricle_membership = m1.src_data
        voxels2world = m1.src_transformation

    # Inverse transformation from world space to voxel space
    world2voxels = np.linalg.inv(voxels2world)
 
    # Read integration points file
    ips = np.genfromtxt(integration_points_file_path, delimiter = ',') # .txt file

    # Perform material property assignment for chunks of integration points using multiprocessing tools
    num_workers = cpu_count()
    chunk_size = int(np.ceil(len(ips) / num_workers))
    chunks = [ips[i:i+chunk_size] for i in range(0, len(ips), chunk_size)]

    common_args = [(chunk, tumour_membership, ventricle_membership, world2voxels) for chunk in chunks]

    with Pool(processes=num_workers) as pool:
        material_props = pool.map(process_chunk, common_args)

    all_material_props = np.vstack(material_props)

    # Save material properties
    np.savetxt(material_properties_out_file, all_material_props, fmt = "%.1f, %.8f, %.8f, %.8f", newline='\n')
    

    end = time.time()
    print(f"\tMaterial properties assigned in: {round(end - start, 2)}s")

def process_chunk(args):
    ips_chunk, tumour_membership, ventricle_membership, world2voxels = args
    
    # Homogeneous coordinate transformation
    homogeneous = lambda x: np.append(x, 1)

    # Material constants
    parenchyma_SM = 842
    tumour_SM = parenchyma_SM * 3
    csf_SM = 4.54

    D_parenchyma = 4.78e-05
    D_tumour = 1.59e-5
    D_ventricle = 0.48058 # this value considers the CSF as a compressible fluid
    # D_ventricle = 0.008869704047541623 # this value considers the CSF as an incompressible fluid

    parenchyma_alpha = -4.7
    tumour_alpha = -4.7
    csf_alpha = 2

    density = 1000

    material_props = np.zeros((len(ips_chunk), 4))
    material_props[:, 0] = density

    for idx, ip in enumerate(ips_chunk):
        # Convert the coordinate metric from the .inp file to the world space metric (mm)
        x = 1000 * np.array(ip[:3])
        # Convert the coordinates from world space to voxel space
        iv, jv, kv = (world2voxels[:3] @ homogeneous(x)).astype(int)
        # Loop over neighboring voxels around the integration point to aggregate the material properties
        sm, alpha, D = 0, 0, 0
        for i in range(iv - 1, iv + 1):
            for j in range(jv - 1, jv + 1):
                for k in range(kv - 1, kv + 1):
                    tumor = tumour_membership[i, j, k]
                    ventricle = ventricle_membership[i, j, k]
                    # If the ventricle membership is below 0.1, set it to 0
                    if ventricle < 0.1: ventricle = 0
                    # Compute the fraction of the voxel that corresponds to parenchyma (1 - tumor - ventricle)
                    parenchyma = 1 - tumor - ventricle

                    sm += parenchyma_SM * parenchyma + tumour_SM * tumor + csf_SM * ventricle
                    alpha += parenchyma_alpha * parenchyma + tumour_alpha * tumor + csf_alpha * ventricle
                    D += D_parenchyma * parenchyma + D_tumour * tumor + D_ventricle * ventricle
        
        # Average the accumulated material properties   
        material_props[idx, 1:] = np.array([sm, alpha, D]) / 8

    return material_props


if __name__ == "__main__":
    main(sys.argv[1])
