# Author: 
# Yu Yue
# University of Western Australia
# 2022

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import os
import re
import sys
import subprocess
import time
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from datetime import datetime
from natsort import natsorted


def main(gravity: str, mtled_path: str = "../MTLED/build/IMLS_EXP") -> None:
    """
    Sets up the simulation using a specified gravity vector,
    generates the corresponding brain forces using the ExplicitSim solver, and 
    calculates the external loadings at the tumor-brain interface based on the simulation output.

    Args:
        gravity (str): A string representing the gravity vector components, formatted as "x y z".
        mtled_path (str): Path to the ExplicitSim solver executable. Default is "../MTLED/build/IMLS_EXP".
    """
    final_brain_model = "./4_final_brain_models/brain_final.inp"
    integration_file = "./mtled_results/integration_points.txt"
    material_file = "./5_material_properties/material_properties.txt"

    print("Simulating tumor reaction forces...")
    print(f"\tGravity vector: {gravity}\n")
    generate_brain_forces(final_brain_model, integration_file, material_file, gravity, mtled_path)
    print("\nCalculating external loadings...")
    calculate_external_forces()


def generate_brain_forces(volume_file: str, integration_points_file: str, material_properties_file: str, gravity: str, mtled_path: str) -> None:
    """
    Constructs a configuration (.ini) file required by the ExplicitSim solver and runs it.
    Simulates forces induced by tumor presence and gravity using ExplicitSim based on the input brain mesh,
    integration points, and material properties.

    Parameters:
        volume_file (str): Path to the brain volume mesh (.inp).
        integration_points_file (str): Path to the integration points file (.txt).
        material_properties_file (str): Path to the file with material properties (density, shear modulus, alpha, Poisson ratio) (.txt).
        gravity (str): A string with the gravity vector to be applied during simulation.
        mtled_path (str): Path to the ExplicitSim solver executable.
    """
    start = time.time()

    # Creates the initialization file for ExplicitSim to simulate the forces applied by the presence of the tumor and gravity
    # These parameters resulted in the best results while simulation most cases in ReMIND
    # (for better results, they would have to be tuned for each case individually...)
    ini_file = "./mtled_results/brain_forces.ini"
    with open(ini_file, "w") as file:
        file.write("\n[Model]\n")
        file.write("MassScaling = 0\n")
        file.write(f"MeshFile = {volume_file}\n")

        file.write("\n[IntegrationOptions]\n")
        file.write("Adaptive = 0\n")
        file.write("AdaptiveEps = 0.1\n")
        file.write("AdaptiveLevel = 10\n")
        file.write("TetrahedronDivisions = 4\n")
        file.write("IntegPointsPerTetrahedron = 4\n")
        file.write(f"ReadFromFile = {integration_points_file}\n")

        file.write("\n[Material]\n")
        file.write("Type = ogden\n")
        file.write(f"ReadFromFile = {material_properties_file}\n")

        file.write("\n[ShapeFunction]\n")
        file.write("Type = mmls\n")
        file.write("BasisFunctionType = quadratic\n")
        file.write("UseExactDerivatives = 1\n")
        file.write("DilatationCoefficient = 1.7\n")

        file.write("\n[Contacts]\n")
        file.write("NodeSet = contact\n")
        file.write("Surface = skull\n")

        file.write("\n[Gravity]\n")
        file.write(f"gravity = {gravity}\n")

        file.write("\n[EBCIEM]\n")
        file.write("UseEBCIEM = 0\n")
        file.write("UseSimplifiedVersion = 0\n")

        file.write("\n[DynamicRelaxation]\n")
        file.write("LoadTime = 1.0\n")
        file.write("EquilibriumTime = 5.0\n")
        file.write("LoadConvRate = 0.99\n")
        file.write("AfterLoadConvRate = 0.99\n")
        file.write("StopUpdateConvRateStepsNum = 2000\n")
        file.write("ConvRateDeviation = 0.0001\n")
        file.write("ForceDispUpdateStepsNum = 200\n")
        file.write("StableConvRateStepsNum = 20\n")
        file.write("ConvRateStopDeviation = 0.002\n")
        file.write("StopConvRateError = 0.2\n")
        file.write("StopAbsError = 0.00001\n")
        file.write("StopStepsNum = 100\n")

        file.write("\n[MTLED]\n")
        file.write("SaveProgressSteps = 0\n")
        file.write("UsePredefinedStableTimeStep = 1\n")
        file.write("StableTimeStep = 0.0005\n")

        file.write("\n[Output]\n")
        file.write(f"FilePath = ./6_brain_forces/\n")
        file.write(f"FileName = brain_wtumor_forces.vtu\n")
        file.write(f"AnimationName = animation_brain_wtumor_forces.pvd\n")


    cmd = [mtled_path, ini_file]

    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    # Runs the process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    # Prints the outputs
    for line in iter(process.stdout.readline, ''):
        print(line, end="")
    process.stdout.close()
    process.wait()
    for line in iter(process.stderr.readline, ''):
        print(line, end="")
    process.stderr.close()


    end = time.time()
    print("\n\tTumor reaction forces generated in", round(end - start, 2), "seconds.")

def calculate_external_forces() -> None:
    """
    Calculates and saves the external forces applied by the tumor to the brain 
    from the mesh file output from the ExplicitSim simulation.
    """
    start = time.time()

    # Since the simulation can generate several intermediate brain volumes (determined by the SaveProgressSteps),
    # we have to get the last timestep to get the final forces
    last_vtu = natsorted([f for f in os.listdir("6_brain_forces") if f.endswith(".vtu")])[-1]

    # Read VTU file and get the array of simulated forces
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(f"./6_brain_forces/{last_vtu}") # .vtu file
    reader.Update()
    data = reader.GetOutput()
    force_val = vtk_to_numpy(data.GetPointData().GetArray(2))

    # Get indices of nodes in the brain/tumor interface
    with open('./2_tumor_idxs/sharing_nodes.txt','r') as file:
        tumour_idx = [node_id.strip() for line in file for node_id in line.split(',') if node_id.strip()]

    # Generate the external loading file with the simulated tumor-induced forces
    num_tumour_nodes = len(tumour_idx)
    ext_load = np.zeros((num_tumour_nodes, 7), dtype=object)
    for i, node_id_str in enumerate(tumour_idx):
        node_id = int(node_id_str)
        force = -force_val[node_id - 1]
        ext_load[i] = [node_id, 1, 1, 1, *force]
    
    # The indices of each external load line should be the indices of the reconstructed brain/tumor interface nodes
    nodes_id_convert = np.genfromtxt('./3_reconstructed_idxs/sharing_nodes_re.txt', delimiter = ',')
    ext_load[:, 0] = nodes_id_convert[:, 0]
    np.savetxt('./6_loadings/ext_load.txt', ext_load, fmt = "%d, %d, %d, %d, %.8f, %.8f, %.8f", newline='\n')


    end = time.time()
    print(f'\tExternal loadings calculated in: {round(end - start, 2)}s\n')
    
    # Print the forces at each axis and the total force magnitude
    force_components = ext_load[:, 4:7]
    sum_forces = np.sum(force_components, axis=0)
    total_force_magnitude = np.linalg.norm(sum_forces)
    
    print(f"\tSum of forces in x direction: {sum_forces[0]:.4f} N")
    print(f"\tSum of forces in y direction: {sum_forces[1]:.4f} N")
    print(f"\tSum of forces in z direction: {sum_forces[2]:.4f} N")
    print(f"\tTotal resultant force magnitude: {total_force_magnitude:.4f} N")


if __name__ == "__main__": #### to do argparse ###
    gravity = sys.argv[1]
    # Regex pattern to match three space-separated numbers, e.g., "0.34 -0.654, 0.1"
    pattern = r"^[-+]?\d*\.\d+|[-+]?\d+(\s[-+]?\d*\.\d+|[-+]?\d+){2}$"
    assert re.match(pattern, gravity), "The string must be in the format 'x y z' where x, y, and z are numbers (integers or floats)."
    main(gravity)
