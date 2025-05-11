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
import subprocess
import time
import sys
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import meshio
import numpy as np
import shutil
from datetime import datetime
from natsort import natsorted


def main(displacement_out_path: str, gravity: str, mtled_path: str = "../MTLED/build/IMLS_EXP") -> None:
    """
    Simulates the nodal displacements after tumor resection and extracts the initial and final integration point coordinates.

    Args:
        displacement_out_path (str): Path where results will be saved.
        gravity (str): Gravity vector used in the simulation, formatted as "x y z".
        mtled_path (str): Path to the ExplicitSim solver executable. Default is "../MTLED/build/IMLS_EXP".
    """
    final_brain_model = "./4_final_brain_models/brain_final_re.inp"
    integration_file = "./mtled_results/integration_points_re.txt"
    material_file = "./5_material_properties/material_properties_re.txt"
    loading_file = "./6_loadings/ext_load.txt"

    results_path = os.path.join(displacement_out_path, f"results_{'_'.join(gravity.split(' '))}")
    print("Simulating nodal displacements after tumor resection...")
    print("\tGravity vector:", gravity)
    generate_displacement_forces(final_brain_model, integration_file, material_file, loading_file, gravity, mtled_path)
    print("\nCalculating displaced coordinates...")
    get_displaced_coordinates(results_path)
    print("\nSaving extra optional files...")
    save_extra_results(results_path)


def generate_displacement_forces(volume_file: str, integration_points_file: str, material_properties_file: str, loading_file: str, gravity: str, mtled_path: str) -> None:
    """
    Constructs a configuration (.ini) file required by the ExplicitSim solver and runs it.
    Simulates nodal displacements using ExplicitSim based on the reconstructed brain volume (without tumor),
     material properties, and the external loadings.

    Args:
        volume_file (str): Path to the brain volume (.inp) file.
        integration_points_file (str): File with predefined integration points for ExplicitSim (.txt).
        material_properties_file (str): File specifying material properties (.txt).
        loading_file (str): File with external force definitions (.txt).
        gravity (str): Gravity vector to be applied in the simulation.
        mtled_path (str): Path to the ExplicitSim solver executable.
    """
    start = time.time()

    # Creates the initialization file for MTLED to simulate the nodal displacements after tumor resection
    # These parameters resulted in the best results while simulation most cases in ReMIND
    # (for better results, they would have to be tuned for each case individually...)
    ini_file = "./mtled_results/displacement_forces.ini"
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

        file.write("\n[External]\n")
        file.write(f"ReadFromFile = {loading_file}\n")
        file.write("FileLoadCurve = smooth\n")

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
        file.write(f"FilePath = ./7_displacements/\n")
        file.write(f"FileName = brain_wotumor_forces.vtu\n")
        file.write(f"AnimationName = animation_brain_wotumor_forces.pvd\n")


    cmd = [mtled_path, ini_file]

    print(f"\nStart time: {datetime.now().strftime('%H:%M:%S')}")
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
    print(f"\n\tNodal displacements generated in: {round(end - start, 2)}s")


def get_displaced_coordinates(results_path: str) -> None:
    """
    Computes the displaced node coordinates of the brain after force simulation by extracting the displacement vectors,
    applying coordinate transformations (from LPS to RAS [deprecated], and unit conversion to mm),
    and outputs both initial and final node coordinates that can be used to interpolate a dense dispalcement field.

    Args:
        results_path (str): Directory where transformed coordinates will be saved.
    """
    start = time.time()

    # Since the simulation can generate several intermediate brain volumes (determined by the SaveProgressSteps),
    # we have to get the last timestep to get the final displacements
    last_vtu = natsorted([f for f in os.listdir("7_displacements") if f.endswith(".vtu")])[-1]

    # Read VTU file and get the array of nodal displacements
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(f"./7_displacements/{last_vtu}") # .vtu file
    reader.Update()
    data = reader.GetOutput()
        
    disp = vtk_to_numpy(data.GetPointData().GetArray(0))

    # Read initial coordinates
    brain_without_tumour = meshio.read(f"./7_displacements/{last_vtu}")
    ini_coord = brain_without_tumour.points

    # Displaced coordinates are calculated by adding the displacement to the initial coordinates
    final_coord = ini_coord + disp

    # LPS to RAS and scale coordinates (it isnecessary to rescale coordinates from .inp files)
    #transform = np.array([[-1, 0, 0],
    #                      [0, -1, 0],
    #                      [0,  0, 1]]) * 1000
    # Keeping it in LPS format for now
    transform = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]]) * 1000
    ini_coord_transformed = ini_coord @ transform.T
    final_coord_transformed = final_coord @ transform.T
    
    # Save coordinate files
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.savetxt(os.path.join(results_path, "init_coord.txt"), ini_coord_transformed, fmt = "%.6f,%.6f,%.6f", newline='\n')
    np.savetxt(os.path.join(results_path, "final_coord.txt"), final_coord_transformed, fmt = "%.6f,%.6f,%.6f", newline='\n')

    end = time.time()
    print(f"\n\tFinal coordinates calculated in: {round(end - start, 2)}s")

def save_extra_results(results_path: str) -> None:
    """
    Copies supporting files (e.g., meshes, forces, .ini configs) into the results folder. Mainly useful for visualization and debugging purposes.

    Args:
        results_path (str): Directory where extra files will be saved.
    """
    if not os.path.exists(os.path.join(results_path, "extra_files")):
        os.makedirs(os.path.join(results_path, "extra_files"))

    shutil.copy("./1_brain_meshes/brain_tetravolume.vtu", os.path.join(results_path, "extra_files", "brain_volume.vtu"))
    shutil.copy("./1_brain_meshes/brain_triangsurface.vtu", os.path.join(results_path, "extra_files", "brain_surface.vtu"))
    shutil.copy("./3_reconstructed_brain_models/re_brain.vtu", os.path.join(results_path, "extra_files", "brain_volume_wo_tumor.vtu"))
    shutil.copy("./5_memberships/brain_img_masked.nrrd", os.path.join(results_path, "extra_files", "brain_img_masked.nrrd"))
    last_vtu1 = natsorted([f for f in os.listdir("6_brain_forces") if f.endswith(".vtu")])[-1]
    last_vtu2 = natsorted([f for f in os.listdir("7_displacements") if f.endswith(".vtu")])[-1]
    shutil.copy(f"./6_brain_forces/{last_vtu1}", os.path.join(results_path, "extra_files", last_vtu1))
    shutil.copy(f"./7_displacements/{last_vtu2}", os.path.join(results_path, "extra_files", last_vtu2))
    shutil.copy(f"./mtled_results/brain_forces.ini", os.path.join(results_path, "extra_files", "brain_forces_mtled.ini"))
    shutil.copy(f"./mtled_results/displacement_forces.ini", os.path.join(results_path, "extra_files", "displacement_forces_mtled.ini"))


if __name__ == "__main__": #### to do argparse ###
    displacements_out_path, gravity = sys.argv[1], sys.argv[2]
    assert os.path.exists(displacements_out_path), "Output path does not exist."
    assert os.path.isdir(displacements_out_path), f"{displacements_out_path} is not a valid directory."
    # Regex pattern to match three space-separated numbers, e.g., "0.34 -0.654, 0.1"
    pattern = r"^[-+]?\d*\.\d+|[-+]?\d+(\s[-+]?\d*\.\d+|[-+]?\d+){2}$"
    assert re.match(pattern, gravity), "The string must be in the format 'x y z' where x, y, and z are numbers (integers or floats)."

    main(displacements_out_path, gravity)
