# Authors: 
# Yu Yue
# University of Western Australia
# 2022

# Edited and optimized by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# April 2025

import os
import subprocess
import time
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import datetime
from datetime import datetime as dt
import meshio
from natsort import natsorted
import sys
import glob
import shutil


def main(input_path, mtled_path: str = "./MTLED/build/IMLS_EXP"):
    """
    Runs the simulations with ExplicitSim for a set of cases in the input directory.

    Args:
        input_path (str): Path to the input directory containing cases.
        mtled_path (str): Path to the ExplicitSim binary (default is "./MTLED/build/IMLS_EXP").
    """
    cases = natsorted(os.listdir(input_path))
    
    bigbig_start = time.time()
    for case in cases:
        # Get the right format for the number of the case we're running  
        case_num = case.split("-")[-1]
        if case.startswith("UPENN-GBM"):
            case_num = case_num[:-3]

        cwd = os.getcwd()
        case_path = os.path.join(cwd, input_path, case)

        mtled_inputs = os.path.join(case_path, "mtled_inputs")
        assert os.path.exists(mtled_inputs), f"MTLED inputs folder does not exist for case '{case}'."

        brain = os.path.join(mtled_inputs, "brain_final.inp")
        int_pts = os.path.join(mtled_inputs, "integration_points.txt")
        mat_props = os.path.join(mtled_inputs, "material_properties.txt")
        sharing_nodes = os.path.join(mtled_inputs, "sharing_nodes.txt")
        sharing_nodes_re = os.path.join(mtled_inputs, "sharing_nodes_re.txt")
        int_pts_re = os.path.join(mtled_inputs, "integration_points_re.txt")
        mat_props_re = os.path.join(mtled_inputs, "material_properties_re.txt")
        brain_re = os.path.join(mtled_inputs, "brain_final_re.inp")

        # Get gravity vectors
        gravity = os.path.join(mtled_inputs, "gravity_vectors.txt")
        with open(gravity, "r") as gv:
            vectors = np.genfromtxt(gv, delimiter=" ", ndmin=2)
            n_vecs = vectors.shape[0]
        
        big_start = time.time()
        print(f"\nStarting simulations for case '{case}'...")
        # Loop through all gravity vectors and run the simulations for each of them
        
        for v in range(n_vecs):
            gravity_vector = vectors[v,:]
            gravity_vector = " ".join(map(str, gravity_vector))

            out = os.path.join(case_path, f"simulation{v+1}")
            loading_file = os.path.join(out, "ext_load.txt")

            # Skip if the simulation has already run
            if not os.path.exists(out):
                os.makedirs(out)
            elif os.path.exists(os.path.join(out, "final_coord.txt")):
                print(f"\tSimulation was already successful for case '{case}', simulation #{v+1}. Skipping.\n")
                continue
            elif os.path.exists(os.path.join(out, "errors.log")):
                print(f"\tSimulation had previous fails for case '{case}', simulation #{v+1}. Skipping.\n")
                continue        

            print(f"Running simulation #{v+1} with gravity vector: [{gravity_vector}]")
            start = time.time()
            # Compute tumor reaction forces and nodal displacements
            # If successful, each function returns True and generates the corresponding output files
            # If not, skip to then next simulation.
            if not os.path.exists(os.path.join(out, "brain_forces.ini")):
                check1 = compute_tumor_reaction_forces(brain, int_pts, mat_props, gravity_vector, sharing_nodes, sharing_nodes_re, loading_file, out, mtled_path)
                if not check1:
                    os.remove(os.path.join(out, "brain_forces.ini"))
                    print(f"\n\tSimulation for case #{case_num} failed. Skipping.")
                    continue
            if not os.path.exists(os.path.join(out, "nodal_displacements.ini")):
                check2 = compute_nodal_displacements(brain_re, int_pts_re, mat_props_re, gravity_vector, loading_file, out, mtled_path)
                if not check2:
                    os.remove(os.path.join(out, "nodal_displacements.ini"))
                    print(f"\n\tSimulation for case #{case_num} failed. Skipping.")
                    continue
            print(f"\nSimulation #{v+1} for case '{case}' completed in: {datetime.timedelta(seconds=time.time()-start)}\n")

            # Save all intermediary files except for the output coordinate .txt files (useful for debugging)
            os.makedirs(os.path.join(out, "extra_files"), exist_ok=True)
            for root, _, files in os.walk(out):
                for file in files:
                    if not file.endswith("coord.txt"):
                        shutil.move(os.path.join(root, file), os.path.join(out, "extra_files", file))

        print(f"Case '{case}' finished all runs in: {datetime.timedelta(seconds=time.time()-big_start)}")
    print(f"\nAll cases completed in: {datetime.timedelta(seconds=time.time()-bigbig_start)}")  

def compute_tumor_reaction_forces(brain, int_pts, mat_props, gravity, sharing_nodes, sharing_nodes_re, loading_file, out, mtled_path, \
                                  load_time="1.0", equilibrium_time="5.0", load_conv_rate="0.99", time_step="0.0003"):
    """
    Computes tumor-induced reaction forces and generates an external load file for the next step of the simulation.

    Args:
        brain (str): Path to the brain mesh file (.inp).
        int_pts (str): Path to the integration points file (.txt).
        mat_props (str): Path to the material properties file (.txt).
        gravity (np.ndarray): Gravity vector ([x,y,z]).
        sharing_nodes (str): Path to the file containing nodes on the tumor-brain interface (.txt).
        sharing_nodes_re (str): Path to the file containing nodes on the tumor-brain interface for the reconstructed brain model without tumor (.txt).
        loading_file (str): Path where the external loading forces will be saved (.txt).
        out (str): Path to the output directory for simulation results.
        mtled_path (str): Path to the ExplicitSim binary.
        load_time (str, optional): Duration of the loading application (default is "1.0").
        equilibrium_time (str, optional): Time for solution to reach equilibrium in dynamic relaxation (default is "5.0").
        load_conv_rate (str, optional): Convergence rate during loading application (default is "0.99").
        time_step (str, optional): Time step for simulation (default is "0.0003").

    Returns:
        bool: True if simulation was successful, False if any errors occurred.
    """
    ini_file = os.path.join(out, "brain_forces.ini")
    with open(ini_file, "w") as file:
        file.write("[Model]\n")
        file.write("MassScaling = 0\n")
        file.write(f"MeshFile = {brain}\n")

        file.write("\n[IntegrationOptions]\n")
        file.write("Adaptive = 0\n")
        file.write("AdaptiveEps = 0.1\n")
        file.write("AdaptiveLevel = 10\n")
        file.write("TetrahedronDivisions = 4\n")
        file.write("IntegPointsPerTetrahedron = 4\n")
        file.write(f"ReadFromFile = {int_pts}\n")

        file.write("\n[Material]\n")
        file.write("Type = ogden\n")
        file.write(f"ReadFromFile = {mat_props}\n")

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
        file.write(f"LoadTime = {load_time}\n")
        file.write(f"EquilibriumTime = {equilibrium_time}\n")
        file.write(f"LoadConvRate = {load_conv_rate}\n")
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
        file.write(f"StableTimeStep = {time_step}\n")

        file.write("\n[Output]\n")
        file.write(f"FilePath = {out}/\n")
        file.write("FileName = brain_w_tumor.vtu\n")
        file.write("AnimationName = animation_brain_w_tumor.pvd\n")

    cmd = [mtled_path, ini_file]

    start = time.time()
    print(f"Start time: {dt.now().strftime('%H:%M:%S')}")

    success_state = True
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        # Catches MTLED convergence errors
        if "reduce" in line.lower() or "increase" in line.lower():
            with open(os.path.join(out, "errors.log"), "a") as error_file:
                error_file.write(f"[{dt.now().strftime('%H:%M:%S')}] An error occurred during the simulation:'\n")
                error_file.write(str(line)+"\n")
            success_state = False
    process.stdout.close()
    process.wait()
    for line in iter(process.stderr.readline, ''):
        print(line, end="")
    process.stderr.close()

    if not success_state:
        return False
    else:
        last_vtu = natsorted(glob.glob(os.path.join(out, "*w_tumor*.vtu")))[-1]

        # Read VTU file and get the array of simulated forces
        reader = vtk.vtkXMLUnstructuredGridReader()
        brain_with_tumor = os.path.join(out, last_vtu)
        reader.SetFileName(brain_with_tumor) # .vtu file
        reader.Update()
        data = reader.GetOutput()
        force_val = vtk_to_numpy(data.GetPointData().GetArray(2))

        # Get indices of nodes in the brain/tumor interface
        with open(sharing_nodes,'r') as file:
            tumour_idx = [node_id.strip() for line in file for node_id in line.split(',') if node_id.strip()]

        # Generate the external loading file with the simulated tumor-induced forces
        num_tumour_nodes = len(tumour_idx)
        ext_load = np.zeros((num_tumour_nodes, 7), dtype=object)
        for i, node_id_str in enumerate(tumour_idx):
            node_id = int(node_id_str)
            force = -force_val[node_id - 1]
            ext_load[i] = [node_id, 1, 1, 1, *force]
        
        # The indices of each external load line should be the indices of the reconstructed brain/tumor interface nodes
        nodes_id_convert = np.genfromtxt(sharing_nodes_re, delimiter = ',')
        ext_load[:, 0] = nodes_id_convert[:, 0]
        np.savetxt(loading_file, ext_load, fmt = "%d, %d, %d, %d, %.8f, %.8f, %.8f", newline='\n')

        print(f"\n\tTumor reaction forces generated in: {datetime.timedelta(seconds=time.time()-start)}")

        return True 

def compute_nodal_displacements(brain_re, int_pts_re, mat_props_re, gravity, loading_file, out, mtled_path, \
                                load_time="1.0", equilibrium_time="5.0", load_conv_rate="0.99", time_step="0.0003"):
    """
    Compute nodal displacements for the brain mesh without the tumor (post-resection) and generate the displaced integration point coordinates.

    Args:
        brain_re (str): Path to the reconstructed brain mesh file (without tumor).
        int_pts_re (str): Path to the integration points file for the reconstructed brain.
        mat_props_re (str): Path to the material properties file for the reconstructed brain.
        gravity (np.ndarray): Gravity vector ([x,y,z]).
        loading_file (str): Path to the external loading file (.txt).
        out (str): Path to the output directory for simulation results.
        mtled_path (str): Path to the ExplicitSim binary.
        load_time (str, optional): Duration of the loading application (default is "1.0").
        equilibrium_time (str, optional): Time for solution to reach equilibrium in dynamic relaxation (default is "5.0").
        load_conv_rate (str, optional): Convergence rate during loading application (default is "0.99").
        time_step (str, optional): Time step for simulation (default is "0.0003").

    Returns:
        bool: True if simulation was successful, False if any errors occurred.
    """
    ini_file = os.path.join(out, "nodal_displacements.ini")
    with open(ini_file, "w") as file:
        file.write("[Model]\n")
        file.write("MassScaling = 0\n")
        file.write(f"MeshFile = {brain_re}\n")

        file.write("\n[IntegrationOptions]\n")
        file.write("Adaptive = 0\n")
        file.write("AdaptiveEps = 0.1\n")
        file.write("AdaptiveLevel = 10\n")
        file.write("TetrahedronDivisions = 4\n")
        file.write("IntegPointsPerTetrahedron = 4\n")
        file.write(f"ReadFromFile = {int_pts_re}\n")

        file.write("\n[Material]\n")
        file.write("Type = ogden\n")
        file.write(f"ReadFromFile = {mat_props_re}\n")

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
        file.write(f"LoadTime = {load_time}\n")
        file.write(f"EquilibriumTime = {equilibrium_time}\n")
        file.write(f"LoadConvRate = {load_conv_rate}\n")
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
        file.write(f"StableTimeStep = {time_step}\n")

        file.write("\n[Output]\n")
        file.write(f"FilePath = {out}/\n")
        file.write("FileName = brain_wo_tumor.vtu\n")
        file.write("AnimationName = animation_brain_wo_tumor.pvd\n")

    cmd = [mtled_path, ini_file]

    start = time.time()
    print(f"\nStart time: {dt.now().strftime('%H:%M:%S')}")

    success_state = True
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        # Catches MTLED convergence errors
        if "reduce" in line.lower() or "increase" in line.lower():
            with open(os.path.join(out, "errors.log"), "a") as error_file:
                error_file.write(f"[{dt.now().strftime('%H:%M:%S')}] An error occurred during the simulation:\n")
                error_file.write(str(line)+"\n")
            success_state = False
    process.stdout.close()
    process.wait()
    for line in iter(process.stderr.readline, ''):
        print(line, end="")
    process.stderr.close()

    if not success_state:
        return False
    else:
        last_vtu = natsorted(glob.glob(os.path.join(out, "*wo_tumor*.vtu")))[-1]

        # Read VTU file and get the array of nodal displacements
        reader = vtk.vtkXMLUnstructuredGridReader()
        brain_without_tumor = os.path.join(out, last_vtu)
        reader.SetFileName(brain_without_tumor)
        reader.Update()
        data = reader.GetOutput()
            
        disp = vtk_to_numpy(data.GetPointData().GetArray(0))

        # Read initial coordinates
        brain_without_tumour = meshio.read(brain_without_tumor)
        ini_coord = brain_without_tumour.points

        # Displaced coordinates are calculated by adding the displacement to the initial coordinates
        final_coord = ini_coord + disp

        # LPS to RAS and scale coordinates (it is necessary to rescale coordinates from .inp files)
        #transform = np.array([[-1, 0, 0],
        #                      [0, -1, 0],
        #                      [0,  0, 1]]) * 1000

        ## Keeping it in LPS format for now ##
        transform = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]) * 1000
        ini_coord_transformed = ini_coord @ transform.T
        final_coord_transformed = final_coord @ transform.T
        
        # Save coordinate files
        np.savetxt(os.path.join(out, "init_coord.txt"), ini_coord_transformed, fmt = "%.6f,%.6f,%.6f", newline='\n')
        np.savetxt(os.path.join(out, "final_coord.txt"), final_coord_transformed, fmt = "%.6f,%.6f,%.6f", newline='\n')

        print(f"\n\tNodal displacements generated in: {datetime.timedelta(seconds=time.time()-start)}")

        return True


if __name__ == "__main__":
    input_path = sys.argv[1]
    assert os.path.exists(input_path), "Input path does not exist."
    assert os.path.isdir(input_path), f"{input_path} is not a valid directory."
    assert len([dir for dir in os.listdir(input_path)]) > 0, "Input path is empty."

    main(input_path)
