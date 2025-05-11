# Author: 
# Yu Yue
# University of Western Australia
# 2022

# Edited by: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025


import subprocess
import time


def main() -> None:
    """
    Generates integration points for both the original and reconstructed brain volume mesh
    models using the ExplicitSim solver.
    """
    final_brain_model = "./4_final_brain_models/brain_final.inp"
    final_reconstructed_brain_model = "./4_final_brain_models/brain_final_re.inp"
    integration_file = "./mtled_outputs/integration_points.txt"
    integration_reconstructed_file = "./mtled_outputs/integration_points_re.txt"

    print("\nGenerating mesh integration points...\n")
    generate_integration_points(final_brain_model, integration_file)
    generate_integration_points(final_reconstructed_brain_model, integration_reconstructed_file)


def generate_integration_points(volume_file: str, integration_points_out_file: str) -> None:
    """
    Generates integration points for a given brain mesh volume file using the ExplicitSim solver.
    Creates a configuration `.ini` file with parameters required by the ExplicitSim executable to
    compute integration points for a mesh.

    Args:
        volume_file (str): Path to the input brain mesh volume file (.inp format).
        integration_points_out_file (str): Path where the output integration points will be saved.
    """
    start = time.time()

    # Creates the initialization file for MTLED to generate the integration points for the brain meshes
    with open("./mtled_outputs/integration_points.ini", "w") as file:
        file.write("\n[Model]\n")
        file.write("MassScaling = 0\n")
        file.write(f"MeshFile = {volume_file}\n")

        file.write("\n[IntegrationOptions]\n")
        file.write("Adaptive = 0\n")
        file.write("AdaptiveEps = 0.1\n")
        file.write("AdaptiveLevel = 10\n")
        file.write("TetrahedronDivisions = 4\n")
        file.write("IntegPointsPerTetrahedron = 4\n")
        file.write(f"SaveToFile = {integration_points_out_file}\n")

        file.write("\n[ShapeFunction]\n")
        file.write("Type = mmls\n")
        file.write("BasisFunctionType = quadratic\n")
        file.write("UseExactDerivatives = 1\n")
        file.write("DilatationCoefficient = 1.7\n")

        file.write("\n[Contacts]\n")
        file.write("NodeSet = contact\n")
        file.write("Surface = skull\n")

        file.write("\n[EBCIEM]\n")
        file.write("UseEBCIEM = 0\n")
        file.write("UseSimplifiedVersion = 0\n")

        file.write("\n[DynamicRelaxation]\n")
        file.write("LoadTime = 0.0001\n")
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


    cmd = ["../MTLED/build/IMLS_EXP", "./mtled_outputs/integration_points.ini"]

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
    print(f"\n\tIntegration points generated in: {round(end-start, 2)}s\n")


if __name__ == "__main__":
    main()
