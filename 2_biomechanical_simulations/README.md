# Biomechanical Simulations
This directory contains the code to run the biomechanical framework in batch. 

Runs the pre-simulation sections of the pipeline for all patient cases in the specified input directory to generate the necessary inputs for the [ExplicitSim](https://bitbucket.org/explicitsim/explicitsim/src/master/) solver. Then, runs the simulation proper, predicting tumor reaction forces and the nodal displacements after tumor resection, outputting the initial and final/displaced integration point coordinates (in the LPS coordinate system).

##### This directory includes adapted and optimized code originally authored by the [Intelligent Systems for Medicine Laboratory](https://isml.ecm.uwa.edu.au/ISML/index.php?title=ISML_Main_Page), The University of Western Australia, which was used in the work described in the following main publications: [1](https://doi.org/10.1016/j.media.2019.06.004), [2](http://dx.doi.org/10.1002/cnm.3539), [3](https://doi.org/10.1016/j.compbiomed.2022.105271), [4](http://dx.doi.org/10.1007/s11548-023-02881-7).

## Installation
Based on the official [documentation](https://bitbucket.org/explicitsim/explicitsim/src/c6109a36474d539e27fefb0bef390d596d7aac51/INSTALL.md).

### Build ExplicitSim:
Install dependencies for ExplicitSim (working for Ubuntu 22.04.5):
```
sudo apt update && sudo apt install cmake \
  doxygen \
  graphviz \
  libboost-dev libboost-filesystem-dev libboost-thread-dev \
  libeigen3-dev \
  libtinyxml2-dev \
  libcgal-dev
```
Build:
```
cd 2_biomechanical_simulations/MTLED/

mkdir build
cd build

cmake  ..

make
make doc
make install
```
To speed up the process, one can use `make -j<N>` instead of `make` to build in parallel, where `<N>` is the number of CPU threads to use.

## Usage
Assuming the [first step](1_dataset_processing) was successfully concluded, the second step is simply run by providing the path of processed data generated in the first step as input to the following scripts that should be run sequentially:

*Note that each of these steps might take several hours or days to complete.*
```
cd 2_biomechanical_simulations/
python3 2.1_generate_solver_inputs.py /path/to/the/first/step/output/directory
```
```
python3 2.2_run_simulations.py /path/to/the/first/step/output/directory
```
<!---
###### The scripts inside the `2_biomechanical_simulations/simulation_pipeline/` [subdirectory](2_biomechanical_simulations/simulation_pipeline/) should, in theory, be able to run individually with the command line for a single case at a time, instead of running the main script which processes a full directory in batch. However, this hasn't been fully tested.
-->