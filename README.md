# Physics-Guided Deep Learning for Sparse Data-Driven Brain Shift Registration
###### **WORK IN PROGRESS...**

This repository contains the implementation of the work done throughout my MSc thesis. 

The project explores the integration of physics-based modeling as supervision for a deep learning model that addresses the challenges of brain shift registration by interpolating sparse intraoperative data. This work aims to improve the efficiency and applicability of intraoperative brain shift predictions, which are essential in image-guided neurosurgery. 

The repository includes all the necessary code and instructions to reproduce the results and experiments discussed in the thesis.

##### The [2_biomechanical_simulations](2_biomechanical_simulations/) directory of this repository includes adapted and optimized code originally authored by the [Intelligent Systems for Medicine Laboratory](https://isml.ecm.uwa.edu.au/ISML/index.php?title=ISML_Main_Page), The University of Western Australia, which was used in the work described in the following main publications: [1](https://doi.org/10.1016/j.media.2019.06.004), [2](http://dx.doi.org/10.1002/cnm.3539), [3](https://doi.org/10.1016/j.compbiomed.2022.105271), [4](http://dx.doi.org/10.1007/s11548-023-02881-7).

## Installation - MANDATORY READING
The code was mostly tested with [Python 3.10](https://www.python.org/downloads/release/python-31011/) in an [Ubuntu 22.04](https://releases.ubuntu.com/jammy/) system (either natively or through [WSL](https://learn.microsoft.com/en-us/windows/wsl/)), thus this is the recommended environment to use when running any code in this repository.

This project utilizes very different codebases, including [SynthSeg](https://github.com/BBillot/SynthSeg), [ExplicitSim](https://bitbucket.org/explicitsim/explicitsim/src/c6109a36474d539e27fefb0bef390d596d7aac51/INSTALL.md), and [3DSIFT-Rank](https://github.com/3dsift-rank/3DSIFT-Rank/tree/Appearance%2BGeometry). Thus, **any errors during installation should first be debugged by reading the main documentation and/or repositories of each of these software**.

### 1. Download the ReMIND and UPENN-GBM datasets:
- Follow the instructions and download the [ReMIND](https://doi.org/10.7937/3RAG-D070) dataset from [here](https://www.cancerimagingarchive.net/collection/remind/).
- Do the same for the [UPENN-GBM](https://doi.org/10.7937/TCIA.709X-DN49) dataset from [here](https://www.cancerimagingarchive.net/collection/upenn-gbm/).

ReMIND images and segmentations were downloaded in the DICOM and NRRD formats, respectively. UPENN-GBM images and segmentations were both downloaded in the NIFTI format.

This repository utilizes the ReMIND and UPENN-GBM data as is; thus, it is expected that the inner directory structures remain unaltered after downloading.

**Below you will find each dataset directory structure as is when downloaded into a `data/` directory and the subdirectories whose absolute paths will later be required as input for the first proper step of the work pipeline:**

### 1.1. ReMIND directory structures
Images and segmentations in DICOM format:
<pre>
<code>data/
|-- ReMIND/
|   |-- manifest-1746853315606/
|   |   |-- ReMIND/ # you will need the path to this directory!
|   |   |   |-- ReMIND-001/
|   |   |   |   |-- 12-25-1982-NA-Intraop-*/...
|   |   |   |   |-- 12-25-1982-NA-Preop-*/...
|   |   |   |-- ReMIND-002/
|   |   |   |   |-- 12-25-1982-NA-Intraop-*/...
|   |   |   |   |-- 12-25-1982-NA-Preop-*/...
|   |   |   |-- ...
|   |   |-- metadata.csv
</code></pre>

Segmentations in NRRD format:
<pre>
<code>data/
|-- ReMIND
|   |-- ReMIND_NRRD_Seg_Sep_2023/ # you will need the path to this directory!
|   |   |-- ReMIND-001/
|   |   |   |-- ReMIND-001-*.nrrd
|   |   |-- ReMIND-002/
|   |   |   |-- ReMIND-002-*.nrrd
|   |   |-- ...
</code></pre>

### 1.2. UPENN-GBM directory structure
<pre>
<code>data/
|-- UPENN-GBM/
|   |-- NIfTI-files/
|   |   |-- automated_segm/ # you will need the path to this directory!
|   |   |   |-- UPENN-GBM-00002_11_automated_approx_segm.nii.gz
|   |   |   |-- ...
|   |   |-- images_DSC/
|   |   |   |-- UPENN-GBM-00002_11/
|   |   |   |   |-- UPENN-GBM-00002_11_DSC.nii.gz
|   |   |   |   |-- ...
|   |   |-- images_DTI/
|   |   |   |-- images_DTI/
|   |   |   |   |-- UPENN-GBM-00002_11/
|   |   |   |   |   |-- UPENN-GBM-00002_11/
|   |   |   |   |   |   |-- UPENN-GBM-00002_11_DTI_AD.nii.gz
|   |   |   |   |   |   |-- ...
|   |   |-- images_segm/ # you will need the path to this directory!
|   |   |   |-- UPENN-GBM-00002_11_segm.nii.gz
|   |   |   |-- ...
|   |   |-- images_structural/
|   |   |   |-- UPENN-GBM-00002_11/
|   |   |   |   |-- UPENN-GBM-00002_11_FLAIR.nii.gz
|   |   |   |   |-- UPENN-GBM-00002_11_T1.nii.gz
|   |   |   |   |-- ...
|   |   |-- images_structural_unstripped/ # you will need the path to this directory!
|   |   |   |-- UPENN-GBM-00002_11/
|   |   |   |   |-- UPENN-GBM-00002_11_FLAIR_unstripped.nii.gz
|   |   |   |   |-- UPENN-GBM-00002_11_T1_unstripped.nii.gz
|   |   |   |   |-- ...
</code></pre>

### 2. Set up [3D Slicer](https://www.slicer.org/) with the required extension:
This work requires 3D Slicer (a free, open-source image computing platform) for some tasks, including the manipulation of segmentation and image volumes, the generation of surface models, and the computation of transforms by a multi-level BSpline interpolation algorithm ([ScatteredTransform](https://github.com/grandwork2/ScatteredTransform)).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.1.** Install dependencies:
```
sudo apt update && sudo apt install git build-essential \
  cmake cmake-curses-gui cmake-qt-gui \
  libqt5x11extras5-dev qtmultimedia5-dev libqt5svg5-dev qtwebengine5-dev libqt5xmlpatterns5-dev qttools5-dev qtbase5-private-dev qtbase5-dev qt5-qmake
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.2.** Build 3D Slicer from source by following the instructions from [here](https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/linux.html#checkout-slicer-source-files) on. This may take up to several hours.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.3.** Clone the ScatteredTransform extension repository:
```
git clone https://github.com/grandwork2/ScatteredTransform.git
``` 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.4.** Build the extension by following the instructions [here](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html#build-an-extension).

### 3. Create and activate a Python virtual environment ([venv](https://docs.python.org/3.10/library/venv.html), [conda](https://docs.conda.io/en/latest/), or other):
```
python3 -m venv .tiagoenv
source .tiagoenv/bin/activate
```

### 4. Clone this repository:
```
git clone https://github.com/tiago-assis/TiagoThesisWork.git
```

### 5. Install the required packages:
```
cd TiagoThesisWork/
pip3 install -r requirements.txt
```

### 6. Follow the `README.md` instructions inside each numerically-ordered folder.
This work follows a very specific pipeline that must be carefully set up and run sequentially. The `README.md` instructions present in the subdirectories of this repository should be read to achieve a high chance of success at reproducing the results. 

The main scripts run the code in batch, i.e., for every patient case inside a specified input directory.

These tutorials assume that all the previous steps have been followed, the correct environment is active, and the user is starting in the main directory of the repository.

## Contents
An overview of the contents of each folder is given below:

- [1_dataset_processing](1_dataset_processing/): contains the code to process the datasets and construct an organized directory with the required inputs ready for the biomechanical simulation pipeline.
  - [1.1_remind](1_dataset_processing/1.1_remind/): specific code for the ReMIND dataset
    - [process_remind_data.py](1_dataset_processing/1.1_remind/process_remind_data.py): finds the tumor segmentations and corresponding image volumes for each case of ReMIND, converts DICOM files to NIfTI and NRRD, predicts a brain segmentation using [SynthSeg](https://github.com/BBillot/SynthSeg), and saves it in an organized output structure.
    - [edit_remind_masks.py](1_dataset_processing/1.1_remind/edit_remind_masks.py): manipulates and edits tumor and brain segmentations to generate a brain surface model with [3D Slicer](https://www.slicer.org/). Automatically called while running `process_remind_data.py`.
    - [remind_config.json](1_dataset_processing/1.1_remind/remind_config.json): configuration file required to set up before running the scripts in this folder; contains the paths for executables, scripts, and input/output directories.
  - [1.2_upenn](1_dataset_processing/1.2_upenn/): specific code for the UPENN-GBM dataset
    - [process_upenn_data.py](1_dataset_processing/1.2_upenn/process_upenn_data.py): similar to the previous script for ReMIND, but tailored for the UPENN-GBM directory structure and image formats.
    - [edit_upenn_masks.py](1_dataset_processing/1.1_remind/edit_remind_masks.py): similar to the previous script for ReMIND, but tailored for the UPENN-GBM directory structure and image formats.
    - [remind_config.json](1_dataset_processing/1.1_remind/remind_config.json): similar to the previous script for ReMIND, but tailored for the UPENN-GBM directory structure and image formats.
- [2_biomechanical_simulations](2_biomechanical_simulations/)*: contains the code for the biomechanical framework and the scripts that run it in batch.
  - [2.1_generate_solver_inputs.py](2_biomechanical_simulations/2.1_generate_solver_inputs.py): runs the pre-simulation sections of the pipeline for all patient cases in the specified input directory to generate the necessary inputs for the [ExplicitSim](https://bitbucket.org/explicitsim/explicitsim/src/master/) solver.
  - [2.2_run_simulations.py](2_biomechanical_simulations/2.2_run_simulations.py): runs the simulation proper, predicting tumor reaction forces and the nodal displacements after tumor resection, outputting the initial and final/displaced integration point coordinates.
  - [MTLED](2_biomechanical_simulations/MTLED/): contains the code for the [ExplicitSim](https://bitbucket.org/explicitsim/explicitsim/src/master/) solver (which uses the [MTLED](https://doi.org/10.1002/cnm.1374) algorithm, thus the folder name) that was altered to include the effects of gravity in the simulations. The inner code of this directory will not be explained here, and it is automatically called while running `2.1_generate_solver_inputs.py` and `2.2_run_simulations.py`.
  - [simulation_pipeline](2_biomechanical_simulations/simulation_pipeline/): contains the code for the biomechanical framework. Inner scripts are automatically called while running `2.1_generate_solver_inputs.py` and `2.2_run_simulations.py`.
    - [generate_meshes.py](2_biomechanical_simulations/simulation_pipeline/generate_meshes.py): generates and optimizes 3D and surface meshes from an input brain surface model.
    - [find_gravity_vectors.py](2_biomechanical_simulations/simulation_pipeline/find_gravity_vectors.py): generates gravity vectors representing possible surgical entry directions based on the tumor center and the brain surface.
    - [identify_tumor.py](2_biomechanical_simulations/simulation_pipeline/identify_tumor.py): identifies the nodes in the brain mesh that belong to the tumor, outputting the mesh element indices of the tumor and the coordinates of the tumor boundary nodes.
    - [reconstruct_model.py](2_biomechanical_simulations/simulation_pipeline/reconstruct_model.py): reconstructs the brain mesh model after removing the tumor elements and nodes, simulating the brain state after resection.
    - [define_skull_boundaries.py](2_biomechanical_simulations/simulation_pipeline/define_skull_boundaries.py): processes the brain mesh volumes by combining them with the skull surface mesh, resulting in an integrated model with skull boundaries and contact node sets representing the skull-brain interface.
    - [generate_integration_points.py](2_biomechanical_simulations/simulation_pipeline/generate_integration_points.py): generates integration points for each element in the brain meshes using the [ExplicitSim](https://bitbucket.org/explicitsim/explicitsim/src/master/) solver.
    - [assign_material_properties.py](2_biomechanical_simulations/simulation_pipeline/assign_material_properties.py): runs the fuzzy classification of the brain image and the assignment of material properties to each integration point.
    - [compute_tumor_reaction_forces.py](2_biomechanical_simulations/simulation_pipeline/compute_tumor_reaction_forces.py): simulates the tumor reaction forces using the ExplicitSim solver and calculates the external loadings at the tumor-brain interface based on the simulation output.
    - [compute_nodal_displacements.py](2_biomechanical_simulations/simulation_pipeline/compute_nodal_displacements.py): simulates the nodal displacements after tumor resection and extracts the initial and final integration point coordinates.
    - Additional temporary subdirectories that are only useful while the code is running and will not be explained here.
- [3_displacement_fields](3_displacement_fields/): contains the code to resample images to an isotropic 1 mm<sup>3</sup> spacing, and calculates a transformation between two sets of coordinates using the [ScatteredTransform](https://github.com/grandwork2/ScatteredTransform) CLI extension of [3D Slicer](https://www.slicer.org/) and the respective displacement field.
  - [3.1_resample_spacing.py](3_displacement_fields/3.1_resample_spacing.py): resamples the spacing of a medical image to a specified new spacing.
  - [3.2_get_displacement_field.py](3_displacement_fields/3.2_get_displacement_field.py): calculates the transform from a set of initial and final coordinates, and generates the corresponding 4D displacement field tensor.
  - [config.json](3_displacement_fields/3.2_get_displacement_field.py): configuration file that specifies the path to the [ScatteredTransform](https://github.com/grandwork2/ScatteredTransform) script.
- [4_keypoints_extraction](4_keypoints_extraction/): contains the code to extract SIFT-Ranked features from NIfTI images.
  - [3DSIFT-Rank](4_keypoints_extraction/3DSIFT-Rank/): contains the code for the [3DSIFT-Rank](https://github.com/3dsift-rank/3DSIFT-Rank/tree/Appearance%2BGeometry) algorithm. The inner code of this directory will not be explained here, and it is automatically called while running `4_extract_sift_features.py`.
  - [4_extract_sift_features.py](4_keypoints_extraction/4_extract_sift_features.py): extracts SIFT-Ranked features from NIfTI medical images in a given directory.

## Contact
Contact at: tiago.assis@alunos.fc.ul.pt

LASIGE - Computer Science and Engineering Research Centre

Faculty of Sciences, University of Lisbon
