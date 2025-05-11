# Dataset Processing
This directory contains the code to process the datasets and construct a organized directory with the required inputs (images, segmentations, surface models) ready for the biomechanical simulation pipeline.

Finds the tumor segmentations and corresponding image volumes for each case of ReMIND, converts DICOM files to NIfTI and NRRD, predicts a brain segmentation using [SynthSeg](https://github.com/BBillot/SynthSeg), manipulates and edits tumor and brain segmentations to generate a brain surface model with [3D Slicer](https://www.slicer.org/), and saves it in an organized ouput structure.

## Installation
### Install [tensorflow](https://www.tensorflow.org/install), [CUDA](https://developer.nvidia.com/cuda-toolkit-archive), and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive):
This is required to run **SynthSeg 2.0 robust**. You will require a CUDA-enabled GPU with updated drivers, the CUDA Toolkit, and the cuDNN library. SynthSeg should be able to run with CPU-only, but that wasn't tested in this work.

The original SynthSeg code required CUDA 10.0 for Python 3.6 or CUDA 10.1 for Python 3.8, and cuDNN v7.6.5. However, this code was tested with Python 3.10, tensorflow 2.19.0, CUDA 12.6, and cuDNN v9.10.0. 

**If your GPU supports CUDA 10.0 or 10.1, follow the original instructions in the SynthSeg [repository](https://github.com/BBillot/SynthSeg) with a new virtual environment specifically to run this step of the work.**

Additional info for CUDA on [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) or [WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Set up
Edit the configuration files with the required paths.

For `1.1_remind/remind_config.json`:
```
{
    "python_path": "/path/to/the/python/executable",
    "slicer_path": "/path/to/the/slicer/executable",
    "remind_path": "/path/to/the/main/remind/dataset",
    "tumor_seg_path": "/path/to/the/remind/segmentations",
    "output_simulation_path": "/path/where/the/outputs/will/be/saved/to"
}
```

For `1.2_upenn/upenn_config.json`:
```
{
    "python_path": "/path/to/the/python/executable",
    "slicer_path": "/path/to/the/slicer/executable",
    "upenn_path": "/path/to/the/upenn/dataset",
    "tumor_segs_paths": 
    {
        "tumor_segs_manual" : "/path/to/the/upenn/manual/segmentations", 
        "tumor_segs_auto": "/path/to/the/upenn/automatic/segmentations"
    },
    "output_simulation_path": "/path/where/the/outputs/will/be/saved/to"
}
```

## Usage
For ReMIND:
```
cd 1_dataset_processing/1.1_remind/
python3 process_remind_data.py
```

For UPENN-GBM:
```
cd 1_dataset_processing/1.2_upenn/
python3 process_upenn_data.py
```