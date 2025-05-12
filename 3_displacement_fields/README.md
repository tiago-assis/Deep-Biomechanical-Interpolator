# Displacement Fields
This directory contains the code to resample medical images to a new voxel spacing (an isotropic 1 mm<sup>3</sup> spacing was used in this work), calculate a transform between two sets of coordinates using the [ScatteredTransform](https://github.com/grandwork2/ScatteredTransform) CLI extension of [3D Slicer](https://www.slicer.org/), and the respective 4D displacement field tensor from this transform in batch.

## Set up
Edit the configuration file with the required path.
```
{
    "scatteredtransform_cli_path": "/path/to/the/scatteredtransform/extension/binary"
}
```

## Usage
Assuming the [first](1_dataset_processing) and [second](2_biomechanical_simulations) steps were successfully concluded, the third step is simply run by providing the path to the data directory being used throughout this work as input to the scripts.

### 1. Image resampling
At this point in the work, resampling of voxel spacing has been done to **ONLY** the processed **ReMIND dataset** (for both images and segmentations), as the UPENN-GBM dataset has isotropic 1 mm<sup>3</sup> voxel spacing for all images natively.

**For this work, the script was run like this:**
```
cd 3_displacement_fields/
python3 3.1_resample_spacing.py --remind --nrrd /path/to/the/first/step/output/directory 
```

---

The resampling of an image to a new voxel spacing can be done by running the `3.1_resample_isotropic_spacing.py` script with the following options:
```
python3 3.1_resample_spacing.py <input_path> [OPTIONS]
```
- `<input_path>` Path to the input directory containing case folders with medical images in NIfTI or NRRD formats.

| Option                          | Description                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------|
| `-s`, `--spacing` *X Y Z*       | Desired output spacing (default: `1.0 1.0 1.0`). Must be 3 positive floats or integers.  |
| `-i`, `--interpolator` *MODE*   | Interpolation mode: `linear`, `nearest`, or `bspline` (default: `linear`).               |
| `-v`, `--verbose`               | Enable verbose output.                                                                   |
| `--nifti`, `--nii`              | Only resample NIfTI (`.nii`, `.nii.gz`) images.                                          |
| `--nrrd`                        | Only resample NRRD (`.nrrd`) images.                                                     |
| `--remind`                      | Only resample the ReMIND cases.                                                          |

When no `--nrrd` or `--nifti` flags are provided, both imaging formats are resampled. If a segmentation file is detected ("seg" or "mask" in the file name), the interpolation mode will automatically be changed to "nearest". 

- The program will output a file with the same format and name as the input file, but with an added `"resampled_"` prefix. The original file is not overwritten.

---

### 2. Get extra imaging modalities
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.1.** For the **ReMIND dataset ONLY**, preoperative T1ce (T1_postcontrast) (or T2 when T1ce was not available) were rigidly registered to the image (now with a new spacing) used for the biomechanical modeling in previous steps (the resampled .nrrd image in each case folder) using the [BRAINSFit](https://doi.org/10.54294/hmb052) algorithm* present in the [General Registration (BRAINS)](https://slicer.readthedocs.io/en/latest/user_guide/modules/brainsfit.html) extension of 3D Slicer, and saved as NIfTI images to the same case folder.

###### \* Any other algorithm/tool could be used for the rigid coregistration step.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2.2.** For the **UPENN-GBM dataset**, the T1ce (T1GD) images were simply **copied** over from the dataset directory to the data directory of this work, as all imaging modalities are natively coregistered to the same template.


### 3. Computing transforms and generating the displacement fields
To compute the transform from a set of initial and final coordinates and extract the corresponding 4D displacement field tensor for each case in the data directory, simply run:
```
python3 3.2_get_displacement_field.py /path/to/the/first/step/output/directory
```
