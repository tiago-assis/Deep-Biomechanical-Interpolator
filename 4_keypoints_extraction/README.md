# Keypoint extraction
This directory contains the code to extract SIFT-Ranked features from images in batch.

## Installation

### Build 3DSIFT-Rank
```
cd 4_keypoints_extraction/3DSIFT-Rank/

mkdir build
cd build

cmake ..

make
```
*To speed up the process, one can use `make -j<N>` instead of `make` to build in parallel, where `<N>` is the number of CPU threads to use.*

<!---
## Set up
Edit the configuration file with the required path.
```
{
    "sift_path": "/path/to/3DSIFT-Rank/build/featExtract/featExtract"
}
```
-->

## Usage
**For this work, the script was run with and without the `-w` flag, thus generating both files:**
```
cd 4_keypoints_extraction/
python3 4_extract_sift_features.py /path/to/the/first/step/output/directory
```
```
python3 4_extract_sift_features.py -w /path/to/the/first/step/output/directory
```

---

Feature extraction can be done by running the `4_extract_sift_features.py` script with the following options:
```
python3 4_extract_sift_features.py <input_path> [OPTIONS]
```
- `<input_path>` Path to the input directory containing case folders with images in NIfTI format.

| Option              | Description                                                             |
|---------------------|-------------------------------------------------------------------------|
| `-w`, `--world`     | Output feature geometry in world coordinates (default is voxel units).  |

- The program will output a `.key` file with the same name as the input file, containing a list of features, with their coordinates, scale, orientation, and descriptor. When the `-w` flag is given, a `"_w"` suffix is added to the file name.