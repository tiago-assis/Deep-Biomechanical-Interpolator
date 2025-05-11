# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import glob
import os
import shutil
from tqdm import tqdm
import psutil
import subprocess
import SimpleITK as sitk
import time
import datetime
import json
from typing import Optional, Tuple
from natsort import natsorted


def main() -> None:
    """
    Initializes the batch processor and runs a batch of brain scan cases for the ReMIND dataset.
    """
    batch_processor = ReMINDProcessor()
    print("Preparing ReMIND data for the biomechanical simulation pipeline...")
    start = time.time()
    batch_processor.run()
    tqdm.write(f"\nAll cases completed in {str(datetime.timedelta(seconds=time.time()-start))}.")

                
class ReMINDProcessor:
    def __init__(self) -> None:
        self.remind_path = ""
        self.tumor_seg_path = ""
        self.output_path = ""
        self.synthseg_predictor = SynthSegPredictor()

        self.parse_configs()

    def parse_configs(self) -> None:
        """
        Parses the configuration file to initialize the proper paths and segmentation directories.
        """
        with open("remind_config.json", "r") as file:
            config = json.load(file)
            self.remind_path = config["remind_path"]
            self.tumor_seg_path = config["tumor_seg_path"]
            self.output_path = config["output_simulation_path"]

    def get_segmentation(self, case: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Locates the appropriate preoperative tumor segmentation file for a given case.
        Searches the segmentation directory of ReMIND. Gives priority to segmentations that contain
        'T1_postcontrast' or 'T2_SPACE' in the filename. If none are found, returns the first
        match (if any). Also extracts and stores the imaging modality from the
        segmentation file for future use in file naming.

        Args:
            case (str): The identifier of the patient case to search within.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the path to the selected
            segmentation file and its inferred modality. If no valid segmentation is found,
            returns (None, None).
        """
        self.seg_modality = None
        seg_paths = glob.glob(os.path.join(self.tumor_seg_path, case, "*preop-SEG-tumor-*.nrrd"))
        # Locate the tumor segmentation, giving priority to T1_postcontrast scans and then T2_SPACE ...
        idx = 0
        for i, seg in enumerate(seg_paths):
            if ("T1_postcontrast" in os.path.split(seg)[1]) or ("T2_SPACE" in os.path.split(seg)[1]):
                idx = i
                break
        # ... otherwise, just use the whatever match was found (idx = 0).
        # If nothing is found (raising an IndexError), return none (i.e., the case has no tumor segmentation available)
        try:
            seg_path = seg_paths[idx]
        except IndexError:
            return None, None
        # Logic to extract a formatted modality name from the tumor segmentation for file naming purposes later
        self.seg_modality = seg_path.split("-")
        self.seg_modality = self.seg_modality[-1].replace(".nrrd", "") if len(self.seg_modality[-1]) > 6 else self.seg_modality[-2]
        return seg_path, self.seg_modality

    def dcm2nifti_nrrd(
        self, 
        dcm_input: str, 
        output: str, 
        modality: str, 
        case_num: str
    ) -> None:
        """
        Converts a DICOM folder into both NIfTI (.nii.gz) and NRRD (.nrrd) formats.

        Args:
            dcm_input (str): Path to the input DICOM folder.
            output (str): Path to the output directory where files will be saved.
            modality (str): Scan type (e.g., 'T1_postcontrast', 'T2_SPACE') used for file naming.
            case_num (str): Case number or identifier used for file naming.
        """
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_input)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Save the image in both NIfTI and NRRD formats (for SynthSeg and later for ExplicitSim)
        sitk.WriteImage(image, os.path.join(output, f"{modality}_{case_num}.nii.gz"), useCompression=True)
        sitk.WriteImage(image, os.path.join(output, f"{modality}_{case_num}.nrrd"), useCompression=True)

    def run(self) -> None:
        """
        Runs the full batch processing pipeline for converting ReMIND DICOM cases into processed
        NIfTI/NRRD volumes, tumor segmentations, predicted brain masks, and 3D surface models.

        For each case:
            - Locates the appropriate tumor segmentation file.
            - Converts the original DICOM brain series to the NIfTI and NRRD formats.
            - Copies the tumor segmentation into the output folder.
            - Runs SynthSeg for brain segmentation, if not already present.
            - Calls a custom script to generate a surface brain model and edited segmentations.
        Skips cases with missing or invalid segmentations, unprocessable DICOM data, or cases already processed. 
        """
        for case in tqdm(natsorted(os.listdir(self.remind_path))):
            if "ReMIND" not in case:
                continue
            # Gets the formatted case number for the ReMIND dataset
            case_num = case.split('-')[-1]

            output_path = os.path.join(self.output_path, case)
            output_surface_model = os.path.join(output_path, f"SurfaceModel_{case_num}.stl")

            # Create the output path or skip this case if the surface model was already generated
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            elif os.path.exists(output_surface_model):
                tqdm.write(f"Surface model for case '{case}' already exists. Skipping.")
                continue
            
            tqdm.write(f"\nStarting case '{case}':")

            tqdm.write("Extracting ReMIND brain volume and tumor segmentation...")
            # Gets the tumor segmentation path and modality for file naming
            # If none is found, skip the case
            seg, modality = self.get_segmentation(case)
            if seg is None:
                tqdm.write(f"Case '{case}' does not have a valid tumor segmentation. Skipping.")
                continue

            # DICOM to NIfTI/NRRD conversion
            dcm_input = glob.glob(os.path.join(self.remind_path, case, "*Preop*", f"*-{modality.replace('_','')}*"))[0]
            self.dcm2nifti_nrrd(dcm_input, output_path, modality, case_num)

            # Move tumor segmentation to the processed data folder
            tumor_seg_path = os.path.join(output_path, f"tumor_{case_num}.seg.nrrd")
            shutil.copy(seg, tumor_seg_path)

            brain_volume_input = glob.glob(os.path.join(output_path, "*.nii.gz"))[0]
            # SynthSeg segmentation output
            predicted_brain_mask = os.path.join(output_path, f"brain_mask_{case_num}.nii.gz")
            # If no brain segmentation was predicted with SynthSeg yet, run the whole pipeline
            # Otherwise, skip the segmentation prediction and just edit the brain segmentation and generate the surface model
            if not os.path.exists(predicted_brain_mask):
                tqdm.write("Predicting brain mask with SynthSeg...")
                self.synthseg_predictor.predict_brain_mask(brain_volume_input, predicted_brain_mask)
            else:
                tqdm.write(f"Predicted brain mask for case '{case}' already exists.")
            tqdm.write("Editing brain mask and generating corresponding surface model...")
            input_ref_vol = glob.glob(os.path.join(output_path, "*T*.nrrd"))[0]
            edited_brain_mask = os.path.join(output_path, f"brain_mask_{case_num}.seg.nrrd")
            self.synthseg_predictor.edit_masks(predicted_brain_mask, tumor_seg_path, input_ref_vol, output_surface_model, edited_brain_mask, tumor_seg_path)
            
            os.remove(predicted_brain_mask)
            
            tqdm.write(f"Case '{case}' processed.")


class SynthSegPredictor:
    def __init__(self) -> None:
        self.python_path = ""
        self.slicer_path = ""

        self.parse_configs()

        # Check if the SynthSeg 2.0 robust model is already downloaded
        # If not, download it from the UCL Sharepoint
        model_path = "../SynthSeg/models/synthseg_robust_2.0.h5"
        if not os.path.exists(model_path):
            print("Downloading SynthSeg 2.0 robust model...")
            os.system(f"curl 'https://liveuclac-my.sharepoint.com/personal/rmappmb_ucl_ac_uk/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Frmappmb%5Fucl%5Fac%5Fuk%2FDocuments%2Fsynthseg%20models%2Fsynthseg%5Frobust%5F2%2E0%2Eh5' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Referer: https://liveuclac-my.sharepoint.com/personal/rmappmb_ucl_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frmappmb%5Fucl%5Fac%5Fuk%2FDocuments%2Fsynthseg%20models%2Fsynthseg%5Frobust%5F2%2E0%2Eh5&parent=%2Fpersonal%2Frmappmb%5Fucl%5Fac%5Fuk%2FDocuments%2Fsynthseg%20models' -H 'DNT: 1' -H 'Sec-GPC: 1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjE0LDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2F0ZW5hbnRhbm9uIzQ2ZDRlYzMzLWMwNTMtNGZhOC05Y2UxLTBjYTdjOTA0NzY1YiwwIy5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhdGVuYW50YW5vbiM0NmQ0ZWMzMy1jMDUzLTRmYTgtOWNlMS0wY2E3YzkwNDc2NWIsMTMzOTEzNjczMjQwMDAwMDAwLDAsMTMzOTE0NTM0MjQ0ODUyMTE5LDAuMC4wLjAsMjU4LDQ2ZDRlYzMzLWMwNTMtNGZhOC05Y2UxLTBjYTdjOTA0NzY1YiwsLDQ0MTNlMWI4LTAyZWEtNDZkMS05OTU5LTE3ZDRiYzdiMGQxZSw0NDEzZTFiOC0wMmVhLTQ2ZDEtOTk1OS0xN2Q0YmM3YjBkMWUsclpBWkF6dndPa3FXVVJ3OVNiSVJlZywwLDAsMCwsLCwyNjUwNDY3NzQzOTk5OTk5OTk5LDAsLCwsLCwsMCwsMTg5OTgyLGQ3V19lbmZqWjMyVl9SdXI5aVJ3T1A1RThjOCx5UHJfcnJNd1diTVhIUTdLbWdSQ29pS2V1N3MsRys1TVNZVlkrOUhEOVlIYXY1OWw0QlZ1dlZHV2oxSlhMN2ZMOC83Y3I5OTY3K0FRbUVLTFRiU3VMSXd6d2FtSEs2bFRIMkp4d2U4L0kyQjg2QWVOWGhUMkJITzhUSnQ4a25yK1J0aUZIVDJaK2Zzbk5hcXMzMVlmQlMrbE5IRVZPY2dnaXBsdE51MURFVVhTTUQrYU5HbXJ0TzFTaWlwUE5NSk4wRFpQT0xkN0FBWjdkQzBobllXYzRkcFZteEdWZkZhakhueml4TTNsTC93eitaTHJkc200Y3RMTEo5VSs0ckU0KzdYY1h4enl4SUUzbFlsa3l1S1Jwbk43SXNVYmlPTmhDNXMwMjl6cnRDdEtyMjFhOFhSQXVsL2M1UXVnemk0dWFqMDBJSjJCOTlpVkU3aC95QWhYc1BYZENydjZqM2RIaDV6NWlMKzNzNVpFMGl0TDl3PT08L1NQPg==; FeatureOverrides_experiments=[]; MicrosoftApplicationsTelemetryDeviceId=48e7db04-4d0f-41e2-b23f-02d8414deef3; ai_session=a+wLTX/zefzc75jxd8/kub|1746893426527|1746893728171' --output {model_path}")

    def parse_configs(self) -> None:
        """
        Parses the configuration file to initialize the proper paths and segmentation directories.
        """
        with open("remind_config.json", "r") as file:
            config = json.load(file)
            self.python_path = config["python_path"]
            self.slicer_path = config["slicer_path"]

    def predict_brain_mask(self,
        brain_volume_input: str,
        predicted_brain_mask: str
    ) -> None:
        """
        Runs SynthSeg to predict a brain segmentation on the provided brain scan.

        Args:
            brain_volume_input (str): Path to the input brain scan (NIfTI format).
            predicted_brain_mask (str): Path to save the predicted brain segmentation (NIfTI format).
        """
        # Uses the "robust" mode of SynthSeg and all the available CPUs.
        threads = str(psutil.cpu_count())
        cmd = [self.python_path, "../SynthSeg/scripts/commands/SynthSeg_predict.py", "--i", brain_volume_input, "--o", predicted_brain_mask, "--threads", threads, "--robust"]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # If debugging, remove the stdout and stderr arguments.

        # Delete the brain volume in NIfTI, as it is no longer needed
        # We're going to use the NRRD format from now on
        os.unlink(brain_volume_input)

    def edit_masks(
        self,
        input_brain_mask: str,
        tumor_seg_path: str,
        input_ref_vol: str,
        output_model: str,
        output_brain_mask: str,
        output_tumor_seg_path: str
    ) -> None:
        """
        Invokes a custom 3D Slicer script to edit the predicted brain mask, turn the tumor segmentation into a proper labelmap volume, 
        and generate a surface brain model for simulations.

        Args:
            input_brain_mask (str): Path to the predicted brain segmentation.
            tumor_seg_path (str): Path to the tumor segmentation.
            input_ref_vol (str): Path to the original brain scan (reference image).
            output_model (str): Path to save the output surface model (STL format).
            output_brain_mask (str): Path to save the edited brain segmentation (NRRD format).
            output_tumor_seg_path (str): Path to save the tumor segmentation (NRRD format).
        """
        # Runs the python script on the 3D Slicer environment.
        # Requires to be able to use the 3D Slicer GUI, unfortunately
        # as also adding "--no-main-window" as an argument to Slicer did not allow the custom script to work.
        cmd = [self.slicer_path, "--no-splash", "--python-script", "./edit_remind_masks.py", input_brain_mask, tumor_seg_path, input_ref_vol, output_model, output_brain_mask, output_tumor_seg_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # If debugging, remove the stdout and stderr arguments.


if __name__ == "__main__":
    main()
