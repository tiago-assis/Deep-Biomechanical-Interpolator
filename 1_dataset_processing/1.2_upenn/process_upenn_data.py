# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import os
from tqdm import tqdm
import psutil
import subprocess
import SimpleITK as sitk
import time
import datetime
import json
from natsort import natsorted


def main() -> None:
    """
    Initializes the batch processor and runs a batch of brain scan cases for the UPENN-GBM dataset.
    """
    batch_processor = UPENNProcessor()
    print("Preparing UPENN-GBM data for the biomechanical simulation pipeline...")
    start = time.time()
    batch_processor.run()
    tqdm.write(f"\nAll cases completed in {str(datetime.timedelta(seconds=time.time()-start))}.")


class UPENNProcessor:
    def __init__(self) -> None:
        self.upenn_path = ""
        self.tumor_segs_paths = ""
        self.output_path = ""
        self.synthseg_predictor = SynthSegPredictor()

        self.parse_config()

    def parse_config(self) -> None:
        """
        Parses the configuration file to initialize the proper paths and segmentation directories.
        """
        with open("upenn_config.json", "r") as file:
            config = json.load(file)
            self.upenn_path = config["upenn_path"]
            self.tumor_segs_paths = list(config["tumor_segs_paths"].values())
            self.output_path = config["output_simulation_path"]

    def run(self) -> None:
        """
        Runs the full batch processing pipeline for converting UPENN-GBM cases into processed
        NIfTI/NRRD volumes, tumor segmentations, predicted brain masks, and 3D surface models.

        For each case:
            - Converts the brain image and tumor segmentation to the NRRD format.
            - Selects the best available tumor segmentation (manual preferred).
            - Runs SynthSeg for brain segmentation, if not already present.
            - Calls a custom script to generate a surface brain model and edited segmentations.
        Skips cases that are post-operative or already processed.
        """
        for case in tqdm(natsorted(os.listdir(self.upenn_path))):
            if "UPENN" not in case:
                continue
            # "_21" is the suffix for postop cases. We only care for the preop ones, so we're skipping these
            if case.endswith("_21"):
                continue

            # Gets the formatted case number for the UPENN-GBM dataset
            case_num = case.split('-')[-1][:-3] 

            input_path = os.path.join(self.upenn_path, case)
            output_path = os.path.join(self.output_path, case)
            output_surface_model = os.path.join(output_path, f"SurfaceModel_{case_num}.stl")

            # Create the output path or skip this case if the surface model was already generated
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            elif os.path.exists(output_surface_model):
                tqdm.write(f"Surface model for case '{case}' already exists. Skipping.")
                continue

            tqdm.write(f"\nStarting case '{case}':")

            # Using T1 scans to generate the brain segmentations with SynthSeg.
            # Any other scan type could be used, although SynthSeg does not predict the tumor segmentation
            # and the presence of the tumor might even disturb the network's prediction.
            # I thought T1 would be the least prone to these errors.
            # I used unstripped because it is useful to let SynthSeg predict the CSF between the brain and the skull
            # for simulation purposes.
            t1 = os.path.join(input_path, f"{case}_T1_unstripped.nii.gz")

            # Convert from NIfTI to NRRD, as the biomechanical simulation pipeline is implemented with NRRD files in mind
            # SynthSeg still requires NIfTi files, though
            brain_img = sitk.ReadImage(t1)
            brain_img_out_path = os.path.join(output_path, f"T1_{case_num}.nrrd")
            sitk.WriteImage(brain_img, brain_img_out_path, useCompression=True)

            # The UPENN-GBM dataset has automated tumor segmentations for all cases, and a few manual ones
            # When the manual segmentations are available, prefer those, otherwise use the automated ones
            tumor_seg1 = os.path.join(self.tumor_segs_paths[0], f"{case}_segm.nii.gz")
            tumor_seg2 = os.path.join(self.tumor_segs_paths[1], f"{case}_automated_approx_segm.nii.gz")
            tumor_seg = tumor_seg1 if os.path.exists(tumor_seg1) else tumor_seg2

            # NIfTI to NRRD again
            tumor_seg_path = os.path.join(output_path, f"tumor_{case_num}.seg.nrrd")
            tumor_mask = sitk.ReadImage(tumor_seg)
            sitk.WriteImage(tumor_mask, tumor_seg_path, useCompression=True)

            # SynthSeg segmentation output
            predicted_brain_mask_out_path = os.path.join(output_path, f"brain_mask_{case_num}.nii.gz")
            # If no brain segmentation was predicted with SynthSeg yet, run the whole pipeline
            # Otherwise, skip the segmentation prediction and just edit the brain segmentation and generate the surface model
            if not os.path.exists(predicted_brain_mask_out_path):
                tqdm.write("Predicting brain mask with SynthSeg...")
                self.synthseg_predictor.predict_brain_mask(t1, predicted_brain_mask_out_path)
            else:
                tqdm.write(f"Predicted brain mask for case #{case_num} already exists.")
            tqdm.write("Editing brain mask and generating corresponding surface model...")
            edited_brain_mask_out_path = os.path.join(output_path, f"brain_mask_{case_num}.seg.nrrd")
            self.synthseg_predictor.edit_masks(predicted_brain_mask_out_path, tumor_seg_path, t1, output_surface_model, edited_brain_mask_out_path, tumor_seg_path)
            
            os.remove(predicted_brain_mask_out_path)

            tqdm.write(f"Case #{case_num} processed.")


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
        with open("upenn_config.json", "r") as file:
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
        cmd = [self.slicer_path, "--no-splash", "--python-script", "./edit_upenn_masks.py", input_brain_mask, tumor_seg_path, input_ref_vol, output_model, output_brain_mask, output_tumor_seg_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # If debugging, remove the stdout and stderr arguments.


if __name__ == "__main__":
    main()
