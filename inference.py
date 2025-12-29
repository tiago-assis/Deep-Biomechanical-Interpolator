import argparse
from typing import Tuple
import os
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from monai.transforms import NormalizeIntensity, ResizeWithPadOrCrop, DivisiblePad
from model_pipeline.interpolators.interpolators import ThinPlateSpline, LinearInterpolation3d
from model_pipeline.networks.unet3d.model import ResidualUNetSE3D

# TO DO: more testing

def interpolate_kpts(kpts_disps: str, interp_mode: str, shape: Tuple[int, int, int], device: str = 'cpu') -> torch.Tensor:
    D, H, W = shape
    kpts_disps = np.genfromtxt(kpts_disps, delimiter=",", dtype=np.float32)

    assert kpts_disps.shape[1] == 6, "Keypoint displacements file must have 6 columns: x, y, z coordinates and disp_x, disp_y, disp_z displacements."
    assert kpts_disps.shape[0] > 0, "No keypoints found in the provided file."

    kpts = torch.from_numpy(kpts_disps[:, :3])
    disps = torch.from_numpy(kpts_disps[:, 3:]).to(device)

    if interp_mode == 'tps':
        interp = ThinPlateSpline(shape).to(device)
    else:
        interp = LinearInterpolation3d(shape).to(device)

    kpts_norm = torch.stack([
        (kpts[:, 2] / (W - 1)) * 2 - 1,
        (kpts[:, 1] / (H - 1)) * 2 - 1,
        (kpts[:, 0] / (D - 1)) * 2 - 1
    ], dim=1).to(device)

    init_ddf = interp(kpts_norm.unsqueeze(
        0), disps.unsqueeze(0))  # (1, 3, D, H, W)

    return init_ddf


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deep biomechanical interpolator with optional initial interpolation.")

    parser.add_argument('-p', '--preop_scan', type=str, required=True,
                        help='Path to the preoperative scan (.nii or .nii.gz).')
    parser.add_argument('-i', '--init_disp', type=str, default=None,
                        help='Path to the initial displacement field (.h5). If not provided, will interpolate from the provided keypoints.')
    parser.add_argument('-k', '--kpt_disps', type=str, default=None,
                        help='Path to the keypoint displacements file (.csv or .txt).')
    parser.add_argument('-m', '--interp_mode', type=str, choices=[
                        'tps', 'linear'], default='linear', help='Interpolation mode (tps or linear).')
    parser.add_argument('-d', '--device', type=str, choices=[
                        'cuda', 'cpu'], default='cuda', help='Device to run the model on (cuda or cpu).')
    parser.add_argument('-o', '--output', type=str, default=".",
                        help='Directory to save the output displacement field.')
    parser.add_argument('-f', '--output_fmt', type=str, choices=[
                        'h5', 'npz'], default='h5', help='Output format for the displacement field (.h5 SimpleITK transform or .npz numpy array).')
    parser.add_argument('-w', '--weights', type=str, default='./checkpoints/res-unet-se_mixedinterp_32_200_5e-4.pt',
                        help='Path to the model weights. Please download the latest weights from: https://github.com/tiago-assis/Deep-Biomechanical-Interpolator/tree/main/checkpoints')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert args.preop_scan.endswith('.nii.gz') or args.preop_scan.endswith(
        '.nii'), "Preoperative scan must be a NIfTI file."
    assert args.init_disp is not None or args.kpt_disps is not None, "Either an *initial displacement field* or a *list of displacements at localized keypoint coordinates* must be provided."
    assert args.init_disp is None or args.init_disp.endswith(('.h5', '.hdf5', '.npz')), "Initial displacement field must be a HDF or NPZ file."
    assert args.kpt_disps is None or args.kpt_disps.endswith(('.csv', 'txt')), "Keypoint displacements must be a CSV text file."
    assert os.path.isdir(args.output), "Output path must be a valid directory."
    assert args.output_fmt in ['h5', 'npz'], "Output format must be either '.h5' or '.npz'."
    assert os.path.exists(args.weights) and os.path.isfile(args.weights), "Path to the model weights file must be valid. Please download it from: https://github.com/tiago-assis/Deep-Biomechanical-Interpolator/tree/main/checkpoints"

    checkpoint = torch.load(args.weights, map_location=args.device)

    preop_scan = sitk.ReadImage(args.preop_scan)
    preop_scan_arr = sitk.GetArrayFromImage(preop_scan) # (D_, H_, W_)

    preop_scan_arr = torch.tensor(preop_scan_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D_, H_, W_)
    preop_scan_arr = DivisiblePad(k=16, value=0)(preop_scan_arr)  # pad to be divisible by 2**4 = 16 // (1, 1, D, H, W)
    preop_scan_arr = NormalizeIntensity()(preop_scan_arr) # standardize
    preop_scan_arr = preop_scan_arr.to(args.device)


    if args.init_disp is not None:
        if args.init_disp.endswith('.h5') or args.init_disp.endswith('.hdf5'):
            transform = sitk.ReadTransform(args.init_disp)
            init_ddf = sitk.TransformToDisplacementField(transform,
                                                        sitk.sitkVectorFloat64,
                                                        preop_scan.GetSize(),
                                                        preop_scan.GetOrigin(),
                                                        preop_scan.GetSpacing(),
                                                        preop_scan.GetDirection()
                                                        )
            init_ddf = sitk.GetArrayFromImage(init_ddf).astype(np.float32)
        else:
            init_ddf = np.load(args.init_disp)
            npz_keys = list(init_ddf.keys())
            if len(npz_keys) != 1:
                raise ValueError("NPZ file must contain exactly one array representing the displacement field.")
            init_ddf = np.load(args.init_disp)[npz_keys[0]].astype(np.float32)
    else:
        init_ddf = interpolate_kpts(args.kpt_disps, interp_mode=args.interp_mode, shape=preop_scan_arr.shape[2:], device=args.device).squeeze(0) # (3, D_, H_, W_)
    

    if init_ddf.shape[0] == 3:
        pass
    elif init_ddf.shape[-1] == 3:
        init_ddf = np.transpose(init_ddf, (3, 0, 1, 2))  # (3, D_, H_, W_)
    else:
        raise ValueError("Initial displacement field has incorrect shape. (3, D, H, W) or (D, H, W, 3) expected.")
    
    
    init_ddf = torch.tensor(init_ddf, dtype=torch.float32).unsqueeze(0)  # (1, 3, D_, H_, W_) or already (1, 3, D, H, W) if interpolated from keypoints
    init_ddf = DivisiblePad(k=16, value=0)(init_ddf)  # (1, 3, D, H, W)
    init_ddf = torch.where(preop_scan_arr > torch.min(preop_scan_arr), init_ddf, 0) # zero out displacements in background
    init_ddf = init_ddf.to(args.device)

    model = ResidualUNetSE3D(
        in_channels=4,
        out_channels=3,
        final_sigmoid=False,
        f_maps=32,
        layer_order="cil",
        num_levels=4,
        is_segmentation=False,
        predict_residual=True,
        se_module="scse"
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    input = torch.cat([init_ddf, preop_scan_arr], dim=1) # (1, 4, D, H, W)
    
    corrected_ddf = model(input).squeeze(0)  # (3, D, H, W)

    corrected_ddf_sitk = corrected_ddf.detach().cpu().numpy().transpose(
        1, 2, 3, 0).astype(np.float64)  # (D, H, W, 3)
    
    if args.output_fmt == 'npz':
        np.savez_compressed(os.path.join(args.output, "corrected_disp_field.npz"), field=corrected_ddf_sitk)
    else:
        corrected_ddf_sitk = sitk.GetImageFromArray(
            corrected_ddf_sitk, isVector=True)
        corrected_ddf_sitk.SetOrigin(preop_scan.GetOrigin())
        corrected_ddf_sitk.SetSpacing(preop_scan.GetSpacing())
        corrected_ddf_sitk.SetDirection(preop_scan.GetDirection())
        corrected_transform = sitk.DisplacementFieldTransform(corrected_ddf_sitk)
        sitk.WriteTransform(corrected_transform, os.path.join(
            args.output, "corrected_disp_field.h5"))
