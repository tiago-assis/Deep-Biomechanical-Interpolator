import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional
from monai.transforms import LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, \
    RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, \
    RandScaleIntensityFixedMeand, RandSimulateLowResolutiond, RandAdjustContrastd, RandFlipd, Compose, \
    ToTensord, OneOf, ResizeWithPadOrCropd, MapTransform, CenterSpatialCrop
from monai.data import NibabelReader, NrrdReader
import nibabel as nib
from model_pipeline.interpolators.interpolators import LinearInterpolation3d, ThinPlateSpline


class LoadRandDDFd(MapTransform):
    def __init__(self, keys: List[str], seed: int | None =None):
        """
        Args:
            keys (List[str]): List of keys in the dictionary to apply the transform to.
            seed (int or None): Random seed for reproducibility. Default is None.
        """
        super().__init__(keys)
        self.seed = seed

    def __call__(self, data: dict) -> dict:
        """
        Loads a random displacement field from a simulation directory specified by
        the path stored under the first key in the input dictionary.

        Args:
            data (dict): Dictionary containing at least the key 'gt_ddf'.
                         The value for this key should be a directory path containing
                         simulation .npz files with displacement fields.

        Returns:
            d (dict): Updated dictionary where the displacement field path
                  is replaced by a randomly chosen displacement field tensor loaded from one
                  of the simulation files. The displacement field tensor has shape
                  transposed as (3, D, H, W).
        """
        d = dict(data)
        gt_ddf = d[self.keys[0]]
        sims = glob.glob(os.path.join(gt_ddf, "simulation*", "*.npz"))
        rng = np.random.default_rng(seed=self.seed)
        rnd_idx = rng.integers(0, len(sims))
        rnd_sim = sims[rnd_idx]
        gt_ddf = np.load(rnd_sim, allow_pickle=True)['field'].transpose(0,3,2,1)
        gt_ddf = torch.from_numpy(gt_ddf)
        d[self.keys[0]] = gt_ddf
        del gt_ddf  
        return d
    
class InterpKptDispsd(MapTransform):
    def __init__(self, keys: List[str], 
                 interp: str = 'tps', 
                 min_kpts: int = 20, 
                 max_kpts: int = 30, 
                 seed: Optional[int] = None, 
                 device: str = 'cuda'):
        """
        Args:
            keys (List[str]): List of dictionary keys to apply the transform.
            interp (str): Interpolation method, either 'tps' (Thin Plate Spline) or 'linear'.
            min_kpts (int): Minimum number of keypoints to sample.
            max_kpts (int): Maximum number of keypoints to sample.
            seed (int or None): Random seed for reproducibility.
            device (str): Device for tensor computations ('cuda' or 'cpu').
        """
        super().__init__(keys)
        self.interp = interp
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.seed = seed
        self.device = device
    
    def __call__(self, data: dict) -> dict:
        """
        Samples a subset of keypoints, interpolates the displacement field on the grid
        using either TPS or linear interpolation, and returns updated data dictionary.

        Args:
            data (dict): Dictionary containing at least the keys 'kpts', 'tumor_seg', and 'gt_ddf'.
                Expected keys include:
                    - Keypoint file path: path to .txt file containing keypoints.
                    - Ground truth displacement field: tensor of shape (3, D, H, W).

        Returns:
            d (dict): Updated dictionary with a new displacement field under key 'init_ddf' interpolated from the sampled keypoints,
                and keypoints not used for interpolation under the key 'kpts'.
        """
        d = dict(data)
        kpts_path = d['kpts']
        #tumor_seg = d['tumor_seg']
        gt_ddf = d['gt_ddf']

        with torch.no_grad():
            gt_ddf = gt_ddf.to(self.device)

            size = gt_ddf.shape
            _, D, H, W = size

            if self.interp == 'linear':
                ddf_interp = LinearInterpolation3d((D,H,W)).to(self.device)
            elif self.interp == 'tps':
                ddf_interp = ThinPlateSpline((D,H,W)).to(self.device)

            kpts = np.genfromtxt(kpts_path, delimiter="\t", skip_header=6, dtype=np.float32)[:,:3]
            _, unique_idxs = np.unique(kpts, axis=0, return_index=True)
            sorted_unique_idxs = np.sort(unique_idxs)
            kpts = kpts[sorted_unique_idxs]
            
            #tumor_coords = np.argwhere(tumor_seg > 0)
            #tumor_center = np.mean(tumor_coords, axis=0)
            #distances = cdist(tumor_center.reshape(1,-1), kpts)
            weights = None
            #weights = np.exp(-distances[0]/15)
            #weights /= np.sum(weights)
            
            rng = np.random.default_rng(seed=self.seed)
            k = rng.integers(self.min_kpts, self.max_kpts+1)
            choices = rng.choice(range(kpts.shape[0]), p=weights, size=k, replace=False)
            kpts_choices = kpts[choices]
            kpts_choices = torch.from_numpy(kpts_choices)

            kpts_norm = torch.stack([
                (kpts_choices[:, 2] / (W - 1)) * 2 - 1,
                (kpts_choices[:, 1] / (H - 1)) * 2 - 1,
                (kpts_choices[:, 0] / (D - 1)) * 2 - 1
            ], dim=1).to(self.device)

            grid = kpts_norm.view(1, -1, 1, 1, 3)
            sampled_disps = F.grid_sample(gt_ddf.unsqueeze(0), grid, mode='bilinear', align_corners=True).squeeze().permute(1,0)
            sampled_disps = sampled_disps.to(self.device)
            
            init_ddf = ddf_interp(kpts_norm.unsqueeze(0), sampled_disps.unsqueeze(0)).squeeze(0)  # (3, D, H, W)

            d['init_ddf'] = init_ddf.cpu()

        # get keypoints not used during interpolation to later compute the TRE
        all_idxs = torch.arange(kpts.shape[0])
        excluded_idxs = torch.from_numpy(choices)
        included_idxs = all_idxs[~torch.isin(all_idxs, excluded_idxs)]
        d['kpts'] = kpts[included_idxs]

        return d

class LoadEdemaRegiond(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        """
        Load edema segmentation mask from a file or directory path, process the mask by
        merging edema labels 2 and 4 into a binary mask, and update the data dictionary.

        Args:
            data (dict): Input dictionary containing at least the key specified in `self.keys[0]`.
                         The corresponding value should be a path to a NIfTI file or directory.

        Returns:
            dict: Updated dictionary where the edema mask under `self.keys[0]` is a binary numpy array
                  indicating edema regions (True for edema, False otherwise).
        """
        d = dict(data)
        upenn_edema_seg = d[self.keys[0]]
        #if os.path.isdir(upenn_edema_seg):
        #    upenn_edema_seg = glob.glob(os.path.join(upenn_edema_seg, "*"))[0]
        mask = nib.load(upenn_edema_seg).get_fdata()
        mask[mask == 2] = 1
        mask[mask == 4] = 1
        mask = mask > 0
        d[self.keys[0]] = mask
        return d  

def get_training_transforms(img_keys: List[str], 
                            kpts_keys: List[str], 
                            disp_field_keys: List[str], 
                            seg_keys: List[str], 
                            upenn_edema_seg_keys: List[str], 
                            augment: bool = True,
                            interp: str = 'tps', 
                            min_kpts: int = 15, 
                            max_kpts: int = 20, 
                            size: Tuple[int, int, int] = (160,192,144),
                            kpts_sampling_seed: Optional[int] = None, 
                            ddf_sampling_seed: Optional[int] = None, 
                            device: str = 'cuda') -> Compose:
    """
    Construct a MONAI Compose object containing a sequence of training data preprocessing
    and augmentation transforms for keypoint-based image registration with simulated DDF fields.

    Args:
        img_keys (List[str]): Keys for image inputs.
        kpts_keys (List[str]): Keys for keypoint file paths.
        disp_field_keys (List[str]): Keys for displacement field path inputs.
        seg_keys (List[str]): Keys for segmentation inputs (excluding edema segmentation).
        upenn_edema_seg_keys (List[str]): Keys for edema segmentations.
        augment (bool, optional): If True, apply random data augmentations. Defaults to True.
        interp (str, optional): Interpolation method for displacement field initialization ("tps" or "linear"). Defaults to "tps".
        min_kpts (int, optional): Minimum number of keypoints to sample. Defaults to 15.
        max_kpts (int, optional): Maximum number of keypoints to sample. Defaults to 20.
        size (Tuple[int, int, int], optional): Final spatial size after resizing. Defaults to (160, 192, 144).
        kpts_sampling_seed (Optional[int], optional): Seed for keypoint sampling. Defaults to None.
        ddf_sampling_seed (Optional[int], optional): Seed for ground-truth displacement field sampling. Defaults to None.
        device (str, optional): Device to perform tensor computations on. Defaults to 'cuda'.

    Returns:
        transform (Compose): A composed transform pipeline to be used during training.
    """

    seg_keys_ = seg_keys.copy()
    seg_keys_ += upenn_edema_seg_keys
    all_keys = img_keys + disp_field_keys + seg_keys_

    load_imgs = LoadImaged(keys=img_keys + seg_keys, image_only=True, 
                           reader=[NibabelReader(), NrrdReader()])
    load_edema_region = LoadEdemaRegiond(keys=upenn_edema_seg_keys)
    load_rand_ddf = LoadRandDDFd(keys=disp_field_keys, seed=ddf_sampling_seed)
    interp_init_ddf = InterpKptDispsd(keys=kpts_keys + seg_keys + disp_field_keys,
                                      interp=interp, min_kpts=min_kpts, max_kpts=max_kpts,
                                      seed=kpts_sampling_seed, device=device)
    ensure_first_channel = EnsureChannelFirstd(keys=img_keys + seg_keys_, channel_dim='no_channel')
    norm = NormalizeIntensityd(keys=img_keys, nonzero=False)
    fixed_size = ResizeWithPadOrCropd(keys=all_keys, spatial_size=size, mode=('constant',) * len(all_keys))
    tensor = ToTensord(keys=all_keys, dtype=torch.float32)

    if augment:
        gauss_noise = RandGaussianNoised(keys=img_keys, std=0.1, prob=0.15)
        gauss_smooth = RandGaussianSmoothd(keys=img_keys,
                                                sigma_x=(0.5, 1.5),
                                                sigma_y=(0.5, 1.5),
                                                sigma_z=(0.5, 1.5),
                                                prob=0.1)
        
        scale_intensity = RandScaleIntensityd(keys=img_keys, factors=[-0.3, 0.3], prob=0.15)
        shift_intensity = RandScaleIntensityFixedMeand(keys=img_keys, factors=[-0.35, 0.5], preserve_range=True,
                                                       prob=0.15)
        sim_lowres = RandSimulateLowResolutiond(keys=img_keys, prob=0.125, zoom_range=(0.5, 1.0))
        adjust_contrast = OneOf(
            transforms=[
                RandAdjustContrastd(keys=img_keys, prob=0.15, gamma=(0.7, 1.5), 
                                    invert_image=False, retain_stats=True),
                RandAdjustContrastd(keys=img_keys, prob=0.15, gamma=(0.7, 1.5),
                                    invert_image=True, retain_stats=True)
            ],
            weights=[0.5, 0.5]
        )

        transform = Compose([
            load_imgs,
            load_edema_region,
            load_rand_ddf,
            interp_init_ddf,
            ensure_first_channel,
            fixed_size,
            norm,
            #affine,  # 1
            gauss_noise,  # 2
            gauss_smooth,  # 3
            scale_intensity,  # 4
            shift_intensity,  # 5
            sim_lowres,  # 6
            adjust_contrast,  # 7
            #mirror_x, mirror_y, mirror_z,  # 8
            tensor
        ])

    else:
        transform = Compose([
            load_imgs,
            load_edema_region,
            load_rand_ddf,
            interp_init_ddf,
            ensure_first_channel,
            fixed_size,
            norm,
            tensor
        ])

    return transform
