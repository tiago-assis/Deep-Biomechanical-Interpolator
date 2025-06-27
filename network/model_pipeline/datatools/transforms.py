import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from monai.transforms import LoadImaged, EnsureChannelFirstd, CropForegroundd, NormalizeIntensityd, \
    RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, \
    RandScaleIntensityFixedMeand, RandSimulateLowResolutiond, RandAdjustContrastd, RandFlipd, Compose, \
    DeleteItemsd, ToTensord, OneOf, ResizeWithPadOrCropd, MapTransform, CenterSpatialCrop
from monai.data import NibabelReader, NrrdReader, ITKReader
import nibabel as nib
from model_pipeline.interpolators.interpolators import LinearInterpolation3d, ThinPlateSpline


class LoadRandDDFd(MapTransform):
    def __init__(self, keys, seed=None):
        super().__init__(keys)
        self.seed = seed

    def __call__(self, data):
        d = dict(data)
        gt_ddf = d[self.keys[0]]
        sims = glob.glob(os.path.join(gt_ddf, "simulation*", "*.npz"))
        rng = np.random.default_rng(seed=self.seed)
        rnd_idx = rng.integers(0, len(sims))
        #rnd_idx = np.random.randint(0, len(sims))
        rnd_sim = sims[rnd_idx]
        gt_ddf = np.load(rnd_sim, allow_pickle=True)['field'].transpose(0,3,2,1)
        gt_ddf = torch.from_numpy(gt_ddf)
        d[self.keys[0]] = gt_ddf
        del gt_ddf  
        return d
    
class InterpKptDispsd(MapTransform):
    def __init__(self, keys, interp='tps', min_kpts=20, max_kpts=30, 
                 kpt_heatmap=False, heatmap_sigma=5.0, 
                 seed=None, device='cuda'):
        super().__init__(keys)
        self.interp = interp
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.kpt_heatmap = kpt_heatmap
        self.sigma = heatmap_sigma
        self.seed = seed
        self.device = device
    
    def __call__(self, data):
        d = dict(data)
        kpts_path = d[self.keys[0]]
        tumor_seg = d[self.keys[2]]
        gt_ddf = d[self.keys[3]]

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
            
            tumor_coords = np.argwhere(tumor_seg > 0)
            tumor_center = np.mean(tumor_coords, axis=0)
            distances = cdist(tumor_center.reshape(1,-1), kpts)
            #weights = None
            weights = np.exp(-distances[0]/20)
            weights /= np.sum(weights)
            
            rng = np.random.default_rng(seed=self.seed)
            k = rng.integers(self.min_kpts, self.max_kpts+1)
            choices = rng.choice(range(kpts.shape[0]), p=weights, size=k, replace=False)
            #k = np.random.randint(self.min_kpts, self.max_kpts+1)
            #choices = np.random.choice(range(kpts.shape[0]), size=k, p=weights, replace=False)
            #choices = np.argsort(distances[0])[:25]
            kpts = kpts[choices]
            kpts = torch.from_numpy(kpts)

            kpts_norm = torch.stack([
                (kpts[:, 2] / (W - 1)) * 2 - 1,
                (kpts[:, 1] / (H - 1)) * 2 - 1,
                (kpts[:, 0] / (D - 1)) * 2 - 1
            ], dim=1).to(self.device)

            grid = kpts_norm.view(1, -1, 1, 1, 3)
            sampled_disps = F.grid_sample(gt_ddf.unsqueeze(0), grid, mode='bilinear', align_corners=True).permute(2,1,0,3,4).squeeze()
            sampled_disps = sampled_disps.to(self.device)
            
            init_ddf = ddf_interp(kpts_norm.unsqueeze(0), sampled_disps.unsqueeze(0)).squeeze(0)  # (3, D, H, W)

            d[self.keys[4]] = init_ddf.cpu()

        if self.kpt_heatmap:
            d['kpt_heatmap'] = self.get_keypoints_heatmap(kpts_norm, size=(D,H,W), sigma=self.sigma, device=self.device)

        return d

    @staticmethod
    def get_keypoints_heatmap(kpts, size, sigma=5.0, ds_factor=2, device='cuda'):
        """
        Generate a heatmap where each keypoint contributes a 3D Gaussian,
        optionally generated at a lower resolution and upsampled for speed.

        Args:
            size: tuple (D, H, W) - original size
            kpts: (N, 3) tensor of coords normalized in [-1, 1]
            sigma: standard deviation of Gaussian at original scale
            device: device to put tensors on
            downsample_factor: int, factor to downsample spatial dims for heatmap generation

        Returns:
            heatmap: (1, D, H, W) torch tensor on CPU (same as your original)
        """
        D, H, W = size
        # Calculate downsampled size
        D_ds, H_ds, W_ds = D // ds_factor, H // ds_factor, W // ds_factor
        
        with torch.no_grad():
            # Convert normalized keypoints [-1, 1] to downsampled voxel coordinates
            kpts_voxel = torch.zeros_like(kpts, device=device, dtype=torch.float32)
            kpts_voxel[:, 0] = ((kpts[:, 0] + 1) / 2) * (W_ds - 1)
            kpts_voxel[:, 1] = ((kpts[:, 1] + 1) / 2) * (H_ds - 1)
            kpts_voxel[:, 2] = ((kpts[:, 2] + 1) / 2) * (D_ds - 1)

            # Create coordinate grid at downsampled size (on GPU)
            z, y, x = torch.meshgrid(
                torch.arange(D_ds, device=device).float(), 
                torch.arange(H_ds, device=device).float(), 
                torch.arange(W_ds, device=device).float(), 
                indexing='ij')
            grid = torch.stack([x, y, z], dim=-1).unsqueeze(0)  # (1, D_ds, H_ds, W_ds, 3)

            # Reshape keypoints for broadcasting
            kpts_voxel = kpts_voxel.view(-1, 1, 1, 1, 3)  # (N, 1, 1, 1, 3)

            # Squared distances (broadcast)
            sq_dist = ((grid - kpts_voxel) ** 2).sum(-1)  # (N, D_ds, H_ds, W_ds)

            # Adjust sigma according to downsampling (smaller grid means smaller sigma)
            sigma_ds = sigma / ds_factor
            heatmap_ds = torch.exp(-sq_dist / (2 * sigma_ds ** 2))  # (N, D_ds, H_ds, W_ds)

            heatmap_ds = heatmap_ds.sum(0, keepdim=True)  # (1, D_ds, H_ds, W_ds)
            heatmap_ds /= heatmap_ds.max()
            heatmap_ds = heatmap_ds.unsqueeze(0)  # Add batch dim: (1, 1, D_ds, H_ds, W_ds)

            heatmap = F.interpolate(
                heatmap_ds,
                size=(D, H, W),
                mode='trilinear',
                align_corners=True
            ).squeeze(0)  # (1, D, H, W)

            return heatmap.cpu()

class LoadEdemaRegiond(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        upenn_edema_seg = d[self.keys[0]]
        if os.path.isdir(upenn_edema_seg): ###
            upenn_edema_seg = glob.glob(os.path.join(upenn_edema_seg, "*"))[0]
        mask = nib.load(upenn_edema_seg).get_fdata()
        mask[mask == 2] = 1
        mask[mask == 4] = 1
        mask = mask > 0
        d[self.keys[0]] = mask
        return d
    
class CenterCropToDivisibled(MapTransform):
    def __init__(self, keys, k_divisible):
        super().__init__(keys)
        self.k_divisible = k_divisible

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            tensor = d[key]
            size = tensor.shape[1:]
            new_size = [dim - (dim % self.k_divisible) for dim in size]
            if list(size) != new_size:
                cropper = CenterSpatialCrop(new_size)
                d[key] = cropper(tensor)
        return d
    

def get_training_transforms(img_keys, kpts_keys, disp_field_keys, seg_keys, upenn_edema_seg_keys, augment=True,
                            interp='tps', min_kpts=15, max_kpts=20, kpt_heatmap=False, heatmap_sigma=5.0, 
                            k_divisible=1, kpts_sampling_seed=None, ddf_sampling_seed=None, device='cuda'):
    seg_keys_ = seg_keys.copy()
    seg_keys_ += upenn_edema_seg_keys
    all_keys = img_keys + disp_field_keys + seg_keys_
    all_keys += ['kpt_heatmap'] if kpt_heatmap else []

    load_imgs = LoadImaged(keys=img_keys + seg_keys, image_only=True, 
                           reader=[NibabelReader(), NrrdReader()])
    load_edema_region = LoadEdemaRegiond(keys=upenn_edema_seg_keys)
    load_rand_ddf = LoadRandDDFd(keys=disp_field_keys, seed=ddf_sampling_seed)
    interp_init_ddf = InterpKptDispsd(keys=kpts_keys + seg_keys + disp_field_keys,
                                      interp=interp, min_kpts=min_kpts, max_kpts=max_kpts,
                                      kpt_heatmap=kpt_heatmap, heatmap_sigma=heatmap_sigma,
                                      seed=kpts_sampling_seed, device=device)
    clean_extra = DeleteItemsd(keys=kpts_keys)
    ensure_first_channel = EnsureChannelFirstd(keys=img_keys + seg_keys_, channel_dim='no_channel')
    foreground_crop = CropForegroundd(keys=all_keys, source_key=seg_keys[0], allow_smaller=True,
                           margin=15, k_divisible=1, mode='edge',
                           start_coord_key=None, end_coord_key=None)
    divisible_crop = CenterCropToDivisibled(keys=all_keys, k_divisible=k_divisible)
    norm = NormalizeIntensityd(keys=img_keys, nonzero=False)
    #fixed_size = ResizeWithPadOrCropd(keys=all_keys, spatial_size=size, mode=('constant',) * len(all_keys))
    tensor = ToTensord(keys=all_keys, dtype=torch.float32)

    if augment:
        #padding_mode = ("constant",) * (len(img_keys) + len(disp_field_keys)) + ("border",) * len(seg_keys_)
        #mode = (3,) * (len(img_keys) + len(disp_field_keys)) + ("nearest",) * len(seg_keys_)
        #rotate_range = (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi)
        #translate_range = (0.0, 0.0, 0.0)
        #scale_range = ((-0.3, 0.3),) * 3
        #affine = OneOf(
        #    transforms=[
        #        RandAffined(keys=all_keys,
        #                    prob=1.0,
        #                    rotate_range=rotate_range,
        #                    scale_range=((0.0, 0.0),) * 3,
        #                    translate_range=translate_range,
        #                    mode=mode,
        #                    padding_mode=padding_mode),
        #        RandAffined(keys=all_keys,
        #                    prob=1.0,
        #                    rotate_range=(0.0, 0.0, 0.0),
        #                    scale_range=scale_range,
        #                    translate_range=translate_range,
        #                    mode=mode,
        #                    padding_mode=padding_mode),
        #        RandAffined(keys=all_keys,
        #                    prob=1.0,
        #                    rotate_range=rotate_range,
        #                    scale_range=scale_range,
        #                    translate_range=translate_range,
        #                    mode=mode,
        #                    padding_mode=padding_mode),
        #        ],
        #        weights=[0.45, 0.45, 0.1]
        #)
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
        #mirror_x = RandFlipd(keys=all_keys, spatial_axis=[0], prob=0.5)
        #mirror_y = RandFlipd(keys=all_keys, spatial_axis=[1], prob=0.5)
        #mirror_z = RandFlipd(keys=all_keys, spatial_axis=[2], prob=0.5)

        transform = Compose([
            load_imgs,
            load_edema_region,
            load_rand_ddf,
            interp_init_ddf,
            clean_extra,
            ensure_first_channel,
            foreground_crop,
            divisible_crop,
            norm,
            #affine,  # 1
            gauss_noise,  # 2
            gauss_smooth,  # 3
            scale_intensity,  # 4
            shift_intensity,  # 5
            sim_lowres,  # 6
            adjust_contrast,  # 7
            #mirror_x, mirror_y, mirror_z,  # 8
            #fixed_size,
            tensor
        ])

    else:
        transform = Compose([
            load_imgs,
            load_edema_region,
            load_rand_ddf,
            interp_init_ddf,
            clean_extra,
            ensure_first_channel,
            foreground_crop,
            divisible_crop,
            norm,
            #fixed_size,
            tensor
        ])

    return transform
