import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from monai.transforms import LoadImaged, EnsureChannelFirstd, CropForegroundd, NormalizeIntensityd, \
    RandAffined, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, \
    RandScaleIntensityFixedMeand, RandSimulateLowResolutiond, RandAdjustContrastd, RandFlipd, Compose, \
    DeleteItemsd, ToTensord, OneOf, ResizeWithPadOrCropd, MapTransform
from monai.data import NibabelReader, NrrdReader, ITKReader
import nibabel as nib
from model_pipeline.interpolators.interpolators import LinearInterpolation3d, ThinPlateSpline


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        rnd_sim = sims[rnd_idx]
        gt_ddf = np.load(rnd_sim, allow_pickle=True)['field'].transpose(0,3,2,1)
        gt_ddf = torch.from_numpy(gt_ddf)
        d[self.keys[0]] = gt_ddf
        del gt_ddf  
        return d
    
class InterpKptDispsd(MapTransform):
    def __init__(self, keys, interp='tps', min_kpts=15, max_kpts=20, seed=None, device=DEVICE):
        super().__init__(keys)
        self.interp = interp
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.seed = seed
        self.device = device
    
    def __call__(self, data):
        d = dict(data)
        kpts = d[self.keys[0]]
        tumor_seg = d[self.keys[2]]
        gt_ddf = d[self.keys[3]]
        gt_ddf = gt_ddf.to(self.device)

        _, D, H, W = gt_ddf.shape

        if self.interp == 'linear':
            ddf_interp = LinearInterpolation3d((D,H,W)).to(self.device)
        elif self.interp == 'tps':
            ddf_interp = ThinPlateSpline((D,H,W)).to(self.device)

        kpts = np.genfromtxt(kpts, delimiter="\t", skip_header=6, dtype=np.float32)[:,:3]
        _, unique_idxs = np.unique(kpts, axis=0, return_index=True)
        sorted_unique_idxs = np.sort(unique_idxs)
        kpts = kpts[sorted_unique_idxs]
        kpts[:, 0] = (kpts[:, 0] / (D - 1)) * 2 - 1
        kpts[:, 1] = (kpts[:, 1] / (H - 1)) * 2 - 1
        kpts[:, 2] = (kpts[:, 2] / (W - 1)) * 2 - 1
        
        tumor_coords = np.argwhere(tumor_seg > 0)
        tumor_center = np.mean(tumor_coords, axis=0)
        distances = cdist(tumor_center.reshape(1,-1), kpts)
        weights = np.exp(-distances[0])
        weights /= np.sum(weights)

        rng = np.random.default_rng(seed=self.seed)
        k = rng.integers(self.min_kpts, self.max_kpts+1)
        choices = rng.choice(range(kpts.shape[0]), p=weights, size=k, replace=False)
        kpts = kpts[choices]
        kpts = torch.from_numpy(kpts).to(self.device)
        grid = kpts.view(1, -1, 1, 1, 3)
        sampled_disps = F.grid_sample(gt_ddf.unsqueeze(0), grid, mode='bilinear', align_corners=True).permute(2,1,0,3,4).squeeze()
        sampled_disps = sampled_disps.to(self.device)
        
        init_ddf = ddf_interp(kpts.unsqueeze(0), sampled_disps.unsqueeze(0)).squeeze(0)  # (3, D, H, W)

        d[self.keys[4]] = init_ddf.cpu()
        del gt_ddf, ddf_interp, kpts, sampled_disps, init_ddf
        return d

#class LoadEdemaRegiond(MapTransform):
#    def __init__(self, keys):
#        super().__init__(keys)
#
#    def __call__(self, data):
#        d = dict(data)
#        upenn_tumor_seg = d[self.keys[0]]
#        if upenn_tumor_seg is not None:
#            #if os.path.isdir(upenn_tumor_seg): # to deal with kaggle making weird folders ## KAGGLE WARNING
#            #    upenn_tumor_seg = glob.glob(os.path.join(upenn_tumor_seg, "*"))[0]
#            mask = nib.load(upenn_tumor_seg).get_fdata()
#            mask[mask == 2] = 1
#            mask[mask == 4] = 1
#            mask = mask > 0
#            d[self.keys[0]] = mask
#            del mask
#        return d
    
def get_training_transforms(img_keys, disp_field_keys, seg_keys, kpts_key,
                            interp='tps', min_kpts=15, max_kpts=20, kpts_sampling_seed=None, 
                            size=(192, 224, 160), device=DEVICE):
    all_keys = img_keys + disp_field_keys + seg_keys

    load_imgs = LoadImaged(keys=img_keys + seg_keys, image_only=True, 
                           reader=[NibabelReader(), NrrdReader()])

    load_rand_ddf = LoadRandDDFd(keys=disp_field_keys)

    interp_init_ddf = InterpKptDispsd(keys=kpts_key + seg_keys + disp_field_keys,
                                      interp=interp,
                                      min_kpts=min_kpts, max_kpts=max_kpts,
                                      seed=kpts_sampling_seed, device=device)
    
    clean_extra = DeleteItemsd(keys=kpts_key)

    ensure_first_channel = EnsureChannelFirstd(keys=img_keys + seg_keys, channel_dim='no_channel')

    crop = CropForegroundd(keys=all_keys, source_key=seg_keys[0], 
                           margin=15, start_coord_key=None, end_coord_key=None)

    norm = NormalizeIntensityd(keys=img_keys, nonzero=True)

    padding_mode_ = ("constant",) * (len(img_keys) + len(disp_field_keys)) + ("border",) * len(seg_keys)
    mode_ = (3,) * (len(img_keys) + len(disp_field_keys)) + ("nearest",) * len(seg_keys)
    rotate_range_ = (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi)
    translate_range_ = (0.0, 0.0, 0.0)
    scale_range_ = ((-0.3, 0.3),) * 3
    affine = OneOf(
        transforms=[
            RandAffined(keys=all_keys,
                        prob=1.0,
                        rotate_range=rotate_range_,
                        scale_range=((0.0, 0.0),) * 3,
                        translate_range=translate_range_,
                        mode=mode_,
                        padding_mode=padding_mode_),
            RandAffined(keys=all_keys,
                        prob=1.0,
                        rotate_range=(0.0, 0.0, 0.0),
                        scale_range=scale_range_,
                        translate_range=translate_range_,
                        mode=mode_,
                        padding_mode=padding_mode_),
            RandAffined(keys=all_keys,
                        prob=1.0,
                        rotate_range=rotate_range_,
                        scale_range=scale_range_,
                        translate_range=translate_range_,
                        mode=mode_,
                        padding_mode=padding_mode_),
            ],
            weights=[0.45, 0.45, 0.1]
    )

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

    mirror_x = RandFlipd(keys=all_keys, spatial_axis=[0], prob=0.5)
    mirror_y = RandFlipd(keys=all_keys, spatial_axis=[1], prob=0.5)
    mirror_z = RandFlipd(keys=all_keys, spatial_axis=[2], prob=0.5)

    fixed_size = ResizeWithPadOrCropd(keys=all_keys, spatial_size=size, mode=('constant',) * len(all_keys))
    
    tensor = ToTensord(keys=all_keys, dtype=torch.float32)

    transform = Compose([
        load_imgs,
        load_rand_ddf,
        interp_init_ddf,
        clean_extra,
        ensure_first_channel,
        crop,
        norm,
        affine,  # 1
        gauss_noise,  # 2
        gauss_smooth,  # 3
        scale_intensity,  # 4
        shift_intensity,  # 5
        sim_lowres,  # 6
        adjust_contrast,  # 7
        mirror_x, mirror_y, mirror_z,  # 8
        fixed_size,
        tensor
    ])

    return transform
