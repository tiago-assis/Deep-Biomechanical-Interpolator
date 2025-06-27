import os
import glob
from natsort import natsorted
from monai.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from model_pipeline.datatools.transforms import get_training_transforms


def generate_datasets(data_path, test_split=0.15, augment=True,
                      interp='tps', min_kpts=15, max_kpts=20, kpt_heatmap=False, heatmap_sigma=5.0, k_divisible=1,
                      device='cuda', kpts_sampling_seed=None, ddf_sampling_seed=None, data_split_seed=None):
        data_path = os.path.join(data_path, "*")
        train_kpts = natsorted(glob.glob(os.path.join(data_path, "keypoints", "*.key")))
        train_kpts = [f for f in train_kpts if "-w" not in f]
        train_imgs = []
        for kpt_path in train_kpts:
            img_path = os.path.join(os.path.dirname(kpt_path), "..", "images")
            if "T1ce" in kpt_path:
                train_imgs.append(glob.glob(os.path.join(img_path, "*T1ce*"))[0])
            elif "T2" in kpt_path:
                train_imgs.append(glob.glob(os.path.join(img_path, "*T2*"))[0])
            else:
                raise NotImplementedError(f"Only keypoints for T1ce and T2 images are supported.")
        train_ddfs = natsorted(glob.glob(os.path.join(data_path, "simulations")))
        train_brain_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*brain_mask*")))
        train_tumor_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*tumor.seg.nrrd")))
        train_edema_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*edema*"))) ##
        train_data = [
            {
                'img': train_imgs[i],                 
                'kpts': train_kpts[i],
                'gt_ddf': train_ddfs[i], 
                'init_ddf': None, 
                'brain_seg': train_brain_segs[i],
                'tumor_seg': train_tumor_segs[i], 
                'upenn_edema_seg': train_edema_segs[i]
            } 
            for i in range(len(train_imgs))
        ]
        
        transform_kwargs = {
            'img_keys': ['img'],
            'kpts_keys': ['kpts'],
            'disp_field_keys': ['gt_ddf', 'init_ddf'],
            'seg_keys': ['brain_seg', 'tumor_seg'],
            'upenn_edema_seg_keys': ['upenn_edema_seg'],
            'interp': interp,
            'min_kpts': min_kpts,
            'max_kpts': max_kpts,
            'kpt_heatmap': kpt_heatmap,
            'heatmap_sigma': heatmap_sigma,
            'k_divisible': k_divisible,
            'device': device
        }
        train_transforms = get_training_transforms(
            **transform_kwargs,
            augment=augment,
            kpts_sampling_seed=kpts_sampling_seed,
            ddf_sampling_seed=ddf_sampling_seed
        )
        val_transforms = get_training_transforms(
            **transform_kwargs,
            augment=False,
            kpts_sampling_seed=42,
            ddf_sampling_seed=42
        )
        
        train_dataset, val_dataset = train_test_split(train_data, test_size=test_split, random_state=data_split_seed)
        train_dataset = Dataset(data=train_dataset, transform=train_transforms)
        val_dataset = Dataset(data=val_dataset, transform=val_transforms)
        return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size=1, shuffle=True, dataloader_generator=None, **dataloader_kwargs):
    pin_memory = False #torch.cuda.is_available() ###  
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, generator=dataloader_generator, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=pin_memory, **dataloader_kwargs)
    return train_dataloader, val_dataloader
