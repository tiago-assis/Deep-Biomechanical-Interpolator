import os
import glob
from natsort import natsorted
from monai.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
from model_pipeline.datatools.transforms import get_training_transforms

CUDA_IS_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_IS_AVAILABLE else 'cpu'

def generate_datasets(data_path, data_split=[0.85, 0.15], interp='tps', min_kpts=15, max_kpts=20, size=(192, 224, 160), device=DEVICE, kpts_sampling_seed=None, data_split_generator=None):
        data_path = os.path.join(data_path, "*")
        train_kpts = natsorted(glob.glob(os.path.join(data_path, "keypoints", "*.key")))
        train_kpts = [f for f in train_kpts if "-w" not in f]
        train_imgs = []
        for kpt_path in train_kpts:
            img_path = kpt_path + "/../../images"
            if "T1ce" in kpt_path:
                train_imgs.append(glob.glob(os.path.join(img_path, "*T1ce*"))[0])
            elif "T2" in kpt_path:
                train_imgs.append(glob.glob(os.path.join(img_path, "*T2*"))[0])
            else:
                raise NotImplementedError(f"Only keypoints for T1ce and T2 images are supported.")
        train_ddfs = natsorted(glob.glob(os.path.join(data_path, "simulations")))
        train_brain_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*brain_mask*")))
        train_tumor_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*tumor.seg.nrrd")))
        train_edema_segs = natsorted(glob.glob(os.path.join(data_path, "segmentations", "*edema*")))
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
            for i in range(len(glob.glob(data_path)))
        ]
        
        transforms = get_training_transforms(
            img_keys=['img'],
            kpts_keys=['kpts'],
            disp_field_keys=['gt_ddf', 'init_ddf'],
            seg_keys=['brain_seg', 'tumor_seg'],
            upenn_edema_seg_keys=['upenn_edema_seg'],
            interp=interp,
            min_kpts=min_kpts,
            max_kpts=max_kpts,
            kpts_sampling_seed=kpts_sampling_seed,
            size=size,
            device=device
        )

        dataset = Dataset(data=train_data, transform=transforms)
        train_dataset, val_dataset = random_split(dataset, data_split, generator=data_split_generator)
        return train_dataset, val_dataset

def generate_dataloaders(train_dataset, val_dataset, batch_size=1, shuffle=True, dataloader_generator=None, **dataloader_kwargs):
    pin_memory = CUDA_IS_AVAILABLE     
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, generator=dataloader_generator, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, generator=dataloader_generator, **dataloader_kwargs)
    return train_dataloader, val_dataloader
