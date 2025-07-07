import os
import glob
from natsort import natsorted
from monai.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model_pipeline.datatools.transforms import get_training_transforms


def generate_datasets(data_path: str, 
                      split: List[float] = [0.1, 0.15], 
                      augment: bool = True, 
                      size: Tuple[int, int, int] = (160,192,144),
                      interp: str = 'tps', 
                      min_kpts: int = 15, 
                      max_kpts: int = 20, 
                      device: str = 'cuda', 
                      kpts_sampling_seed: Optional[int] = None, 
                      ddf_sampling_seed: Optional[int] = None, 
                      data_split_seed: Optional[int] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepares and returns training, validation, and test datasets from a data directory of medical image data.

    Args:
        data_path (str): Path to the root data directory.
        split (List[float], optional): List specifying validation and test splits. Defaults to [0.1, 0.15].
        augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        size (Tuple[int, int, int], optional): Target shape (D, H, W) of the input tensors. Defaults to (160, 192, 144).
        interp (str, optional): Interpolation method for displacement field. Defaults to 'tps'.
        min_kpts (int, optional): Minimum number of keypoints to sample. Defaults to 15.
        max_kpts (int, optional): Maximum number of keypoints to sample. Defaults to 20.
        device (str, optional): Device to load data to ('cuda' or 'cpu'). Defaults to 'cuda'.
        kpts_sampling_seed (Optional[int], optional): Random seed for keypoint sampling. Defaults to None.
        ddf_sampling_seed (Optional[int], optional): Random seed for DDF generation. Defaults to None.
        data_split_seed (Optional[int], optional): Seed for train/val/test split. Defaults to None.

    Returns:
        train_dataset, val_dataset, test_dataset (Tuple[Dataset, Dataset, Dataset]): A tuple containing the train, validation, and test datasets.
    """
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
        'size': size,
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
        
    val_split = split[-2] + split[-1]
    test_split = split[-1] / val_split

    train_dataset, val_dataset = train_test_split(train_data, test_size=val_split, random_state=data_split_seed)
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=test_split, random_state=data_split_seed)

    train_dataset = Dataset(data=train_dataset, transform=train_transforms)
    val_dataset = Dataset(data=val_dataset, transform=val_transforms)
    test_dataset = Dataset(data=test_dataset, transform=val_transforms)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset: Dataset, 
                    val_dataset: Dataset, 
                    test_dataset: Dataset, 
                    batch_size: int = 1, 
                    shuffle: bool = True, 
                    dataloader_generator: Optional[Any] = None, 
                    **dataloader_kwargs: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        dataloader_generator (Optional[Any], optional): Random generator for deterministic data loading. Defaults to None.
        **dataloader_kwargs (dict): Additional keyword arguments to pass to the DataLoader.

    Returns:
        train_dataloader, val_dataloader, test_dataloader (Tuple[DataLoader, DataLoader, DataLoader]): A tuple containing the training, validation, and test DataLoaders.
    """
    pin_memory = torch.cuda.is_available() ###  

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, generator=dataloader_generator, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=pin_memory, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, **dataloader_kwargs)

    return train_dataloader, val_dataloader, test_dataloader
