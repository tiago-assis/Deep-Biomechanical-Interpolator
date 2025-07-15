import os
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any


def dilate_binary_mask(mask: torch.Tensor, kernel_size: int = 5, iters: int = 1) -> torch.Tensor:
    """
    Perform morphological dilation on a 3D binary mask.

    Args:
        mask (torch.Tensor): Binary mask tensor of shape (B, 1, D, H, W) with values in [0, 1].
        kernel_size (int): Size of the dilation kernel (must be odd).
        iters (int): Number of dilation iterations to apply.

    Returns:
        mask (torch.Tensor): Dilated binary mask of the same shape as input.
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."

    kernel = torch.ones((mask.shape[1], mask.shape[1], kernel_size, kernel_size, kernel_size), device=mask.device)
    padding = kernel_size // 2
    # apply convolution to binary mask for n iterations
    for n in range(iters):
        mask = F.conv3d(mask.float(), kernel, padding=padding)
        mask = (mask > 0.5).float()
    return mask


def warp_tensor(img: torch.Tensor, 
                disp: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                mode: str = 'bilinear', 
                padding_mode: str = 'zeros', 
                bg_value: Union[torch.Tensor, float] = 0.0) -> torch.Tensor:
    """
    Warps a 3D tensor using a displacement field.

    Args:
        img (torch.Tensor): Input image tensor of shape (B, C, D, H, W).
        disp (torch.Tensor): Displacement field tensor of shape (B, 3, D, H, W).
        mask (Optional[torch.Tensor]): Binary mask of shape (B, 1, D, H, W) specifying valid regions.
        mode (str): Interpolation mode: 'bilinear' or 'nearest'. Default is 'bilinear'.
        padding_mode (str): Padding mode for outside grid values: 'zeros', 'border', or 'reflection'.
        bg_value (float or torch.Tensor): Value to assign to background (unmapped) voxels.

    Returns:
        warped_img (torch.Tensor): Warped image tensor of the same shape as `img`.
    """
    B, _, D, H, W = img.shape
    device = img.device

    # create base grid in normalized [-1, 1] coordinates
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y, grid_z), dim=3)  # (D, H, W, 3)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()  # (B, D, H, W, 3)

    # normalize disps to [-1, 1]
    norm_disp = torch.stack([
        disp[:, 2] / ((W - 1) / 2),
        disp[:, 1] / ((H - 1) / 2),
        disp[:, 0] / ((D - 1) / 2)
    ], dim=1).permute(0, 2, 3, 4, 1)

    # add displacement to base grid
    sampling_grid = base_grid + norm_disp

    warped_img = F.grid_sample(img, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if mask is not None:
        if mask.shape[0] != B:
            mask = mask.expand(B, -1, -1, -1, -1).contiguous()

        # warp the mask to know where the warped image has valid brain tissue
        warped_mask = F.grid_sample(mask.float(), sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        warped_mask = (warped_mask > 0.5).float()  # (B, 1, D, H, W)

        # combine warped image with background
        warped_img = warped_img * warped_mask + (img - img * mask) + (mask - warped_mask) * bg_value

    return warped_img


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: Union[torch.Tensor, float],
                   save_dir: str = 'checkpoints',
                   is_best: bool = False,
                   filename: Optional[str] = None,
                   additional_info: Optional[Dict[str, Any]] = None,
                   verbose: bool = False):
    """
    Save a model and optimizer checkpoint to disk with best-model tracking.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        epoch (int): Epoch number.
        loss (float or torch.Tensor): Current loss.
        save_dir (str): Directory to save the checkpoint.
        is_best (bool): Whether this is the best checkpoint so far.
        filename (Optional[str]): Optional base name for checkpoint file.
        additional_info (Optional[Dict[str, Any]]): Extra metadata to include.
        verbose (bool): Whether to print save info to console.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': timestamp,
        'filename': filename
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    if filename is None:
        filename = "checkpoint"
    best_filename = filename
    filename += f"_epoch{epoch}_{timestamp}.pt"
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    
    # handle best model
    if is_best:
        best_path = os.path.join(save_dir, "best_checkpoints")
        os.makedirs(best_path, exist_ok=True)
        shutil.copyfile(save_path, os.path.join(best_path, best_filename+"_best.pt"))
        if verbose:
            print(f"\nNew best model saved to {best_path} (MSE: {loss:.4f})")
    if verbose:
        print(f"\nCheckpoint saved to {save_path}")


def load_checkpoint(checkpoint_path: str, 
                    model: torch.nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None, 
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Load a model and optimizer from a checkpoint file.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        model (torch.nn.Module): Model to load into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load into.
        verbose (bool): Whether to print load status.

    Returns:
        Dict[str, Any]: Loaded checkpoint metadata.
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if verbose:
        filename = checkpoint.get('filename')
        print(f"\nLoaded checkpoint {'\''+filename+'\' ' if filename else ''}from epoch {checkpoint['epoch']} (MSE: {checkpoint['loss']:.4f})")
        
    return checkpoint
