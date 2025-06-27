import os
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F


def dilate_binary_mask(mask, kernel_size=5, iters=1, binary_mask=True):
    """
    Dilates a binary 3D mask.
    
    Args:
        mask (torch.Tensor): binary mask of shape (B, C, D, H, W)
        kernel_size (int): size of the dilation kernel (must be odd)
        iter (int): number of dilation steps
        
    Returns:
        torch.Tensor: dilated mask (same shape, binary)
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."

    kernel = torch.ones((mask.shape[1], mask.shape[1], kernel_size, kernel_size, kernel_size), device=mask.device)
    padding = kernel_size // 2
    for _ in range(iters):
        mask = F.conv3d(mask.float(), kernel, padding=padding)
        if binary_mask:
            mask = (mask > 0.5).float()
    return mask


def warp_tensor(img, disp, mask=None, mode='bilinear', padding_mode='zeros', bg_value=0.0):
    """
    Warp the tensor img according to the displacement field disp.

    img: Tensor to warp (B, C, D, H, W)
    disp: Displacement field tensor (B, 3, D, H, W)
    mode: Interpolation mode to use
    padding_mode: Padding mode
    """
    B, _, D, H, W = img.shape
    device = img.device

    # Create base grid in normalized [-1, 1] coordinates
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y, grid_z), dim=3)  # (D, H, W, 3) ######### attention to axis ordering
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()  # (B, D, H, W, 3)

    # Normalize disp to [-1, 1]
    norm_disp = torch.stack([
        disp[:, 2] / ((W - 1) / 2),
        disp[:, 1] / ((H - 1) / 2),
        disp[:, 0] / ((D - 1) / 2)
    ], dim=1).permute(0, 2, 3, 4, 1)

    # Add displacement to base grid
    sampling_grid = base_grid + norm_disp

    warped_img = F.grid_sample(img, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if mask is not None:
        if mask.shape[0] != B:
            mask = mask.expand(B, -1, -1, -1, -1).contiguous()

        # Warp the mask to know where the warped image has valid brain tissue
        warped_mask = F.grid_sample(mask.float(), sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        warped_mask = (warped_mask > 0.5).float()  # (B, 1, D, H, W)

        # Combine warped image with background
        warped_img = warped_img * warped_mask + (img - img * mask) + bg_value * (mask - warped_mask)

    return warped_img


def save_checkpoint(model,
                   optimizer,
                   epoch,
                   loss,
                   save_dir='checkpoints',
                   inner_dir='',
                   is_best=False,
                   filename=None,
                   additional_info=None,
                   verbose=False):
    """
    Save training checkpoint with best-model tracking.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_dir: Regular checkpoint directory
        best_dir: Best model directory
        is_best: Whether this is the best model so far
        filename: Optional custom filename
        additional_info: Extra metadata to save
    """
    # Create directories if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare checkpoint data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': timestamp
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save regular checkpoint
    
    if filename is None:
        filename = "checkpoint"
    filename += f"_epoch{epoch}_{timestamp}.pt"
    
    save_path = os.path.join(save_dir, inner_dir, filename)
    torch.save(checkpoint, save_path)
    
    # Handle best model
    if is_best:
        best_path = os.path.join(save_dir, inner_dir, filename+"_best.pt")
        os.makedirs(best_path, exist_ok=True)
        shutil.copyfile(save_path, best_path)
        if verbose:
            print(f"New best model saved to {best_path} (loss: {loss:.4f})")
    if verbose:
        print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, verbose=False):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load state into
        optimizer (torch.optim): Optional optimizer to load state
        
    Returns:
        dict: Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if verbose:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.4f})")
    return checkpoint