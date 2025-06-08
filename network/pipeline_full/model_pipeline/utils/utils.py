import torch
import torch.nn.functional as F


def warp_tensor(img, disp, mode='bilinear', padding_mode='border'):
    """
    warp the tensor img according to the displacement field disp
    img: Tensor to warp (1, C, X, Y, Z)
    disp: Displacement field tensor (1, X, Y, Z, 3)
    mode: interpolation mode to use
    padding_mode: padding mode
    """
    _, _, D, H, W = img.shape
    device = img.device
    identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, D, H, W), align_corners=True)
    return F.grid_sample(img, identity + disp, mode=mode, padding_mode=padding_mode, align_corners=True)
