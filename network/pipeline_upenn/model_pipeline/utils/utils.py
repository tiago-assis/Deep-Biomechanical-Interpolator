import torch
import torch.nn.functional as F


def warp_tensor(img, disp, mode='bilinear', padding_mode='border'):
    """
    warp the tensor img according to the displacement field disp
    img: Tensor to warp (1, C, X, Y, Z)
    disp: Displacement field tensor (1, 3, X, Y, Z)
    mode: interpolation mode to use
    padding_mode: padding mode
    """
    _, _, D, H, W = img.shape
    device = img.device
    disp = disp.permute(0, 2, 3, 4, 1)
    identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, D, H, W), align_corners=True)
    return F.grid_sample(img, identity + disp, mode=mode, padding_mode=padding_mode, align_corners=True)
