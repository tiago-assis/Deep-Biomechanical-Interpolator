import torch
import torch.nn as nn
import torch.nn.functional as F


def dilate_mask(mask, kernel_size=5, iters=1):
    """
    Dilates a binary 3D mask.
    
    Args:
        mask (torch.Tensor): binary mask of shape (B, 1, D, H, W)
        kernel_size (int): size of the dilation kernel (must be odd)
        iter (int): number of dilation steps
        
    Returns:
        torch.Tensor: dilated mask (same shape, binary)
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=mask.device)
    padding = kernel_size // 2
    for _ in range(iters):
        mask = F.conv3d(mask.float(), kernel, padding=padding)
        mask = (mask > 0).float()
    return mask


# Code adapted from https://github.com/SamuelJoutard/DrivingPointsPredictionMIR/blob/01e3dd8c4188e70a6113209335f2ecaf1ce0a75d/losses.py
class LNCC(nn.Module):
    def __init__(self, w=2):
        super(LNCC, self).__init__()
        self.w = torch.ones(1, 1, (2*w+1), (2*w+1), (2*w+1)).cuda() / (2*w + 1)**3
        self.conv = nn.Conv3d(1, 1, (2*w+1), 1, w, bias=False)
        self.conv.weight.data = self.w
        self.conv.weight.requires_grad = False

    def forward(self, M, R):
        M_m = self.conv(M)
        R_m = self.conv(R)  
        MM_m = self.conv(M*M)
        RR_m = self.conv(R*R)
        MR_m = self.conv(M*R)
        M_var = torch.sqrt(torch.clamp(MM_m - M_m**2, min=0) + 1e-5)
        R_var = torch.sqrt(torch.clamp(RR_m - R_m**2, min=0) + 1e-5)
        corr = (MR_m - M_m * R_m) / (M_var * R_var + 1e-5)
        return -corr.mean()
    
def jacobian(disp):
    """
    Compute the jacobian of a displacement field B, 3, X, Y, Z
    """
    d_dx = disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dy = disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dz = disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
    jac = torch.stack([d_dx, d_dy, d_dz], dim=1) # B, [ddisp_./dx, disp_./dy, ddisp_./dz], [ddisp_x/d., ddisp_y/d., ddisp_z/d.], X, Y, Z
    return F.pad(jac, (0, 1, 0, 1, 0, 1)) # B, 3, 3, X, Y, Z

def Jacobian_det(disp, mask=None):
    """
    Computes mean jacobian determinant of the deformation field, given displacement field
    """
    jac = jacobian((disp)[:, [2, 1, 0]])
    jac[:, 0, 0] += 1.0
    jac[:, 1, 1] += 1.0
    jac[:, 2, 2] += 1.0
    det = (
        jac[:, 0, 0] * jac[:, 1, 1] * jac[:, 2, 2] +
        jac[:, 0, 1] * jac[:, 1, 2] * jac[:, 2, 0] +
        jac[:, 0, 2] * jac[:, 1, 0] * jac[:, 2, 1] -
        jac[:, 0, 0] * jac[:, 1, 2] * jac[:, 2, 1] - 
        jac[:, 0, 1] * jac[:, 1, 0] * jac[:, 2, 2] -
        jac[:, 0, 2] * jac[:, 1, 1] * jac[:, 2, 0]
    )
    if mask is not None:
        mask = dilate_mask(mask)
        mask = 1.0 - mask
        penalty = ((det-1)**2) * mask.squeeze(1)
        return penalty.sum() / (mask.sum() + 1e-8)
    else:
        return ((det-1)**2).mean()

def Hessian_penalty(disp):
    """
    Computes bending energy of the displacement field
    """
    jac = jacobian(disp) # B, 3, 3, X, Y, Z
    B, _, __, X, Y, Z = jac.size()
    hess = jacobian(torch.reshape(jac, (B, -1, X, Y, Z)))
    return (hess**2).sum((1,2)).mean()
