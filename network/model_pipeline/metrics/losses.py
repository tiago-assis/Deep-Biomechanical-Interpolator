import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        assert pred.shape == target.shape, f"Shapes must match: {pred.shape} vs {target.shape}"
        assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."

        loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() + self.eps)
        return loss

# Code adapted from https://github.com/SamuelJoutard/DrivingPointsPredictionMIR/blob/01e3dd8c4188e70a6113209335f2ecaf1ce0a75d/losses.py
class LNCC(nn.Module):
    def __init__(self, win=3, eps=1e-5):
        super().__init__()
        self.win = win
        self.kernel = 2 * self.win + 1
        self.eps = eps

        # Create normalized 3D mean filter
        weight = torch.ones(1, 1, self.kernel, self.kernel, self.kernel, requires_grad=False) / self.kernel ** 3
        self.register_buffer('weight', weight)

    def conv(self, x):
        # Expect input to be (B, 1, D, H, W)
        return F.conv3d(x, self.weight, padding=self.win)

    def forward(self, I, J):
        """
        I, J: Input tensors (B, 1, D, H, W)
        Returns: scalar negative LNCC loss
        """
        I_mean = self.conv(I)
        J_mean = self.conv(J)
        I2_mean = self.conv(I * I)
        J2_mean = self.conv(J * J)
        IJ_mean = self.conv(I * J)

        cross = IJ_mean - (I_mean * J_mean)
        I_var = I2_mean - (I_mean ** 2)
        J_var = J2_mean - (J_mean ** 2)
        lncc = cross * cross / (I_var * J_var + self.eps)

        return -lncc.mean()
    

def jacobian(disp):
    """
    Compute the jacobian of a displacement field B, 3, X, Y, Z
    """
    d_dx = disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dy = disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dz = disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
    jac = torch.stack([d_dx, d_dy, d_dz], dim=1) # B, [ddisp_./dx, ddisp_./dy, ddisp_./dz], [ddisp_x/d., ddisp_y/d., ddisp_z/d.], X, Y, Z
    jac = F.pad(jac, (0, 1, 0, 1, 0, 1)) # B, 3, 3, X, Y, Z
    return jac


class JacobianDetLoss(nn.Module):
    def __init__(self, mode='negative', eps=1e-6):
        super().__init__()
        assert mode in ['negative', 'unit'], f"Mode must be either 'negative' or 'unit'. 'negative' penalizes negative values, 'unit' penalizes deviations from 1"
        self.mode = mode
        self.eps = eps

    def forward(self, disp, mask=None, return_matrix=False):
        jac = jacobian((disp)[:, [2, 1, 0]]) ### dim reordering
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
            assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."
            mask = mask.squeeze(1)

            if self.mode == 'unit':
                penalty = ((det-1)**2)
                penalty = (penalty * mask).sum() / (mask.sum() + self.eps)
            else:
                penalty = F.relu(-det)
                penalty = ((penalty > 0).float() * mask).sum() / (det.numel() - (1.0 - mask).sum() + self.eps)
            
        else:
            if self.mode == 'unit':
                penalty = ((det-1)**2).mean()
            else:
                penalty = F.relu(-det)
                penalty = ((penalty > 0).float()).sum() / (det.numel() + self.eps)

        return (penalty, det.cpu().unsqueeze(1)) if return_matrix else penalty


#class JacobianDetLoss(nn.Module):
#    def __init__(self, mode='negative', eps=1e-6):
#        super().__init__()
#        assert mode in ['negative', 'unit'], f"Mode must be either 'negative' or 'unit'. 'negative' penalizes negative values, 'unit' penalizes deviations from 1"
#        self.mode = mode
#        self.eps = eps
#
#    def forward(self, disp, mask=None, return_matrix=False):
#        jac = jacobian((disp)[:, [2, 1, 0]]) ### dim reordering
#        jac[:, 0, 0] += 1.0
#        jac[:, 1, 1] += 1.0
#        jac[:, 2, 2] += 1.0
#        det = (
#            jac[:, 0, 0] * jac[:, 1, 1] * jac[:, 2, 2] +
#            jac[:, 0, 1] * jac[:, 1, 2] * jac[:, 2, 0] +
#            jac[:, 0, 2] * jac[:, 1, 0] * jac[:, 2, 1] -
#            jac[:, 0, 0] * jac[:, 1, 2] * jac[:, 2, 1] - 
#            jac[:, 0, 1] * jac[:, 1, 0] * jac[:, 2, 2] -
#            jac[:, 0, 2] * jac[:, 1, 1] * jac[:, 2, 0]
#        )
#        
#        if mask is not None:
#            assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."
#            mask = mask.squeeze(1)
#
#            if self.mode == 'unit':
#                penalty = ((det-1)**2)
#            else:
#                penalty = F.relu(-det)
#            
#            penalty = (penalty * mask).sum() / (mask.sum() + self.eps)
#
#        else:
#            if self.mode == 'unit':
#                penalty = ((det-1)**2).mean()
#            else:
#                penalty = F.relu(-det).mean()
#
#        return (penalty, det.cpu().unsqueeze(1)) if return_matrix else penalty
    
    
class BendingEnergyLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, disp, mask=None, return_matrix=False):
        be = 0.0
        for i in range(disp.shape[1]):
            u = disp[:, i:i+1]
            dzz = u[:, :, 2:, 1:-1, 1:-1] - 2 * u[:, :, 1:-1, 1:-1, 1:-1] + u[:, :, :-2, 1:-1, 1:-1]
            dyy = u[:, :, 1:-1, 2:, 1:-1] - 2 * u[:, :, 1:-1, 1:-1, 1:-1] + u[:, :, 1:-1, :-2, 1:-1]
            dxx = u[:, :, 1:-1, 1:-1, 2:] - 2 * u[:, :, 1:-1, 1:-1, 1:-1] + u[:, :, 1:-1, 1:-1, :-2]
            be += dxx**2 + dyy**2 + dzz**2

        be = F.pad(be, (1, 1, 1, 1, 1, 1))
        
        if mask is not None:
            assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."

            reg = (be * mask).sum() / (mask.sum() + self.eps)
        else:
            reg = be.mean()

        return (reg, be.cpu()) if return_matrix else reg
    