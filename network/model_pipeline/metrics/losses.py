import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss, only computing the loss over elements where the mask is non-zero.
    """
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error loss over the regions specified by a mask.

        Args:
            pred (Tensor): Predicted output of shape (B, C, (D, )H, W).
            target (Tensor): Ground truth tensor of the same shape as `pred`.
            mask (Tensor): Binary mask tensor of shape (B, 1, (D, )H, W).

        Returns:
            Tensor: The computed masked mean squared error loss.
        """
        assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."

        loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() + self.eps)
        return loss


# Some code adapted from https://github.com/SamuelJoutard/DrivingPointsPredictionMIR/blob/01e3dd8c4188e70a6113209335f2ecaf1ce0a75d/losses.py
class LNCC(nn.Module):
    """
    Local Normalized Cross-Correlation (LNCC) loss for 3D image volumes, 
    computing the similarity between two volumetric inputs over local windows.
    """
    def __init__(self, win: int = 3, eps: float = 1e-6):
        """
        Args:
            win (int, optional): Span of the local window (kernel radius). The full kernel size is (2*win + 1). Defaults to 3.
            eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.win = win
        self.kernel = 2 * self.win + 1
        self.eps = eps

        # Create normalized 3D mean filter
        weight = torch.ones(1, 1, self.kernel, self.kernel, self.kernel, requires_grad=False) / self.kernel ** 3
        self.register_buffer('weight', weight)

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a 3D mean filter to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, D, H, W).

        Returns:
            torch.Tensor: Smoothed tensor of the same shape as the input.
        """
        return F.conv3d(x, self.weight, padding=self.win)

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative Local Normalized Cross-Correlation (LNCC) loss between inputs.

        Args:
            I (torch.Tensor): First input tensor of shape (B, 1, D, H, W).
            J (torch.Tensor): Second input tensor of shape (B, 1, D, H, W).

        Returns:
            torch.Tensor: Scalar tensor representing the negative LNCC loss.
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
    

def jacobian(disp: torch.Tensor) -> torch.Tensor:
    """
    Computes the spatial Jacobian matrix of a 3D displacement field.

    Args:
        disp (torch.Tensor): Displacement field of shape (B, 3, X, Y, Z).

    Returns:
        jac (torch.Tensor): Jacobian matrix of shape (B, 3, 3, X, Y, Z), where
        the dimensions correspond to (B, [∂disp/∂x, ∂disp/∂y, ∂disp/∂z], [∂x, ∂y, ∂z], X, Y, Z).
    """
    d_dx = disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dy = disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dz = disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
    jac = torch.stack([d_dx, d_dy, d_dz], dim=1) # B, [ddisp_./dx, ddisp_./dy, ddisp_./dz], [ddisp_x/d., ddisp_y/d., ddisp_z/d.], X, Y, Z
    jac = F.pad(jac, (0, 1, 0, 1, 0, 1)) # B, 3, 3, X, Y, Z
    return jac


def jacobian_det(jac: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of a Jacobian matrix field.

    Args:
        jac (torch.Tensor): Jacobian matrix of shape (B, 3, 3, X, Y, Z).

    Returns:
        det (torch.Tensor): Determinant of the Jacobian at each voxel with shape (B, X, Y, Z).
    """
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
    return det


class JacobianDetLoss(nn.Module):
    """
    Loss function based on the Jacobian determinant of a displacement field.
    Supports penalizing negative Jacobian determinants (for fold detection)
    or deviations from 1 (for topology preservation). Optional masking allows
    spatial restriction of the loss computation.
    """
    def __init__(self, mode: str = 'negative', eps: float = 1e-6):
        """
        Args:
            mode (str, optional): Loss mode. 'negative' penalizes only negative determinant values (i.e., foldings). 
                'unit' penalizes deviations from 1. Defaults to 'negative'.
            eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        assert mode in ['negative', 'unit'], f"Mode must be either 'negative' or 'unit'. 'negative' penalizes negative values, 'unit' penalizes deviations from 1"
        self.mode = mode
        self.eps = eps

    def forward(self, disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the Jacobian determinant loss.

        Args:
            disp (torch.Tensor): Displacement field of shape (B, 3, X, Y, Z).
            mask (torch.Tensor, optional): Binary mask of shape (B, 1, X, Y, Z).
                If provided, loss is computed only in masked regions.
            return_matrix (bool, optional): If True, also returns the full
                Jacobian determinant matrix.

        Returns:
            penalty (torch.Tensor): Scalar tensor representing the Jacobian determinant loss.
        """
        jac = jacobian(disp[:, [2, 1, 0]]) ### dim reordering
        det = jacobian_det(jac)
        
        if mask is not None:
            assert mask.shape[1] == 1 and mask.min() >= 0 and mask.max() <= 1, "Mask tensor should be a 1-channel binary mask."
            mask = mask.squeeze(1)

            if self.mode == 'unit':
                penalty = ((det-1)**2)
                penalty = (penalty * mask).sum() / (mask.sum() + self.eps)
            else:
                penalty = F.relu(-det)
                penalty = (penalty * mask).sum() / (mask.sum() + self.eps)
            
        else:
            if self.mode == 'unit':
                penalty = ((det-1)**2).mean()
            else:
                penalty = F.relu(-det).mean()

        return penalty
    
    
class BendingEnergyLoss(nn.Module):
    """
    Computes the bending energy (second-order smoothness) of a 3D displacement field.
    Penalizes the second spatial derivatives to encourage smooth, non-oscillatory deformations.
    Optional masking allows spatial restriction of the loss computation.
    """
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, disp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the bending energy loss.

        Args:
            disp (torch.Tensor): Displacement field of shape (B, 3, D, H, W).
            mask (torch.Tensor, optional): Binary mask of shape (B, 1, D, H, W).
                If provided, loss is computed only over masked regions.

        Returns:
            Tensor: Scalar tensor representing the bending energy loss.
        """
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

        return reg
    

def tr_error(kpts: torch.Tensor, gt_ddf: torch.Tensor, pred_ddf: torch.Tensor) -> torch.Tensor:
    """
    Computes the Target Registration Error (TRE) per keypoint.

    Args:
        kpts (torch.Tensor): Keypoints in voxel coordinates of shape (B, N, 3).
        gt_ddf (torch.Tensor): Ground-truth displacement field of shape (B, 3, D, H, W)
            in voxel space.
        pred_ddf (torch.Tensor): Predicted displacement field of shape (B, 3, D, H, W)
            in voxel space.

    Returns:
        tre (torch.Tensor): Scalar tensor representing the TRE for each keypoint.
    """
    B, _, D, H, W = pred_ddf.shape
    device = pred_ddf.device

    # normalized voxel coordinates
    kpts_norm = torch.stack([
        (kpts[..., 2] / (W - 1)) * 2 - 1,
        (kpts[..., 1] / (H - 1)) * 2 - 1,
        (kpts[..., 0] / (D - 1)) * 2 - 1 
    ], dim=-1).to(device)  # (B, N, 3)

    grid = kpts_norm.view(B, -1, 1, 1, 3)  # (B, N, 1, 1, 3)

    # sample displacements
    pred_disp = F.grid_sample(pred_ddf, grid, mode='bilinear', align_corners=True)
    gt_disp = F.grid_sample(gt_ddf, grid, mode='bilinear', align_corners=True)

    # reshape to (B, N, 3)
    pred_disp = pred_disp.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    gt_disp = gt_disp.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    # apply displacements to keypoints
    pred_kpts = kpts + pred_disp
    gt_kpts = kpts + gt_disp

    tre = torch.norm(pred_kpts - gt_kpts, dim=-1)  # (B, N)

    return tre



def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Dice score for binary predictions.

    Args:
        pred (torch.Tensor): Predicted binary mask of shape (B, 1, D, H, W).
        target (torch.Tensor): Ground truth binary mask of shape (B, 1, D, H, W).
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        dice (torch.Tensor): Scalar tensor representing the dice score.
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = torch.sum(pred * target, dim=(1, 2, 3, 4))
    union = torch.sum(pred, dim=(1, 2, 3, 4)) + torch.sum(target, dim=(1, 2, 3, 4))
    dice = (2. * intersection + eps) / (union + eps)
    return dice


def hd95(pred: torch.Tensor, target: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between two binary volumetric masks.

    Args:
        pred (torch.Tensor): Predicted binary mask of shape (1, 1, D, H, W).
        target (torch.Tensor): Ground truth binary mask of shape (1, 1, D, H, W).
        chunk_size (int, optional): Number of surface points to process per chunk 
            when computing pairwise distances. Reduces memory overhead. Default is 1024.

    Returns:
        hd95 (torch.Tensor): Scalar tensor representing the HD95 distance.
    """
    device = pred.device
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)

    # surface extraction
    pred_eroded = F.conv3d(pred.float(), kernel, padding=1) == 27
    target_eroded = F.conv3d(target.float(), kernel, padding=1) == 27
    pred_surface = pred.bool() & (~pred_eroded)
    target_surface = target.bool() & (~target_eroded)

    # get surface coordinates
    pred_pts = torch.nonzero(pred_surface[0, 0], as_tuple=False).float()
    target_pts = torch.nonzero(target_surface[0, 0], as_tuple=False).float()

    def compute_min_dists(A, B, chunk_size=1024):
        min_dists = []
        for i in range(0, A.shape[0], chunk_size):
            chunk = A[i:i + chunk_size]
            dists = torch.cdist(chunk.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
            min_d = dists.min(dim=1).values
            min_dists.append(min_d)
        return torch.cat(min_dists, dim=0)

    # compute nearest distances by chunks
    pred_to_target = compute_min_dists(pred_pts, target_pts, chunk_size)
    target_to_pred = compute_min_dists(target_pts, pred_pts, chunk_size)

    hd95 = max(
        torch.quantile(pred_to_target, 0.95),
        torch.quantile(target_to_pred, 0.95)
    )

    #if return_asd:
    #    asd = 0.5 * (pred_to_target.mean() + target_to_pred.mean())
    #    return hd95, asd

    return hd95

def pct_negative_jdet(disp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the percentage of negative Jacobian determinant values in a displacement field.

    Args:
        disp (torch.Tensor): Displacement field of shape (B, 3, D, H, W) in voxel space.
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-6.

    Returns:
        penalty (torch.Tensor): Scalar tensor representing the percentage of negative Jacobian determinants.
    """
    jac = jacobian(disp[:, [2, 1, 0]]) ### dim reordering
    det = jacobian_det(jac)
    penalty = F.relu(-det)
    penalty = ((penalty > 0).float()).sum() / (det.numel() + eps)
    
    return penalty
