import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
import math

# Code adapted from https://github.com/SamuelJoutard/DrivingPointsPredictionMIR/blob/01e3dd8c4188e70a6113209335f2ecaf1ce0a75d/models.py#L685
#@inbook{Joutard_2022,
#   title={Driving Points Prediction for Abdominal Probabilistic Registration},
#   ISBN={9783031210143},
#   ISSN={1611-3349},
#   url={http://dx.doi.org/10.1007/978-3-031-21014-3_30},
#   DOI={10.1007/978-3-031-21014-3_30},
#   booktitle={Machine Learning in Medical Imaging},
#   publisher={Springer Nature Switzerland},
#   author={Joutard, Samuel and Dorent, Reuben and Ourselin, Sebastien and Vercauteren, Tom and Modat, Marc},
#   year={2022},
#   pages={288–297} 
#}
class LinearInterpolation3d(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1)+size, align_corners=True).view(1, -1, 3)
        self.register_buffer("grid", grid)

        pads = torch.ones((1, 8, 3))
        pads[0, 1, 0] = -1
        pads[0, 2, 1] = -1
        pads[0, 3, 2] = -1
        pads[0, 4, 0] = -1
        pads[0, 4, 1] = -1
        pads[0, 5, 0] = -1
        pads[0, 5, 2] = -1
        pads[0, 6, 1] = -1
        pads[0, 6, 2] = -1
        pads[0, 7, 0] = -1
        pads[0, 7, 1] = -1
        pads[0, 7, 2] = -1
        self.register_buffer("pads", pads)

        pads_values = torch.zeros((1, 8, 3))
        self.register_buffer("pads_values", pads_values)
    
    def _get_barycentric_coordinates(self, points_tri, target):
        s = points_tri.find_simplex(target)
        dim = target.shape[1]
        
        b0 = (points_tri.transform[s, :dim].transpose([1, 0, 2]) *
            (target - points_tri.transform[s, dim])).sum(axis=2).T
        coord = np.c_[b0, 1 - b0.sum(axis=1)]

        return coord, s

    def _linear_interp_material(self, points, target):
        """
        Linearly interpolate signal at target locations
        points: numpy array (N, D)
        target: numpy array (N, D)
        """
        points_triangulated = Delaunay(points)
        c, s = self._get_barycentric_coordinates(points_triangulated, target)
        
        return points_triangulated.simplices, points_triangulated.transform, c, s
    
    def _linear_interp(self, points, values, target):
        """
        points: points where the signal is known; torch tensor (B, N, D)
        values: signal; torch tensor (B, N, C)
        target: where the signal needs to be interpolated; torch tensor (B, M, D)
        """
        device = points.device
        B = points.size(0)

        if B>1:
            raise NotImplementedError("Linear interpolation not implemented for batches larger than 1.")

        points_np = points.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
            
        simplices, T, coords, s = self._linear_interp_material(points_np[0], target_np[0])
        simplices = torch.tensor(simplices).long().to(device)  # n_simplices, D+1
        T = torch.tensor(T).float().to(device) # n_simplices, D+1
        coords = torch.tensor(coords).float().to(device) # M, D+1
        s = torch.tensor(s).long().to(device) # M

        res = (values[0, simplices[s]] * coords[:, :, None]).sum(1) # M,C
            
        return res[None, :]
    
    def forward(self, kpts, disp):
        """
        kpts: B, N, 3
        disp: B, N, 3
        """
        kpts_pad = torch.cat([kpts, self.pads], dim=1)
        disp_pad = torch.cat([disp, self.pads_values], dim=1)
        interp = self._linear_interp(kpts_pad, disp_pad, self.grid)
        return torch.reshape(interp, (kpts.size(0),)+self.size+(3,)).permute(0, 4, 1, 2, 3)
    
# Code adapted from VoxelMorph++ https://github.com/mattiaspaul/VoxelMorphPlusPlus/blob/0f8da77b4d5bb4df80d188188df9725013bb960b/src/utils_voxelmorph_plusplus.py#L271
#@misc{heinrich2022voxelmorphgoingcranialvault,
#      title={Voxelmorph++ Going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation}, 
#      author={Mattias P. Heinrich and Lasse Hansen},
#      year={2022},
#      eprint={2203.00046},
#      archivePrefix={arXiv},
#      primaryClass={cs.CV},
#      url={https://arxiv.org/abs/2203.00046}, 
#}
class ThinPlateSpline(nn.Module):
    def __init__(self, shape, step=4, lambd=0.1, unroll_step_size=2**12):
        super().__init__()
        self.shape = shape  # Output grid shape: (D, H, W)
        self.step = step    # Downsampling step for coarse grid
        self.lambd = lambd
        self.unroll_step_size = unroll_step_size

        # Precompute the identity affine grid for interpolation
        D1, H1, W1 = [s // step for s in shape]
        grid = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0),  # Identity
            size=(1, 1, D1, H1, W1),
            align_corners=True
        )
        self.register_buffer("base_grid", grid.view(-1, 3))  # Flattened 3D grid

    def forward(self, kpts, disps):
        """
        kpts: (1, N, 3) - keypoints from source
        disps: (1, N, 3) - corresponding displacements
        Returns: dense displacement field of shape (1, 3, D, H, W)
        """
        x1 = kpts[0]  # (N, 3)
        y1 = disps[0]  # (N, 3)
        x2 = self.base_grid    # (M, 3) - dense grid to warp

        # Compute TPS parameters
        theta = self._fit(x1, y1)

        # Compute transformed grid
        M = x2.shape[0]
        y2 = torch.zeros((1, M, 3), device=x2.device)

        n_chunks = math.ceil(M / self.unroll_step_size)
        for j in range(n_chunks):
            j1 = j * self.unroll_step_size
            j2 = min((j + 1) * self.unroll_step_size, M)
            y2[0, j1:j2, :] = self._z(x2[j1:j2], x1, theta)

        # Reshape and interpolate back to full resolution
        D1, H1, W1 = [s // self.step for s in self.shape]
        y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)  # (1, 3, D1, H1, W1)
        y2 = F.interpolate(y2, size=self.shape, mode='trilinear', align_corners=True)
        return y2

    def _fit(self, c, f):
        """Compute TPS parameters (theta)"""
        device = c.device
        n = c.shape[0]
        f_dim = f.shape[1]

        U = self._u(self._d(c, c))  # (n, n)
        K = U + torch.eye(n, device=device) * self.lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        return theta

    def _z(self, x, c, theta):
        """Apply TPS transformation"""
        U = self._u(self._d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)  # (M, 3)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()

    def _d(self, a, b):
        """Pairwise Euclidean distances"""
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.T)
        return torch.sqrt(torch.clamp(dist, min=0.0))

    def _u(self, r):
        """Radial basis function for TPS"""
        return (r ** 2) * torch.log(r + 1e-6)
