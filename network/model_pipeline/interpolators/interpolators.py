import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay
import math
from typing import Tuple


class LinearInterpolation3d(nn.Module):
    """
    Performs linear interpolation of sparse 3D displacements onto a dense grid using Delaunay triangulation.

    Based on:
    'Driving Points Prediction for Abdominal Probabilistic Registration'
    https://github.com/SamuelJoutard/DrivingPointsPredictionMIR/blob/01e3dd8c4188e70a6113209335f2ecaf1ce0a75d/models.py#L685
    """
    def __init__(self, size: Tuple[int, int, int]):
        """
        Args:
            size (Tuple[int, int, int]): Output volume size (D, H, W).
        """
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
    
    def _get_barycentric_coordinates(self, points_tri: Delaunay, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes barycentric coordinates of each target point in the Delaunay triangulation.

        Args:
            points_tri (Delaunay): Delaunay triangulation of input points.
            target (np.ndarray): Target points of shape (M, D).

        Returns:
            coord (np.ndarray): Barycentric coordinates for each target point.
            s (np.ndarray): Indices of the simplex each target lies in.
        """
        s = points_tri.find_simplex(target)
        dim = target.shape[1]
        
        b0 = (points_tri.transform[s, :dim].transpose([1, 0, 2]) *
            (target - points_tri.transform[s, dim])).sum(axis=2).T
        coord = np.c_[b0, 1 - b0.sum(axis=1)]

        return coord, s

    def _linear_interp_material(self, points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares Delaunay interpolation data.

        Args:
            points (np.ndarray): Known points (N, D).
            target (np.ndarray): Target query points (M, D).

        Returns:
            simplices (np.ndarray): Indices of simplices.
            coords (np.ndarray): Barycentric coordinates of targets.
            s (np.ndarray): Simplex indices for targets.
        """
        points_triangulated = Delaunay(points)
        c, s = self._get_barycentric_coordinates(points_triangulated, target)
        simplices = points_triangulated.simplices
        return simplices, c, s
    
    def _linear_interp(self, points: torch.Tensor, values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Performs linear interpolation from sparse `points` to `target` locations.

        Args:
            points (torch.Tensor): Known coordinates of shape (B, N, D).
            values (torch.Tensor): Values at known points of shape (B, N, C).
            target (torch.Tensor): Query locations of shape (B, M, D).

        Returns:
            res (torch.Tensor): Interpolated values at target locations (B, M, C).
        """
        device = points.device
        B = points.size(0)

        if B>1:
            raise NotImplementedError("Linear interpolation not implemented for batches larger than 1.")

        points_np = points.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
            
        simplices, coords, s = self._linear_interp_material(points_np[0], target_np[0])
        simplices = torch.tensor(simplices).long().to(device)
        coords = torch.tensor(coords).float().to(device)
        s = torch.tensor(s).long().to(device)

        res = (values[0, simplices[s]] * coords[:, :, None]).sum(1)
            
        return res[None, :]
    
    def forward(self, kpts: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Interpolates sparse displacements defined at `kpts` over a dense 3D grid.

        Args:
            kpts (torch.Tensor): Keypoint coordinates of shape (B, N, 3).
            disp (torch.Tensor): Displacements at keypoints of shape (B, N, 3).

        Returns:
            torch.Tensor: Dense displacement field of shape (B, 3, D, H, W).
        """
        kpts_pad = torch.cat([kpts, self.pads], dim=1)
        disp_pad = torch.cat([disp, self.pads_values], dim=1)
        interp = self._linear_interp(kpts_pad, disp_pad, self.grid)
        return torch.reshape(interp, (kpts.size(0),)+self.size+(3,)).permute(0, 4, 1, 2, 3)
    

class ThinPlateSpline(nn.Module):
    """
    Thin Plate Spline (TPS) interpolation layer for 3D displacement fields.

    Based on:
    'VoxelMorph++' - https://github.com/mattiaspaul/VoxelMorphPlusPlus/blob/0f8da77b4d5bb4df80d188188df9725013bb960b/src/utils_voxelmorph_plusplus.py#L271
    """
    def __init__(self, shape: Tuple[int, int, int], step: int = 4, lambd: float = 0.1, unroll_step_size: int = 2**12):
        """
        Args:
            shape (Tuple[int, int, int]): Output 3D volume shape (D, H, W).
            step (int): Downsampling factor for initial interpolation grid.
            lambd (float): Regularization coefficient for TPS fitting.
            unroll_step_size (int): Chunk size for applying TPS transformation.
        """
        super().__init__()
        self.shape = shape
        self.step = step
        self.lambd = lambd
        self.unroll_step_size = unroll_step_size

        # Precompute the identity affine grid for interpolation
        D1, H1, W1 = [s // step for s in shape]
        grid = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0),
            size=(1, 1, D1, H1, W1),
            align_corners=True
        )
        self.register_buffer("base_grid", grid.view(-1, 3))  # Flattened 3D grid

    def forward(self, kpts: torch.Tensor, disps: torch.Tensor) -> torch.Tensor:
        """
        Apply TPS to interpolate displacements from sparse keypoints to a dense grid.

        Args:
            kpts (torch.Tensor): Source keypoints of shape (1, N, 3).
            disps (torch.Tensor): Displacements at keypoints of shape (1, N, 3).

        Returns:
            y2 (torch.Tensor): Dense displacement field of shape (1, 3, D, H, W).
        """
        x1 = kpts[0]  # (N, 3)
        y1 = disps[0]  # (N, 3)
        x2 = self.base_grid

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

    def _fit(self, c: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Fit TPS parameters.

        Args:
            c (torch.Tensor): Control points of shape (N, 3).
            f (torch.Tensor): Displacements at control points (N, 3).

        Returns:
            theta (torch.Tensor): TPS coefficients of shape (N+4, 3).
        """
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

    def _z(self, x: torch.Tensor, c: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply TPS transformation to new points.

        Args:
            x (torch.Tensor): Query points of shape (M, 3).
            c (torch.Tensor): Control points of shape (N, 3).
            theta (torch.Tensor): TPS coefficients of shape (N + 4, 3).

        Returns:
            torch.Tensor: Transformed points (M, 3).
        """
        U = self._u(self._d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)  # (M, 3)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()

    def _d(self, a, b):
        """
        Compute pairwise Euclidean distances between sets of points.

        Args:
            a (torch.Tensor): Set of query points (M, 3)
            b (torch.Tensor): Set of control points (N, 3)

        Returns:
            torch.Tensor: Distance matrix of shape (M, N)
        """
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.T)
        return torch.sqrt(torch.clamp(dist, min=0.0))

    def _u(self, r: torch.Tensor) -> torch.Tensor:
        """
        Radial basis function used in TPS.

        Args:
            r (torch.Tensor): Distance matrix (M, N).

        Returns:
            torch.Tensor: Radial basis matrix (M, N).
        """
        return (r ** 2) * torch.log(r + 1e-6)
