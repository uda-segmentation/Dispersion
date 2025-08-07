# ---------------------------------------------------------------
# Anonymous submission for AAAI 2026.
# Paper title: "Unsupervised Domain Adaptation for Semantic Segmentation Based on Instance Directional Dispersion"
# No author or affiliation information included in this version.
# ---------------------------------------------------------------

import os
from typing import Tuple, Optional

import torch
import numpy as np
from ctypes import cdll, c_int, c_float, POINTER


class DispersionExtractor:
    """
    Wrapper around the external .so library to compute the dispersion feature
    from segmentation and depth maps. The input is expected to be a 2D segmentation
    label map and corresponding depth map; output is a single-channel dispersion map.

    The underlying C function signature is assumed to be:
        void get_src(int* seg, float* depth, float* out);
    where all arrays have H * W elements in row-major order.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        so_path: Optional[str] = None,
    ):
        """
        Args:
            crop_size: (H, W) spatial size expected by the native library.
            so_path: filesystem path to the shared object (.so) file.
        """
        self.H, self.W = crop_size
        self.N = self.H * self.W

        if so_path is None:
            raise ValueError("so_path must be provided to load the dispersion library.")
        if not os.path.isfile(so_path):
            raise FileNotFoundError(f"Shared library not found at: {so_path}")

        # Load the shared library once and set argument types.
        self._lib = cdll.LoadLibrary(so_path)
        # Define expected function signature: get_src(int* seg, float* depth, float* out)
        self._lib.get_src.argtypes = (
            POINTER(c_int * self.N),
            POINTER(c_float * self.N),
            POINTER(c_float * self.N),
        )
        self._out_buf = (c_float * self.N)()

    def _compute_single(self, seg_hw: torch.Tensor, depth_hw: torch.Tensor) -> torch.Tensor:
        """
        Compute dispersion for one image pair.

        Args:
            seg_hw: [H, W] integer segmentation labels (torch tensor).
            depth_hw: [H, W] float depth map (torch tensor).
        Returns:
            Tensor of shape [1, H, W], raw dispersion scaled.
        """
        if seg_hw.shape != (self.H, self.W):
            raise ValueError(f"seg_hw must be ({self.H},{self.W}), got {tuple(seg_hw.shape)}")
        if depth_hw.shape != (self.H, self.W):
            raise ValueError(f"depth_hw must be ({self.H},{self.W}), got {tuple(depth_hw.shape)}")

        # Flatten and convert to appropriate ctypes arrays
        seg_np = seg_hw.cpu().numpy().astype(np.int32).ravel()
        depth_np = depth_hw.cpu().numpy().astype(np.float32).ravel()

        c_seg = (c_int * self.N)(*seg_np)
        c_depth = (c_float * self.N)(*depth_np)

        # Call the external C function
        self._lib.get_src(c_seg, c_depth, self._out_buf)

        # Wrap output and reshape
        out_np = np.ctypeslib.as_array(self._out_buf).reshape(1, self.H, self.W)
        out_tensor = torch.from_numpy(out_np.astype(np.float32))  # [1, H, W]
        return out_tensor

    @torch.no_grad()
    def __call__(
        self,
        seg_maps: torch.Tensor,
        depth_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch computation of dispersion maps.

        Args:
            seg_maps: Tensor of shape [B, H, W] or [B, 1, H, W], integer labels.
            depth_maps: Tensor of shape [B, H, W] or [B, 1, H, W], float depth.
        Returns:
            Tensor of shape [B, 1, H, W]: dispersion maps.
        """
        # Normalize shapes: squeeze possible channel dim
        if seg_maps.dim() == 4 and seg_maps.size(1) == 1:
            seg_maps = seg_maps[:, 0]
        if depth_maps.dim() == 4 and depth_maps.size(1) == 1:
            depth_maps = depth_maps[:, 0]

        B = seg_maps.shape[0]
        outputs = []
        for i in range(B):
            seg_hw = seg_maps[i]
            depth_hw = depth_maps[i]
            disp_raw = self._compute_single(seg_hw, depth_hw)  # [1, H, W]

            outputs.append(disp_raw)

        # Stack to [B, 1, H, W]
        return torch.stack(outputs, dim=0)
