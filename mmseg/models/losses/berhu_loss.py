import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module()
class BerHuLoss(nn.Module):
    """BerHu (Inverse Huber) Loss."""
    def __init__(self, ignore_index=1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self,
                prediction: torch.Tensor,
                ground_truth: torch.Tensor,
                imagemask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            prediction:    Tensor[B, 1, H, W] or [B, H, W]
            ground_truth:  Tensor same shape as prediction
            imagemask:     Optional bool Tensor same spatial shape to further mask out pixels
        Returns:
            Scalar loss (averaged over valid pixels)
        """
        # 构造有效像素掩码
        mask = ground_truth != self.ignore_index
        if imagemask is not None:
            mask &= imagemask

        # 绝对误差
        diff = torch.abs(prediction - ground_truth)

        # 计算阈值 c，仅基于有效像素
        if mask.any():
            c = 0.2 * diff[mask].max()
        else:
            return diff.new_tensor(0.)

        # berHu：d ≤ c 用 d，d > c 用 (d² + c²) / (2c)
        # 为避免 c=0 导致除零，加个 eps
        eps = 1e-6
        linear_part = diff
        nonlinear_part = (diff * diff + c * c) / (2 * c + eps)
        loss = torch.where(diff <= c, linear_part, nonlinear_part)

        # 只取有效像素并平均
        valid_loss = loss[mask]
        return valid_loss.mean() if valid_loss.numel() > 0 else diff.new_tensor(0.)
