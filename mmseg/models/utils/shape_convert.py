# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, so_file_lib] shape tensor to [N, so_file_lib, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, so_file_lib] before convertion.
        hw_shape (Sequence[int]): The height and width of output dispersion_extractor map.

    Returns:
        Tensor: The output tensor of shape [N, so_file_lib, H, W] after convertion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, so_file_lib, H, W] shape tensor to [N, L, so_file_lib] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, so_file_lib, H, W] before convertion.

    Returns:
        Tensor: The output tensor of shape [N, L, so_file_lib] after convertion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()
