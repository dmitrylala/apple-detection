from typing import Tuple

import torch
from torch.nn.functional import fold


def get_overlapping_mask(
        images_shape: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int]
):
    """
    Get mask of overlapping regions after patching
    :param images_shape: tuple(batch_size, n_channels, height, width)
    :param kernel_size: Size of each patch: (ker_h, ker_w)
    :param strides: Strides for height and width: (stride_h, stride_w)
    :return: mask with weights, shape (batch_size, n_channels, height, width)
    """
    mask_patches = patchify(torch.ones(images_shape), kernel_size, strides)
    mask_unpatched = unpatchify(mask_patches, images_shape, kernel_size, strides)
    return 1.0 / mask_unpatched


def patchify(
        images: torch.Tensor,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int]
):
    """
    Convert image batch to patches
    :param images: Input batch of images, shape: (batch_size, n_channels, height, width)
    :param kernel_size: Size of each patch: (ker_h, ker_w)
    :param strides: Strides for height and width: (stride_h, stride_w)
    :return: extracted patches with shape (n_patches, n_channels, ker_h, ker_w)
    """
    _, c, _, _ = images.shape
    kh, kw = kernel_size
    dh, dw = strides

    patches = images.unfold(1, c, c).unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(-1, c, kh, kw)
    return patches


def unpatchify(
        patches: torch.Tensor,
        output_shape: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int]
):
    """
    Reconstruct image back from patches
    :param patches: Extracted patches with shape (n_patches, n_channels, ker_h, ker_w)
    :param output_shape: Desired shape of output, should be (batch_size, n_channels, height, width)
    :param kernel_size: Size of each patch: (ker_h, ker_w)
    :param strides: Strides for height and width: (stride_h, stride_w)
    :return: Folded patches, applied summation in overlapped regions
    """
    b, c, h, w = output_shape
    kh, kw = kernel_size

    # reshape patches to match F.fold input
    patches = patches.contiguous().view(b, c, -1, kh * kw)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(b, c * kh * kw, -1)  # [B, C*prod(kernel_size), L] as expected by Fold

    output = fold(
        patches, output_size=(h, w), kernel_size=kernel_size, stride=strides
    )
    return output
