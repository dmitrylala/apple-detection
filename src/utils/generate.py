from typing import Tuple

import cv2

from .utils import mask_2_base64


def get_masks_info(nonzero_masks):
    masks_info = {
        'objects': []
    }
    for mask in nonzero_masks:
        x, y, w, h = cv2.boundingRect(mask)
        mask_info = {
            'bitmap': {
                'origin': [x, y],
                'data': mask_2_base64(mask[y: y + h, x: x + w])
            }
        }
        masks_info['objects'].append(mask_info)
    return masks_info


def get_strides(img_shape: Tuple[int, int], overlapping: Tuple[float, float], kernel_size: Tuple[int, int]):
    h, w = img_shape
    ker_h, ker_w = kernel_size
    perc_h, perc_w = overlapping

    overlap_min_h = int(ker_h * perc_h)
    overlap_min_w = int(ker_w * perc_w)

    stride_h = 0
    for overlap_h in range(overlap_min_h, ker_h):
        n_patches = (h - overlap_h) / (ker_h - overlap_h)
        if n_patches.is_integer():
            stride_h = ker_h - overlap_h
            break

    stride_w = 0
    for overlap_w in range(overlap_min_w, ker_w):
        n_patches = (w - overlap_w) / (ker_w - overlap_w)
        if n_patches.is_integer():
            stride_w = ker_w - overlap_w
            break

    return stride_h, stride_w


def get_n_patches(img_shape: Tuple[int, int], strides: Tuple[int, int], kernel_size: Tuple[int, int]):
    return tuple([
            int((img_shape[i] - kernel_size[i] + strides[i]) / strides[i]) for i in range(2)
        ])


def get_actual_overlap(strides: Tuple[int, int], kernel_size: Tuple[int, int]):
    return tuple([(kernel_size[i] - strides[i]) / kernel_size[i] for i in range(2)])
