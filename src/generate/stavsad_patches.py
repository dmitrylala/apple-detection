import argparse
import json
import os

import torch
from torchvision.transforms.functional import to_pil_image
from alive_progress import alive_bar

from src.datasets import StavsadApples, FUJI_IMSIZE, STAVSAD_MODES, STAVSAD_CUT
from src.utils.generate import get_masks_info, get_n_patches, get_strides
from src.utils.patches import patchify

KERNEL_SIZE = (FUJI_IMSIZE, FUJI_IMSIZE)
MIN_OVERLAPPING = {
    (3000, 4000): (0.3, 0.9),  # n_patches=(5, 32), actual_overlap will be (0.517578125, 0.90625)
    (3000, 2700): (0.3, 0.5)  # n_patches=(5, 5), actual_overlap will be (0.517578125, 0.5908203125)
}
IMG_EXT = '.jpg'
MASK_EXT = '.json'


def process(image_name, image, masks, image_folder, masks_folder, strides, n_patches):
    # convert image and masks to batch with size 1
    image_b = image.unsqueeze(0)
    masks_b = masks.unsqueeze(0)

    # dividing into patches
    image_patches = patchify(image_b, KERNEL_SIZE, strides)
    masks_patches = patchify(masks_b, KERNEL_SIZE, strides)

    data_info_part = {}
    with alive_bar(len(image_patches), title=f"Image {image_name}") as bar:
        for j, (img_patch, masks_patch) in enumerate(zip(image_patches, masks_patches)):
            patch_name = image_name + f'_{j}'
            img_name = patch_name + IMG_EXT
            mask_json_name = patch_name + MASK_EXT

            # save image patch
            patch_path = os.path.join(image_folder, img_name)
            to_pil_image(img_patch).save(patch_path)

            # get and save nonzero masks patches
            nonzero_masks = masks_patch[torch.sum(masks_patch, dim=(1, 2)) != 0, :, :].numpy()

            patch_info = get_masks_info(nonzero_masks)
            patch_info['global_offset'] = [(j % n_patches[1]) * strides[1], (j // n_patches[1]) * strides[0]]
            patch_json_path = os.path.join(masks_folder, mask_json_name)

            with open(patch_json_path, 'w') as f:
                json.dump(patch_info, f)

            data_info_part[img_name] = mask_json_name 
            bar()

    return data_info_part


def main(root_ds: str, src_mode: str):
    if src_mode not in STAVSAD_MODES + STAVSAD_CUT:
        raise ValueError(f"Bad mode: {src_mode}")

    ds = StavsadApples(root_ds, mode=src_mode)
    mode = src_mode + '_patches'

    img_size = ds[0][0].shape[1:]
    strides = get_strides(img_size, MIN_OVERLAPPING[img_size], KERNEL_SIZE)
    n_patches = get_n_patches(img_size, strides, KERNEL_SIZE)

    # creating directories for patches
    root = os.path.join(root_ds, mode)
    image_folder = os.path.join(root, "images")
    masks_folder = os.path.join(root, "masks")
    os.makedirs(root, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    # data_info['image name'] = 'mask json name'
    data_info = {}
    json_path = os.path.join(root, mode + '.json')

    args = (
        (ds.get_img_name(i), image, target['masks'], image_folder, masks_folder, strides, n_patches)
        for i, (image, target) in enumerate(ds)
    )
    for inputs in args:
       data_info_part = process(*inputs)
       data_info.update(data_info_part) 

    with open(json_path, 'w') as f:
        json.dump(data_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get patches from stavsad dataset")
    parser.add_argument('root_ds', type=str, help="Path to 'stavsad' directory")
    parser.add_argument('-m', dest='src_mode', type=str, help="'train', 'train_cut', 'val', 'val_cut', 'all' or 'all_cut'")
    args = parser.parse_args()
    main(**vars(args))
