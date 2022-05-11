import argparse
import json
import os

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.datasets import StavsadApples, STAVSAD_MODES
from src.utils.generate import get_masks_info

IMG_EXT = '.jpg'
MASK_EXT = '.json'
THRES = 1300


def main(root_ds: str, src_mode: str):
    if src_mode not in STAVSAD_MODES:
        raise ValueError(f"Bad mode: {src_mode}")

    ds = StavsadApples(root_ds, mode=src_mode)
    mode = src_mode + '_cut'

    # creating directories for cut images
    root = os.path.join(root_ds, mode)
    image_folder = os.path.join(root, "images")
    masks_folder = os.path.join(root, "masks")
    os.makedirs(root, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    # data_info['image name'] = 'mask json name'
    data_info = {}
    json_path = os.path.join(root, mode + '.json')

    for i, (image, target) in tqdm(enumerate(ds)):
        name = ds.get_img_name(i) + '_cut'
        masks = target['masks']

        img_name = name + IMG_EXT
        mask_json_name = name + MASK_EXT

        # save image cut
        img_path = os.path.join(image_folder, img_name)
        to_pil_image(image[:, :, THRES:]).save(img_path)

        # get and save nonzero masks cut
        nonzero_masks = masks[torch.sum(masks[:, :, THRES:], dim=(1, 2)) != 0, :, :][:, :, THRES:].numpy()

        masks_info = get_masks_info(nonzero_masks)
        patch_json_path = os.path.join(masks_folder, mask_json_name)
        with open(patch_json_path, 'w') as f:
            json.dump(masks_info, f)

        data_info[img_name] = mask_json_name

    with open(json_path, 'w') as f:
        json.dump(data_info, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get cut images from stavsad dataset")
    parser.add_argument('root_ds', help="Path to 'stavsad' directory")
    parser.add_argument('src_mode', help="'train', 'val' or 'all'")
    args = parser.parse_args()
    main(**vars(args))
