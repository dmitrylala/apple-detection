import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype

from src.utils import base64_2_mask

STAVSAD_MODES = ('train', 'val', 'all')
STAVSAD_CUT = ('train_cut', 'val_cut', 'all_cut')
STAVSAD_PATCHES = (
    'train_patches', 'train_cut_patches',
    'val_patches', 'val_cut_patches',
    'all_patches', 'all_cut_patches',
    'all_patches_pred'
)
STAVSAD_ALL = STAVSAD_MODES + STAVSAD_CUT + STAVSAD_PATCHES

NUM_APPLES_THRES = 5


class StavsadApples(Dataset):
    """
    Dataset with stavsad apples
    """
    def __init__(
        self, root: str, mode: str, transforms=None, thres_apples=NUM_APPLES_THRES,
        masks_dir: str = "masks"    
    ):
        """
        Constructor of stavsad apples dataset
        :param root:
        :param mode:
        :param transforms:
        """
        if mode not in os.listdir(root):
            raise ValueError(f"No folder for such mode: {mode}")

        self.root = os.path.join(root, mode)
        self.image_folder = os.path.join(self.root, "images")
        self.masks_folder = os.path.join(self.root, masks_dir)

        json_path = os.path.join(self.root, mode + '.json')
        with open(json_path) as f:
            gt_json = json.load(f)

        # samples - list of tuples (image_path, List[(origin, encoded_bitmap_data)])
        self.samples = []
        for key, value in gt_json.items():
            img_path = os.path.join(self.image_folder, key)
            mask_path = os.path.join(self.masks_folder, value)

            with open(mask_path) as f:
                mask_json = json.load(f)

            mask_data = []
            for obj_data in mask_json['objects']:
                mask_data.append((
                    tuple(obj_data['bitmap']['origin']),
                    obj_data['bitmap']['data']
                ))

            if len(mask_data) > thres_apples:
                self.samples.append((img_path, mask_data))

        self.transforms = transforms

    def __getitem__(self, idx):
        # getting image and mask_json_data
        img_path, mask_data = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        masks, bboxes = [], []
        for origin, encoded_data in mask_data:
            origin_x, origin_y = origin
            mask = np.zeros(image.size[::-1], dtype=np.uint8)

            bitmap = base64_2_mask(encoded_data)
            height, width = bitmap.shape

            # putting bitmap to empty mask using origin
            mask[origin_y: origin_y + height, origin_x: origin_x + width] = bitmap
            masks.append(mask)

            # bbox coords in format [xmin, ymin, xmax, ymax]
            bboxes.append([origin_x, origin_y, origin_x + width, origin_y + height])
        masks = np.array(masks, dtype=np.uint8)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if bboxes.sum() == 0:
            area = 0
        else:
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        image = pil_to_tensor(image)
        image = convert_image_dtype(image, torch.float32)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_img_name(self, idx):
        return Path(self.samples[idx][0]).stem
