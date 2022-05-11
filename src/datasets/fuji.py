import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype

FUJI_IMSIZE = 1024
NUM_APPLES_THRES = 1


class FujiApples(Dataset):
    """
    Dataset with Fuji apples
    """
    def __init__(self, root: str, mode: str, transforms=None, thres_apples=NUM_APPLES_THRES):
        """
        Constructor of fuji apples dataset
        :param root: fuji apples dir with "images" and "masks" folders
        :param mode: "train" or "val"
        :param transforms: transformations performed; expects tensor as input
        """
        if mode not in ["train", "val"]:
            raise ValueError(f"Bad mode: {mode}")

        if mode not in os.listdir(root):
            raise ValueError(f"No folder for such mode: {mode}")

        self.root = os.path.join(root, mode)
        self.image_folder = os.path.join(self.root, "images")
        self.masks_folder = os.path.join(self.root, "masks")

        json_path = os.path.join(self.root, mode + '.json')
        with open(json_path) as f:
            gt_json = json.load(f)

        # samples - list of tuples (image_path, mask_path)
        self.samples = []
        for key, value in gt_json.items():
            img_path = os.path.join(self.image_folder, key)
            mask_path = os.path.join(self.masks_folder, value)

            # adding to set only samples with at least 'thres_apples' apples
            mask_csv = pd.read_csv(mask_path)
            if mask_csv.shape[0] >= thres_apples:
                self.samples.append((img_path, mask_path))

        self.transforms = transforms

    def __getitem__(self, idx):
        # getting image and mask_csv
        img_path, mask_csv_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize((FUJI_IMSIZE, FUJI_IMSIZE), Image.ANTIALIAS)
        mask_csv = pd.read_csv(mask_csv_path)

        # converting mask from polygons to set of binary masks and getting bounding boxes
        masks, bboxes = [], []
        for attributes_str in mask_csv["region_shape_attributes"]:
            attributes = json.loads(attributes_str)
            points_x, points_y = attributes["all_points_x"], attributes["all_points_y"]
            polygon = list(zip(points_x, points_y))

            # draw polygon
            mask = Image.new("L", image.size)
            drawing = ImageDraw.Draw(mask)
            drawing.polygon(polygon, fill=1, outline=1)

            # bbox coords in format [xmin, ymin, xmax, ymax]
            # saving only correct bboxes
            if max(points_x) > min(points_x) and max(points_y) > min(points_y):
                bboxes.append([min(points_x), min(points_y), max(points_x), max(points_y)])
                masks.append(np.array(mask))
        masks = np.array(masks, dtype=np.uint8)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((len(bboxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
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

    def get_img_name(self, idx) -> str:
        return Path(self.samples[idx][0]).stem
