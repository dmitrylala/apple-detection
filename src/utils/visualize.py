from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid


def show(imgs, figsize=(12, 7), save=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)  # (width, height) in inches
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
    plt.axis('off')

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)

    plt.show()


def draw_predicts(
        apple_data: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        proba_threshold=0.5,
        confidence_threshold=0.5,
        bbox_width=4,
        bbox_color="red",
        mask_alpha=0.7
):
    apples_visualization = []

    for img, src_target in apple_data:
        if not src_target:
            apples_visualization.append((255 * img).to(torch.uint8))
            continue

        target = {key: val.clone() for key, val in src_target.items()}
        if 'scores' in target:
            indices = target['scores'] > confidence_threshold
            target['boxes'] = target['boxes'][indices]
            target['masks'] = target['masks'][indices]

        # converting masks to boolean and image to uint8
        res = (255 * img).to(torch.uint8)
        bool_masks = target['masks'] > proba_threshold
        bool_masks = bool_masks.squeeze(1)

        if torch.any(bool_masks):
            res = draw_segmentation_masks(res, masks=bool_masks, alpha=mask_alpha)
            if 'boxes' in target:
                res = draw_bounding_boxes(res, target["boxes"], colors=bbox_color, width=bbox_width)

        apples_visualization.append(res)

    if len(apples_visualization) == 1:
        return apples_visualization[0]
    return apples_visualization


def visualize_apples(
        apple_data: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        proba_threshold=0.5,
        bbox_width=4,
        bbox_color="red",
        mask_alpha=0.7,
        figsize=(12, 7),
        nrow=2,
        save=None
):
    apples_visualization = draw_predicts(apple_data, proba_threshold, bbox_width, bbox_color, mask_alpha)
    show(make_grid(apples_visualization, nrow=nrow), figsize=figsize, save=save)
