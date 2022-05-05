import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from Trainer import Trainer
from utils import get_config
from utils.visualize import draw_predicts


def compare_predictions(cfg_path: str):
    cfg = get_config(cfg_path)
    device = torch.device(cfg.device)

    model_base = Trainer.get_model(cfg.model_name, pretrained=False).to(device)
    model_base.load_state_dict(torch.load(cfg.weights_path_base, map_location=device))
    model_base.eval()

    model_prop = Trainer.get_model(cfg.model_name, pretrained=False).to(device)
    model_prop.load_state_dict(torch.load(cfg.weights_path_proposed, map_location=device))
    model_prop.eval()

    val_dl = Trainer.get_stavsad_dataloaders(
        cfg.data_dir, cfg.mode, 1, cfg.num_workers,
        Trainer.get_augs(train=True), Trainer.get_augs(train=False)
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    # getting predictions
    with torch.no_grad():
        for i, (image, target) in tqdm(enumerate(val_dl)):
            image = image[0]
            target = target[0]
            if i >= cfg.batch_num:
                break
            images = [image, image, image]
            predictions = [target, model_base([image.to(device)])[0], model_prop([image.to(device)])[0]]
            vis_data = list(zip(images, predictions))
            apples_visualization = draw_predicts(vis_data, confidence_threshold=cfg.confidence_threshold)
            grid = to_pil_image(make_grid(apples_visualization, nrow=3).detach())
            grid.save(os.path.join(cfg.out_dir, f'vis_{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apples predictions visualization')
    parser.add_argument('cfg_path', help='Path to compare.yaml')
    args = parser.parse_args()
    compare_predictions(**vars(args))
