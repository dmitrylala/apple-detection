import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from Trainer import Trainer
from datasets import FUJI_IMSIZE
from utils import get_config
from utils.visualize import draw_predicts


def show_fuji_predictions(cfg_path: str):
    cfg = get_config(cfg_path)
    device = torch.device(cfg.device)
    model = Trainer.get_model(cfg.model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    transforms = Compose([
        ToTensor(),
        ConvertImageDtype(torch.float32)
    ])

    images = []
    for img_path in cfg.pics_paths:
        image = Image.open(img_path).convert("RGB").resize((FUJI_IMSIZE, FUJI_IMSIZE), Image.ANTIALIAS)
        images.append(transforms(image))

    # getting predictions
    predictions = []
    with torch.no_grad():
        for img in tqdm(images):
            prediction = model([img.to(device)])[0]
            predictions.append(prediction)

    vis_data = list(zip(images, predictions))
    apples_visualization = draw_predicts(vis_data, confidence_threshold=cfg.confidence_threshold)

    if cfg.grid:
        grid = make_grid(apples_visualization, nrow=4)
        grid = grid.detach()
        grid = to_pil_image(grid)
        grid.save(cfg.out_dir + 'grid.png')
    else:
        for img_path, predict in zip(cfg.pics_paths, apples_visualization):
            name, ext = img_path.split('/')[-1].split('.')
            predict = predict.detach()
            predict = to_pil_image(predict)
            predict.save(cfg.out_dir + name + f'_result{cfg.confidence_threshold}' + '.' + ext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuji apples predictions visualization')
    parser.add_argument('cfg_path', help='Path to eval.yaml')
    args = parser.parse_args()
    show_fuji_predictions(**vars(args))
