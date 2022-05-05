import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from Trainer import Trainer
from datasets import StavsadApples
from utils import read_config
from utils.visualize import draw_predicts


def show_stavsad_predictions(cfg_path: str):
    cfg = get_train_config(cfg_path)
    device = torch.device(cfg.device)
    model = Trainer.get_model(cfg.model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    ds = StavsadApples(cfg.data_dir, mode='all')
    images = [ds[i][0] for i in range(5)]

    # getting predictions
    predictions = []
    with torch.no_grad():
        for img in tqdm(images):
            prediction = model([img.to(device)])[0]
            predictions.append(prediction)

    vis_data = list(zip(images, predictions))
    apples_visualization = draw_predicts(vis_data, confidence_threshold=cfg.confidence_threshold)

    for i, predict_vis in enumerate(apples_visualization):
        predict_vis = predict_vis.detach()
        predict_vis = to_pil_image(predict_vis)
        predict_vis.save(os.path.join(cfg.out_dir, f'stavsad_result{i}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stavsad apples predictions visualization')
    parser.add_argument('cfg_path', help='Path to eval.yaml')
    args = parser.parse_args()
    show_stavsad_predictions(**vars(args))
