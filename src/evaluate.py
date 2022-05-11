import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Trainer import Trainer
from datasets import StavsadApples
from utils import collate_fn, EvaluateConfig
from utils.engine import evaluate


def evaluate_metrics(cfg_path: str):
    cfg = EvaluateConfig(cfg_path)
    device = torch.device(cfg.device)
    model = Trainer.get_model(cfg.model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    ds = StavsadApples(cfg.data_dir, cfg.mode)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    metric_names = [
        "AP@[0.5:0.95]_all",
        "AP@[0.5]_all",
        "AP@[0.75]_all",
        "AP@[0.5:0.95]_small",
        "AP@[0.5:0.95]_medium",
        "AP@[0.5:0.95]_large",
        "AR@[0.5:0.95]_all",
        "AR@[0.5:0.95]_small",
        "AR@[0.5:0.95]_medium",
        "AR@[0.5:0.95]_large",
    ]

    results = {}
    coco_evaluator = evaluate(model, dl, cfg.device)
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        for i, name in zip(indices, metric_names):
            results[f"{iou_type}/{name}"] = [coco_eval.stats[i]]

    pd.DataFrame.from_dict(data=results).to_csv(cfg.df_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset metrics evaluation')
    parser.add_argument('cfg_path', help='Path to eval.yaml')
    args = parser.parse_args()
    evaluate_metrics(**vars(args))
