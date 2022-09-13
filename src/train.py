import argparse

from Trainer import Trainer
from utils import TrainConfig


def train(cfg_path: str):
    cfg = TrainConfig(cfg_path)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('cfg_path', help='Path to train_cfg.yaml')
    args = parser.parse_args()
    train(**vars(args))
