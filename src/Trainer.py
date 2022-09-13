import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datasets import FujiApples, StavsadApples
from models import maskrcnn_resnet50_fpn, maskrcnn_resnet101_fpn
from utils import collate_fn
from utils.engine import train_one_epoch, evaluate
from utils.transforms import Compose, RandomHorizontalFlip, Normalize
from utils.visualize import draw_predicts

TRAIN_PRINT_FREQ = 100


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.out_dir = os.path.join(cfg.out_dir, cfg.model_name)
        os.makedirs(self.out_dir, exist_ok=True)

        log_dir = os.path.join(self.out_dir, "runs", cfg.unique_name)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("Config", str(self.cfg), global_step=None, walltime=None)

    def train(self):
        model = self.get_model(self.cfg.model_name).to(self.cfg.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._get_optimizer(params, self.cfg.optimizer, self.cfg.lr, self.cfg.weight_decay)

        lr_scheduler = None
        if self.cfg.scheduler != "None":
            lr_scheduler = self._get_scheduler(
                optimizer,
                self.cfg.scheduler,
                self.cfg.milestones,
                self.cfg.sch_gamma,
                self.cfg.patience
            )

        train_ds, val_ds = self.get_datasets(self.cfg.data_dir, self.get_augs(self.cfg.augs))
        train_dl, val_dl = self.get_dataloaders(
            train_ds, val_ds, self.cfg.batch_size, self.cfg.num_workers
        )

        # creating directory for model weights
        weights_dir = os.path.join(self.out_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"Epoch: {epoch}")

            # training and logging
            metric_logger = train_one_epoch(
                model, optimizer, train_dl, self.cfg.device, epoch, print_freq=TRAIN_PRINT_FREQ
            )
            self._log_train(metric_logger, epoch)

            if lr_scheduler is not None:
                lr_scheduler.step()

            # evaluation and logging
            coco_evaluator = evaluate(model, val_dl, self.cfg.device)
            self._log_val(coco_evaluator, epoch)

            # logging predictions on selected images from validation set
            if hasattr(self.cfg, "val_images"):
                self._log_images(model, val_ds, epoch)

            # saving weights
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, f"{self.cfg.unique_name}_{epoch}.pth")
            )

    def _log_train(self, metric_logger, epoch):
        for name, meter in metric_logger.meters.items():
            if "loss" not in name:
                self.writer.add_scalar(name, meter.value, epoch)
                continue
            if name == "loss":
                self.writer.add_scalar(name + "/total", meter.value, epoch)
                continue
            loss_type = name[5:]
            self.writer.add_scalar("loss/" + loss_type, meter.value, epoch)

    def _log_val(self, coco_evaluator, epoch):
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

        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            for i, name in zip(indices, metric_names):
                self.writer.add_scalar(f"{iou_type}/{name}", coco_eval.stats[i], epoch)

    def _log_images(self, model, ds, epoch):
        model.eval()
        confidence_thresholds = [0.0, 0.5, 0.8]
        for i, (image, target) in enumerate(ds):
            img_name = ds.get_img_name(i)
            if img_name not in self.cfg.val_images:
                continue

            prediction = model([image.to(self.cfg.device)])[0]
            drawn_gt = draw_predicts([(image, target)]).detach()
            for conf in confidence_thresholds:
                drawn_pred = draw_predicts([(image, prediction)], confidence_threshold=conf).detach()
                grid = make_grid([drawn_gt, drawn_pred], nrow=2)
                self.writer.add_image(f"conf{conf:.1f}/" + img_name, grid, epoch)

    @staticmethod
    def get_model(model_name, pretrained=True):
        print(f"Loading {model_name} model")
        if model_name == "MaskRCNN_resnet50_fpn":
            return maskrcnn_resnet50_fpn(num_classes=2, pretrained=pretrained)
        elif model_name == "MaskRCNN_resnet101_fpn":
            return maskrcnn_resnet101_fpn(num_classes=2, pretrained=pretrained)
        raise ValueError(f"{model_name} model isn't supported by now")

    @staticmethod
    def _get_optimizer(parameters, optimizer_name, lr, weight_decay):
        if optimizer_name == "Adam":
            return torch.optim.Adam(
                parameters, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(
                parameters, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                parameters, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"{optimizer_name} optimizer isn't supported by now")

    @staticmethod
    def _get_scheduler(optimizer, scheduler_name, milestones, sch_gamma, patience):
        if scheduler_name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
        elif scheduler_name == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=sch_gamma)
        elif scheduler_name == "None":
            return None
        raise ValueError(f"{scheduler_name} scheduler isn't supported by now")

    @staticmethod
    def get_augs(augs_names: List[str]):
        augs = {
            "RandomHorizontalFlip": RandomHorizontalFlip(0.5),
            "Normalize": Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        }
        transforms = Compose([augs[aug_name] for aug_name in augs_names])
        return transforms

    @staticmethod
    def get_datasets(data_dir, augs=None):
        print(f"Loading {data_dir} dataset")

        if 'fuji' in data_dir:
            train_ds = FujiApples(data_dir, "train", transforms=augs)
            val_ds = FujiApples(data_dir, "val")
        elif 'stavsad' in data_dir:
            train_ds = StavsadApples(data_dir, "train_cut_patches", transforms=augs)
            val_ds = StavsadApples(data_dir, "val_cut_patches")
        else:
            raise ValueError(f"No such dataset: {data_dir}")

        return train_ds, val_ds

    @staticmethod
    def get_dataloaders(train_ds, val_ds, train_batch_size, num_workers):
        train_dl = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )

        return train_dl, val_dl
