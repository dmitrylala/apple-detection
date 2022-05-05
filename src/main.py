import pickle
from pathlib import Path

import torch
from torchsummary import summary

from datasets import StavsadApples, FujiApples
from models import maskrcnn_resnet50_fpn, maskrcnn_resnet101_fpn


def main():

    fuji_root, stavsad_root = Path("data/fuji"), Path("data/stavsad")

    fuji_ds = FujiApples(fuji_root, "train")
    # stavsad_ds = StavsadApples(stavsad_root, "all_cut_patches")
    # print(len(fuji_ds), len(stavsad_ds))

    maskrcnn_101 = maskrcnn_resnet101_fpn(num_classes=2, pretrained=True)
    maskrcnn_101.train()
    image, target = fuji_ds[0]
    lol = maskrcnn_101([image], [target])
    print(maskrcnn_101)
    # print(summary(maskrcnn_101, (3, 1024, 1024), device="cpu"))

    # images = ["_MG_8065_08"]
    # for i, (img, target) in enumerate(fuji_ds):
    #     name = fuji_ds.get_img_name(i)
    #     if fuji_ds.get_img_name(i) in images:
    #         to_pil_image(draw_predicts([(img, target)])).save(pics_path / (name + "_gt.png"))
    #         to_pil_image(img).save(pics_path / (name + "_img.png"))

    # augs = Trainer.get_augs(train=False)
    # ds_cut = StavsadApples(root, 'all_cut_patches', augs)
    #
    # for i, (image, target) in enumerate(ds_cut):
    #     if ds_cut.get_img_name(i) == '00029_cut_8':
    #         to_pil_image(draw_predicts([(image, target)])).show()

    # ds_fuji_train = FujiApples(fuji_root, 'train')
    # ds_fuji_val = FujiApples(fuji_root, 'val')
    #
    # # ds_stav_all = StavsadApples(stavsad_root, 'all')
    # # ds_stav_all_cut = StavsadApples(stavsad_root, 'all_cut')
    # ds_stav_all_cut_patches = StavsadApples(stavsad_root, 'all_cut_patches')
    #
    # ds_sets = [[ds_fuji_train, ds_fuji_val], [ds_stav_all_cut_patches]]
    # names = ["fuji", "stavsad cut and patched"]
    # for ds_set, name in zip(ds_sets, names):
    #     areas = []
    #     for ds in ds_set:
    #         for _, target in tqdm(ds):
    #             areas += list(target["area"].numpy())
    #     areas = np.array(areas)
    #     areas_count = {
    #         "small": np.sum(areas < 32 ** 2),
    #         "medium": np.sum((areas > 32 ** 2) * (areas < 96 ** 2)),
    #         "large": np.sum(areas > 96 ** 2)
    #     }
    #     print(f"{name}, {areas_count}")
    #     plt.hist(areas, bins=100)
    #     plt.title(name)
    #     plt.show()

    # cfg = get_config('configs/stavsad_train/base.yaml')
    # writer = SummaryWriter('lol')
    # cfg_dict = cfg._asdict()
    # for key in cfg_dict:
    #     if isinstance(cfg_dict[key], list):
    #         cfg_dict[key] = torch.Tensor(cfg_dict[key])
    # print(cfg_dict['milestones'], type(cfg_dict['milestones']))
    # writer.add_hparams(cfg_dict, {'accuracy': 20, 'recall': 10})

    # cfg = get_train_config('configs/fuji_trained/adam-1.yaml')
    # print(cfg)


if __name__ == '__main__':
    main()
