# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from . import transforms as T
import torchvision.transforms as T
from PIL import Image
from timm.data import create_transform
from .torchvision_transforms.transforms import Resize as New_Resize

def build_clip_transforms(cfg, is_train=True):
    if cfg.AUG.USE_TIMM and is_train:
        print('=> use timm transform for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        transforms = create_transform(
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.MODEL.PIXEL_MEAN,
            std=cfg.MODEL.PIXEL_STD,
        )

        return transforms

    transforms = None
    if is_train:
        aug = cfg.AUG
        scale = aug.SCALE
        ratio = aug.RATIO 
        if len(cfg.AUG.TRAIN.IMAGE_SIZE) == 2:
            ts = [
                T.RandomResizedCrop(
                    cfg.AUG.TRAIN.IMAGE_SIZE[0], scale=scale, ratio=ratio,
                    interpolation=cfg.AUG.INTERPOLATION
                ),
                T.RandomHorizontalFlip(),
            ]
        elif len(cfg.AUG.TRAIN.IMAGE_SIZE) == 1 and cfg.AUG.TRAIN.MAX_SIZE is not None:  # designed for pretraining fastrcnn
            ts = [
                New_Resize(
                    cfg.AUG.TRAIN.IMAGE_SIZE[0], max_size=cfg.AUG.TRAIN.MAX_SIZE,
                    interpolation=cfg.AUG.INTERPOLATION
                ),
                T.RandomHorizontalFlip(),
            ]

        ts.append(T.ToTensor())
        transforms = T.Compose(ts)
    else:
        pass

    return transforms

