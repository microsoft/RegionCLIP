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

    # normalize_transform = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    # NOTE: normalization is applied in rcnn.py, to keep consistent as Detectron2
    # normalize = T.Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD) # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    transforms = None
    if is_train:
        aug = cfg.AUG
        scale = aug.SCALE
        ratio = aug.RATIO 
        if len(cfg.AUG.TRAIN.IMAGE_SIZE) == 2:  # Data Augmentation from MSR-CLIP 
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

        cj = aug.COLOR_JITTER
        if cj[-1] > 0.0:
            ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

        gs = aug.GRAY_SCALE
        if gs > 0.0:
            ts.append(T.RandomGrayscale(gs))

        gb = aug.GAUSSIAN_BLUR
        if gb > 0.0:
            ts.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gb))

        ts.append(T.ToTensor())
        # NOTE: normalization is applied in rcnn.py, to keep consistent as Detectron2
        #ts.append(normalize)

        transforms = T.Compose(ts)
    else:
        # for zeroshot inference of grounding evaluation
        transforms = T.Compose([
            T.Resize(
                cfg.AUG.TEST.IMAGE_SIZE[0],
                interpolation=cfg.AUG.TEST.INTERPOLATION
            ),
            T.ToTensor(),
        ])
        return transforms

    return transforms

