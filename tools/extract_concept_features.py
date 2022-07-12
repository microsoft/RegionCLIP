#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import torch
from torch.nn import functional as F
import numpy as np
import time

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.data.datasets.clip_prompt_utils import pre_tokenize

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.modeling.backbone.clip_backbone import build_clip_language_encoder

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_inputs(cfg, file_name):
    """ Given a file name, return a list of dictionary with each dict corresponding to an image
    (refer to detectron2/data/dataset_mapper.py)
    """
    # image loading
    dataset_dict = {}
    image = utils.read_image(file_name, format=cfg.INPUT.FORMAT)
    dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1] # h, w before transforms
    
    # image transformation
    augs = utils.build_augmentation(cfg, False)
    augmentations = T.AugmentationList(augs) # [ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice')]
    aug_input = T.AugInput(image)
    transforms = augmentations(aug_input)
    image = aug_input.image
    h, w = image.shape[:2]  # h, w after transforms
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    return [dataset_dict]

def create_model(cfg):
    """ Given a config file, create a detector
    (refer to tools/train_net.py)
    """
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )
    
    assert model.clip_crop_region_type == "RPN"
    assert model.use_clip_c4 # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool # use att_pool from CLIP to match dimension
    model.roi_heads.box_predictor.vis = True # get confidence scores before multiplying RPN scores, if any
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model

def main(args):
    cfg = setup(args)

    # create model
    model = create_model(cfg)

    # input concepts
    concept_file = os.path.join(cfg.INPUT_DIR, 'concepts.txt')
    
    concept_feats = []
    with open(concept_file, 'r') as f:
        for line in f:
            concept = line.strip()
            with torch.no_grad():
                token_embeddings = pre_tokenize([concept]).to(model.device)[0]
                text_features = model.lang_encoder.encode_text(token_embeddings)
                # average over all templates
                text_features = text_features.mean(0, keepdim=True)
                concept_feats.append(text_features)

    concept_feats = torch.stack(concept_feats, 0)
    concept_feats = torch.squeeze(concept_feats).cpu()
    saved_path = os.path.join(cfg.OUTPUT_DIR, 'concept_embeds.pth')
    torch.save(concept_feats, saved_path)

    print("done!")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
