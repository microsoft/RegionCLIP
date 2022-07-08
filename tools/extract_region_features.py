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

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.clip_rcnn import visualize_proposals

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

def extract_region_feats(cfg, model, batched_inputs, file_name):
    """ Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    # model inference  
    # 1. localization branch: offline modules to get the region proposals           
    images = model.offline_preprocess_image(batched_inputs)
    features = model.offline_backbone(images.tensor)
    proposals, _ = model.offline_proposal_generator(images, features, None)    
    #visualize_proposals(batched_inputs, proposals, model.input_format) 

    # 2. recognition branch: get 2D feature maps using the backbone of recognition branch
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    # 3. given the proposals, crop region features from 2D image features
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposal_boxes, model.backbone.layer4
    )
    att_feats = model.backbone.attnpool(box_features)  # region features

    if cfg.MODEL.CLIP.TEXT_EMB_PATH is None: # save features of RPN regions
        results = model._postprocess(proposals, batched_inputs) # re-scale boxes back to original image size

        # save RPN outputs into files
        im_id = 0 # single image
        pred_boxes = results[im_id]['instances'].get("proposal_boxes").tensor # RPN boxes, [#boxes, 4]
        region_feats = att_feats # region features, [#boxes, d]

        saved_dict = {}
        saved_dict['boxes'] = pred_boxes.cpu()
        saved_dict['feats'] = region_feats.cpu()
    else: # save features of detection regions (after per-class NMS)
        # 4. prediction head classifies the regions (optional)
        predictions = model.roi_heads.box_predictor(att_feats)  # predictions[0]: class logits; predictions[1]: box delta
        pred_instances, keep_indices = model.roi_heads.box_predictor.inference(predictions, proposals) # apply per-class NMS
        results = model._postprocess(pred_instances, batched_inputs) # re-scale boxes back to original image size

        # save detection outputs into files
        im_id = 0 # single image
        pred_boxes = results[im_id]['instances'].get("pred_boxes").tensor # boxes after per-class NMS, [#boxes, 4]
        pred_classes = results[im_id]['instances'].get("pred_classes")# class predictions after per-class NMS, [#boxes], class value in [0, C]
        pred_probs = F.softmax(predictions[0], dim=-1)[keep_indices[im_id]] # class probabilities, [#boxes, #concepts+1], background is the index of C
        region_feats = att_feats[keep_indices[im_id]] # region features, [#boxes, d]
        # assert torch.all(results[0]['instances'].get("scores") == pred_probs[torch.arange(pred_probs.shape[0]).cuda(), pred_classes]) # scores

        saved_dict = {}
        saved_dict['boxes'] = pred_boxes.cpu()
        saved_dict['classes'] = pred_classes.cpu()
        saved_dict['probs'] = pred_probs.cpu()
        saved_dict['feats'] = region_feats.cpu()

    saved_path = os.path.join(cfg.OUTPUT_DIR, os.path.basename(file_name).split('.')[0] + '.pth')
    torch.save(saved_dict, saved_path)

def main(args):
    cfg = setup(args)

    # create model
    model = create_model(cfg)

    # input images
    image_files = [os.path.join(cfg.INPUT_DIR, x) for x in os.listdir(cfg.INPUT_DIR)]
    
    # process each image
    start = time.time()
    for i, file_name in enumerate(image_files):
        if i % 100 == 0:
            print("Used {} seconds for 100 images.".format(time.time()-start))
            start = time.time()
        
        # get input images
        batched_inputs = get_inputs(cfg, file_name)

        # extract region features
        with torch.no_grad():
            extract_region_feats(cfg, model, batched_inputs, file_name)

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
