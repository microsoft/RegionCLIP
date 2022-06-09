#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl

  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"

  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")

    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
    
    #############################################################################################################################
    # first, use above code to convert ckpt names
    # second, use following code to convert CVPR OVR pretrained model (including V2L projection layer) into Detectron2 format
    kk = torch.load('/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/cvpr_ovr_models/model_final.pth')
    ll = pkl.load(open('/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/cvpr_ovr_models/model_pretrained_visbackbone_convert-d2.pth','rb'))
    ll['model']['module.roi_heads.box_predictor.emb_pred.weight'] = kk['model']['module.roi_heads.box.predictor.emb_pred.weight']
    ll['model']['module.roi_heads.box_predictor.emb_pred.bias'] = kk['model']['module.roi_heads.box.predictor.emb_pred.bias']
    for k in list(ll['model'].keys()):
        replace_it = False
        if 'stem.module.backbone.body' in k:
            new_k = k.replace('stem.module.backbone.body.','backbone.')
            replace_it = True
        elif 'module.backbone.body.res5.' in k:
            new_k = k.replace('module.backbone.body.res5.','roi_heads.res5.')
            replace_it = True
        elif 'module.backbone.body' in k:
            new_k = k.replace('module.backbone.body.','backbone.')
            replace_it = True
        elif 'module' in k:
            new_k = k.replace('module.','')
            replace_it = True
        
        if replace_it:
            ll['model'][new_k] = ll['model'][k]
            trash = ll['model'].pop(k)
    for k in ll['model']: 
        print(k)
    torch.save(ll, '/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/cvpr_ovr_models/cvpr_ovr_pretrained_weights.pth')
    # torch.save(kk['model']['module.roi_heads.box.predictor.cls_score.weight'][1:].cpu(), '/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/cvpr_ovr_models/coco48_cls_emb.pth')