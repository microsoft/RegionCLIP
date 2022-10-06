from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import DropPath, trunc_normal_
from .build import BACKBONE_REGISTRY
from .clip import build_tokenizer
from detectron2.layers import ShapeSpec
from .backbone import Backbone

import os
import json
import logging
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed(
            patch_size=kwargs['patch_size'],
            embed_dim=kwargs['embed_dim'],
        )

        self.embed_dim = kwargs['embed_dim']
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        self.num_features = [self.embed_dim]
        self.out_features = ['stage5']

        self._out_feature_channels = {
            'stage5': self.embed_dim,             
        }
        self._out_feature_strides = {
            'stage5': 16,             
        }

        self.output_is_normalized = False

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def interpolate_pos_embed(self, new_size):
        num_patches = self.patch_embed.num_patches
        num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = self.patch_embed.grid_size
        # height (== width) for the new position embedding
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            extra_tokens = self.pos_embed[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = self.pos_embed[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            return new_pos_embed
        else:
            return self.pos_embed

    def forward_features(self, x):
        B = x.shape[0]

        self.new_h = self.patch_embed.patch_size[0] * (x.shape[2] // self.patch_embed.patch_size[0])
        self.new_w = self.patch_embed.patch_size[1] * (x.shape[3] // self.patch_embed.patch_size[1])
        x = x[:, :, :self.new_h, :self.new_w]

        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        pos_embed = self.interpolate_pos_embed((self.new_h // self.patch_embed.patch_size[0], self.new_w // self.patch_embed.patch_size[1]))
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :]  # global pool without cls token
            outcome = self.fc_norm(x)            
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        self.output_is_normalized = True

        return outcome

    def forward(self, x):

        out = self.forward_features(x)
        outs = {
            'stage5': out.view(-1, self.new_h // self.patch_embed.patch_size[0], self.new_w // self.patch_embed.patch_size[1], self.embed_dim).permute(0, 3, 1, 2).contiguous()
        }

        return outs


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }

    def from_pretrained(self, pretrained='', verbose=True):
        if pretrained != '':
            assert os.path.isfile(pretrained), "checkpoint not available"
            logger.info(f'=> loading pretrained model {pretrained}')
            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'model' in checkpoint:
                logger.info('Load from original MAE checkpoints')
                checkpoint_model = checkpoint['model']
            else:
                logger.info('Load from original Mainz checkpoints')
                checkpoint_model = checkpoint
            
            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(self, checkpoint_model)

            # load pre-trained model
            msg = self.load_state_dict(checkpoint_model, strict=False)
            logger.info(msg)

    def get_layer_id_for_vit(self, name, num_layers):
        """
        Assign a parameter with its layer id
        Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        """
        if name in ['cls_token', 'pos_embed']:
            return 0
        elif name.startswith('patch_embed'):
            return 0
        elif name.startswith('blocks'):
            return int(name.split('.')[1]) + 1
        else:
            return num_layers
    
    def param_groups_lrd(self, config):
        weight_decay=config['OPTIMIZER_PARAMS']['weight_decay']
        no_weight_decay_list=self.no_weight_decay()
        layer_decay=config['CUSTOMIZED_PARAMS_FUNC']['LAYER_DECAY']
        base_lr = config['START_LEARNING_RATE']
        
        param_group_names = {}
        param_groups = {}

        num_layers = len(self.blocks) + 1

        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            # no decay: all 1D parameters and model specific ones
            if p.ndim == 1 or n in no_weight_decay_list:
                g_decay = "no_decay"
                this_decay = 0.0
            else:
                g_decay = "decay"
                this_decay = weight_decay
                
            layer_id = self.get_layer_id_for_vit(n, num_layers)
            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    # "lr": this_scale * base_lr,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    # "lr": this_scale * base_lr,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

        logger.info("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
        
        return list(param_groups.values())

    @property
    def dim_out(self):
        return self.embed_dim


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def image_encoder(config_encoder, verbose, **kwargs):
    spec = config_encoder['SPEC']
    if 'PREDEFINE' in config_encoder:
        if config_encoder['PREDEFINE'] == 'vitb16':
            vit = vit_base_patch16(
                num_classes=config_encoder['NUM_CLASSES'], 
                drop_path_rate=spec['DROP_PATH_RATE'], 
                global_pool=True, # this way we can evaluate the zero-shot od and seg
                )
        elif config_encoder['PREDEFINE'] == 'vitl16':
            vit = vit_large_patch16(
                num_classes=config_encoder['NUM_CLASSES'], 
                drop_path_rate=spec['DROP_PATH_RATE'], 
                global_pool=True, # this way we can evaluate the zero-shot od and seg                
                )
        elif config_encoder['PREDEFINE'] == 'vith14':
            vit = vit_huge_patch14(
                num_classes=config_encoder['NUM_CLASSES'], 
                drop_path_rate=spec['DROP_PATH_RATE'], 
                global_pool=True, # this way we can evaluate the zero-shot od and seg                
                )
        else:
            raise NotImplementedError
    else:
        # by default we use cls token in mae vit
        vit = VisionTransformer(
            img_size=config_encoder['IMAGE_SIZE'][0],
            patch_size=spec['PATCH_SIZE'],
            in_chans=3,
            num_classes=config_encoder['NUM_CLASSES'],
            embed_dim=spec['EMBED_DIM'],
            depth=spec['DEPTH'],
            num_heads=spec['NUM_HEADS'],
            mlp_ratio=4.,
            qkv_bias=spec['QKV_BIAS'],
            # qk_scale=None,
            drop_rate=spec['DROP_RATE'],
            attn_drop_rate=spec['ATTN_DROP_RATE'],
            drop_path_rate=spec['DROP_PATH_RATE'],
            # hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            global_pool=(not spec['USE_CLS_TOKEN'])
        )

    if config_encoder['LOAD_PRETRAINED']:
        vit.from_pretrained(
            config_encoder['PRETRAINED'],
            # config_encoder['PRETRAINED_LAYERS'],
            verbose
        )

    return vit


@BACKBONE_REGISTRY.register()
def build_clip_vit_backbone(cfg, input_shape):
    """
    Create a CLIP Swin instance from config.

    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """    
    spec_vision = cfg.MODEL.VIT

    return vit_base_patch16(
        img_size=spec_vision['IMG_SIZE'], 
        num_classes=spec_vision['NUM_CLASSES'], 
        drop_path_rate=spec_vision['DROP_PATH_RATE'], 
        global_pool=True, # this way we can evaluate the zero-shot od and seg
    )