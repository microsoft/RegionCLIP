from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_
from .build import BACKBONE_REGISTRY
from .clip import build_tokenizer


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from detectron2.layers import ShapeSpec
from .backbone import Backbone

logger = logging.getLogger(__name__)


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm != None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden_features)),
            ("act", act_layer()),
            ("fc2", nn.Linear(hidden_features, out_features))
        ]))

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(
                x, 'b (h w) c -> b c h w',
                h=H, w=W
            )

        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):

    def __init__(self, dim, base_dim, groups=8, base_groups=8, qkv_bias=True, dynamic_scale=True, standparam=True):
        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dynamic_scale = dynamic_scale

        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups

        self.base_dim = base_dim
        self.base_groups = base_groups
        self.base_group_dim = base_dim // base_groups

        self.group_wm = self.group_dim / self.base_group_dim  # Width multiplier for each group.
        self.standparam = standparam

    def forward(self, x, size):
        B, N, C = x.shape
        assert C == self.dim

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, groups, N, group_dim].

        scale = N ** -0.5 if self.dynamic_scale else self.dim ** -0.5

        # Change the scaling factor.
        # Ref: examples/Transformer/model.py in muP.
        # Note: We consider backward compatiblity and follow https://github.com/microsoft/mup/issues/18.
        if self.standparam:
            scale = N ** -0.5 if self.dynamic_scale else self.dim ** -0.5
        else:
            assert self.dynamic_scale  # Currently only support dynamic scale.
            scale = N ** -0.5

        q = q * scale
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        
        if not self.standparam:
            # Follow https://github.com/microsoft/mup/issues/18.
            attention = attention / self.group_wm

        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):

    def __init__(self, dim, base_dim, groups, base_groups, mlp_ratio=4., qkv_bias=True,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 conv_at_attn=True, conv_at_ffn=True, dynamic_scale=True, standparam=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, base_dim, groups=groups, base_groups=base_groups, qkv_bias=qkv_bias, dynamic_scale=dynamic_scale, standparam=standparam),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, base_dim, num_heads, base_num_heads, window_size, qkv_bias=True, standparam=True):

        super().__init__()
        
        self.window_size = window_size

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.base_dim = base_dim
        self.base_num_heads = base_num_heads
        base_head_dim = base_dim // base_num_heads

        # Change the scaling factor.
        # Ref: examples/Transformer/model.py in muP.
        # Note: We consider backward compatiblity and follow https://github.com/microsoft/mup/issues/17.
        if standparam:
            scale = float(head_dim) ** -0.5
        else:
            # TODO: Here we ensure backward compatibility, which may not be optimal.
            #       We may add an argument called backward_comp. If it is set as False, we use
            #          float(head_dim) ** -1 * math.sqrt(attn_mult)
            #       as in the Transformer example in muP.
            base_scale = float(base_head_dim) ** -0.5  # The same as scaling in standard parametrization.
            head_wm = head_dim / base_head_dim  # Width multiplier for each head.
            scale = base_scale / head_wm
            # scale_1 = (float(base_head_dim) ** 0.5) * (float(head_dim) ** -1) # Equivalent implementation as shown in the muP paper.
            # assert np.isclose(scale, scale_1)
        self.scale = scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):

        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(
            -1, self.window_size, self.window_size, C
        )
        x = window_reverse(x, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):

    def __init__(self, dim, base_dim, num_heads, base_num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, conv_at_attn=True, conv_at_ffn=True, standparam=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, base_dim, num_heads, base_num_heads, window_size, qkv_bias=qkv_bias, standparam=standparam),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(Backbone):
    """ DaViT: Dual-Attention Transformer

    Args:
        img_size (int | tuple(int)): Input image size. Default: 224 
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of spatial and channel blocks in different stages. Default: (1, 1, 3, 1)
        patch_size (tuple(int)): Patch sizes in different stages. Default: (7, 2, 2, 2)
        patch_stride (tuple(int)): Patch strides in different stages. Default: (4, 2, 2, 2) 
        patch_padding (tuple(int)): Patch padding sizes in different stages. Default: (3, 0, 0, 0)
        patch_prenorm (tuple(bool)): Use pre-normalization or not in different stages. Default: (False, False, False, False)
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        base_embed_dims (tuple(int)): Patch embedding dimension (base case for muP). Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        base_num_heads (tuple(int)): Number of attention heads in different layers (base case for muP). Default: (4, 8, 12, 16)
        num_groups (tuple(int)): Number of groups in channel attention in different layers. Default: (3, 6, 12, 24)
        base_num_groups (tuple(int)): Number of groups in channel attention in different layers (base case for muP). Default: (3, 6, 12, 24)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        enable_checkpoint (bool): If True, enabling checkpoint. Default: False
        conv_at_attn (bool): If True, add convolution layer before attention. Default: True
        conv_at_ffn (bool): If True, add convolution layer before ffn. Default: True
        dynamic_scale (bool): If True, scale of channel attention is respect to the number of tokens. Default: True
        standparam (bool): Use standard parametrization or mu-parametrization. Default: True (i.e., use standard paramerization)
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        base_embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        base_num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        base_num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
        dynamic_scale=True,
        standparam=True, 
        out_features=["stage2", "stage3", "stage4", "stage5"], 
     ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)
        self.out_features = out_features

        num_stages = len(embed_dims)
        self.img_size = img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]

        self._out_feature_strides = {}
        self._out_feature_channels = {}
        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i]
            )
            convs.append(conv_embed)

            logger.info(f'=> Depth offset in stage {i}: {depth_offset}')
            block = MySequential(
                *[
                    MySequential(OrderedDict([
                        (
                            'spatial_block', SpatialBlock(
                                embed_dims[i],
                                base_embed_dims[i],
                                num_heads[i],
                                base_num_heads[i],
                                window_size,
                                drop_path_rate=dpr[depth_offset+j*2],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                                standparam=standparam
                            )
                        ),
                        (
                            'channel_block', ChannelBlock(
                                embed_dims[i],
                                base_embed_dims[i],
                                num_groups[i],
                                base_num_groups[i],
                                drop_path_rate=dpr[depth_offset+j*2+1],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                                dynamic_scale=dynamic_scale,
                                standparam=standparam
                            )
                        )
                    ])) for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i]*2

            stage = f'stage{i + 2}'
            if stage in self.out_features:
                self._out_feature_channels[stage] = embed_dims[i]
                self._out_feature_strides[stage] = 4 * 2 ** i

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.out_indices = []
        self.output_is_normalized = False
        
        self.norm = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if standparam:
            self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = MuReadout(self.embed_dims[-1], num_classes, readout_zero_init=True)  # Follow examples/ResNet/resnet.py in muP.


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }

    def custom_init_weights(self, use_original_init=True):
        self.use_original_init = use_original_init
        logger.info('Custom init: {}'.format('original init' if self.use_original_init else 'muP init'))
        self.apply(self._custom_init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _custom_init_weights(self, m):
        # Customized initialization for weights.
        if self.use_original_init:
            # Original initialization. 
            # Note: This is not SP init. We do not implement SP init here.
            custom_trunc_normal_ = trunc_normal_
            custom_normal_ = nn.init.normal_
        else:
            # muP.
            custom_trunc_normal_ = mup.init.trunc_normal_
            custom_normal_ = mup.init.normal_

        # These initializations will overwrite the existing inializations from the modules and adjusted by set_base_shapes().
        if isinstance(m, MuReadout):
            pass  # Note: MuReadout is already zero initialized due to readout_zero_init=True.
        elif isinstance(m, nn.Linear):
            custom_trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            custom_normal_(m.weight, std=0.02)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # Follow P24 Layernorm Weights and Biases.
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):  # Follow P24 Layernorm Weights and Biases.
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _try_remap_keys(self, pretrained_dict):
        remap_keys = {
            "conv_embeds": "convs",
            "main_blocks": "blocks",
            "0.cpe.0.proj": "spatial_block.conv1.fn.dw",
            "0.attn": "spatial_block.window_attn.fn",
            "0.cpe.1.proj": "spatial_block.conv2.fn.dw",
            "0.mlp": "spatial_block.ffn.fn.net",
            "1.cpe.0.proj": "channel_block.conv1.fn.dw",
            "1.attn": "channel_block.channel_attn.fn",
            "1.cpe.1.proj": "channel_block.conv2.fn.dw",
            "1.mlp": "channel_block.ffn.fn.net",
            "0.norm1": "spatial_block.window_attn.norm",
            "0.norm2": "spatial_block.ffn.norm",
            "1.norm1": "channel_block.channel_attn.norm",
            "1.norm2": "channel_block.ffn.norm"
        }

        full_key_mappings = {}
        for k in pretrained_dict.keys():
            old_k = k
            for remap_key in remap_keys.keys():
                if remap_key in k:
                    print(f'=> Repace {remap_key} with {remap_keys[remap_key]}')
                    k = k.replace(remap_key, remap_keys[remap_key])

            full_key_mappings[old_k] = k

        return full_key_mappings

    def from_state_dict(self, pretrained_dict, pretrained_layers=[], verbose=True):
        model_dict = self.state_dict()
        stripped_key = lambda x: x[14:] if x.startswith('image_encoder.') else x
        full_key_mappings = self._try_remap_keys(pretrained_dict)

        pretrained_dict = {
            stripped_key(full_key_mappings[k]): v for k, v in pretrained_dict.items()
            if stripped_key(full_key_mappings[k]) in model_dict.keys()
        }
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )
            if need_init:
                if verbose:
                    print(f'=> init {k} from pretrained state dict')

                need_init_state_dict[k] = v
        self.load_state_dict(need_init_state_dict, strict=False)

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            print(f'=> loading pretrained model {pretrained}')
            pretrained_dict = torch.load(pretrained, map_location='cpu')

            self.from_state_dict(pretrained_dict, pretrained_layers, verbose)

    def forward_features(self, x):
        input_size = (x.size(2), x.size(3))

        outs = {}
        for i, (conv, block) in enumerate(zip(self.convs, self.blocks)):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)
            if i in self.out_indices:
                out = x.view(-1, *input_size, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
                outs["stage{}".format(i + 2)] = out       

        if len(self.out_indices) == 0:
            outs["stage5"] = x.view(-1, *input_size, self.embed_dims[-1]).permute(0, 3, 1, 2).contiguous()
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@BACKBONE_REGISTRY.register()
def build_clip_davit_backbone(cfg, input_shape):
    """
    Create a CLIP Swin instance from config.

    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """    
    spec_vision = cfg.MODEL.DAVIT

    # Dummy values for muP parameters.
    base_embed_dims = spec_vision['DIM_EMBED']
    base_num_heads = spec_vision['NUM_HEADS']
    base_num_groups = spec_vision['NUM_GROUPS']

    return DaViT(      
        num_classes=0,
        depths=spec_vision['DEPTHS'],
        embed_dims=spec_vision['DIM_EMBED'],
        base_embed_dims=base_embed_dims,
        num_heads=spec_vision['NUM_HEADS'],
        base_num_heads=base_num_heads,
        num_groups=spec_vision['NUM_GROUPS'],
        base_num_groups=base_num_groups,
        patch_size=spec_vision['PATCH_SIZE'],
        patch_stride=spec_vision['PATCH_STRIDE'],
        patch_padding=spec_vision['PATCH_PADDING'],
        patch_prenorm=spec_vision['PATCH_PRENORM'],
        drop_path_rate=spec_vision['DROP_PATH_RATE'],
        img_size=input_shape,
        window_size=spec_vision.get('WINDOW_SIZE', 7),
        enable_checkpoint=spec_vision.get('ENABLE_CHECKPOINT', False),
        conv_at_attn=spec_vision.get('CONV_AT_ATTN', True),
        conv_at_ffn=spec_vision.get('CONV_AT_FFN', True),
        dynamic_scale=spec_vision.get('DYNAMIC_SCALE', True),
        standparam=True,
    )