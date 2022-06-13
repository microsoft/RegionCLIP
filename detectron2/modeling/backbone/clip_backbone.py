from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from detectron2.layers.blocks import FrozenBatchNorm2d
from detectron2.layers import ShapeSpec

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_type='FronzenBN'):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(planes) # nn.BatchNorm2d(planes)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(planes * self.expansion) # nn.BatchNorm2d(planes * self.expansion)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            if norm_type == 'FronzenBN':
                this_norm = FrozenBatchNorm2d(planes * self.expansion) #("1", nn.BatchNorm2d(planes * self.expansion))
            elif norm_type == 'SyncBN':
                this_norm = nn.SyncBatchNorm(planes * self.expansion)
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", this_norm), #("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(Backbone):
    """
    Extended from CLIP implementation. It contains following changes:
    1. change all nn.BatchNorm2d() to FrozenBatchNorm2d(), due to small batch size of detection training
    2. add self._out_feature_strides according to standard ResNet
    2. modify forward() to be compatible with Detectron2
    3. add freeze() and output_shape() to be compatible with Detectron2
    4. add build_clip_resnet_backbone() to build this ModifiedResNet

    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, 
        out_features=None, freeze_at=0, depth=None, pool_vec=True, create_att_pool=False, norm_type='FronzenBN'):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.norm_type = norm_type

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn1 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn1 = nn.SyncBatchNorm(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn2 = FrozenBatchNorm2d(width // 2)  # nn.BatchNorm2d(width // 2)
        elif norm_type == 'SyncBN':
            self.bn2 = nn.SyncBatchNorm(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        if norm_type == 'FronzenBN':
            self.bn3 = FrozenBatchNorm2d(width) # nn.BatchNorm2d(width)
        elif norm_type == 'SyncBN':
            self.bn3 = nn.SyncBatchNorm(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        if 'res5' in out_features:  # FPN
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        else:  # C4, layer4 created here won't be used in backbone, but used in roi_head
            self.layer4 = self._make_layer(width * 8, layers[3], stride=2) # None
        
        self.pool_vec = pool_vec
        if self.pool_vec or create_att_pool:  # pool a vector representation for an image
            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # if create_att_pool:  # freeze attnpool layer
        #     for p in self.attnpool.parameters(): p.requires_grad = False

        self._out_features = out_features if out_features else []
        if depth in [50,101]: # resnet50 or resnet 101
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048} if 'res5' in self._out_features \
                else {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        elif depth in [200]: # resnet50x4
            # FPN: ["res2", "res3", "res4", "res5"]; C4: ["res4"]
            self._out_feature_channels = {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280, 'res5': 2560} if 'res5' in self._out_features \
                else {'stem': 80, 'res2': 320, 'res3': 640, 'res4': 1280}
            self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32} if 'res5' in self._out_features \
                else  {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16}  # anti-aliasing strided conv???        
        self.freeze(freeze_at)


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, norm_type=self.norm_type)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = x.type(self.conv1.weight.dtype) # det2 resnet50: [3, 800, 1216]; CLIP resnet50: [3, 224, 224]
        x = stem(x) # det2 resnet50: [64, 200, 304]; CLIP resnet50: [64, 56, 56]
        if "stem" in self._out_features:
            outputs["stem"] = x
        x = self.layer1(x) # det2 resnet50: [256, 200, 304]; CLIP resnet50: [256, 56, 56]
        outputs['res2'] = x if "res2" in self._out_features else None
        x = self.layer2(x) # det2 resnet50: [512, 100, 152]; CLIP resnet50: [512, 28, 28]
        outputs['res3'] = x if "res3" in self._out_features else None
        x = self.layer3(x) # det2 resnet50: [1024, 50, 76]; CLIP resnet50: [1024, 14, 14]
        outputs['res4'] = x if "res4" in self._out_features else None
        x = self.layer4(x)  if "res5" in self._out_features else x # det2 resnet50: [2048, 25, 38]; CLIP resnet50: [2048, 7, 7]
        outputs['res5'] = x if "res5" in self._out_features else None

        if self.pool_vec:  # pool a vector representation for an image, for global image classification
            x = self.attnpool(x) # CLIP resnet50: [1024]
            return x
        else:  # for FPN
            return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        def cnnblockbase_freeze(nn_module):
            """
            Make this block not trainable.
            This method sets all parameters to `requires_grad=False`,
            and convert all BatchNorm layers to FrozenBatchNorm

            Returns:
                the block itself
            """
            for p in nn_module.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(nn_module)
        
        if freeze_at >= 1: # stem
            cnnblockbase_freeze(self.conv1)
            cnnblockbase_freeze(self.bn1)
            cnnblockbase_freeze(self.conv2)
            cnnblockbase_freeze(self.bn2)
            cnnblockbase_freeze(self.conv3)
            cnnblockbase_freeze(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2): 
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    cnnblockbase_freeze(block)  
        return self

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(Backbone):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 out_features,
                 freeze_at,
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                out_features=out_features,
                freeze_at=freeze_at,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


@BACKBONE_REGISTRY.register()
def build_vit_clip(cfg, input_shape):
    """
    Create the whole CLIP instance from config.

    Returns:
        CLIP: a :class:`CLIP` instance.
    """
    # port standard ResNet config to CLIP ModifiedResNet
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = ['res5'] # includes the whole ResNet # cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    
    # num_blocks_per_stage = {
    #     18: [2, 2, 2, 2],
    #     34: [3, 4, 6, 3],
    #     50: [3, 4, 6, 3],
    #     101: [3, 4, 23, 3],
    #     152: [3, 8, 36, 3],
    # }[depth]
    vision_layers = 12 # num_blocks_per_stage
    vision_width = 768 # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    
    # default configs of CLIP
    embed_dim = 512 # 1024
    image_resolution = 224
    vision_patch_size = 32 # None
    context_length = 77
    vocab_size = 49408
    transformer_width = 512
    transformer_heads = 8
    transformer_layers = 12

    model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            out_features, freeze_at
        )
    return model

@BACKBONE_REGISTRY.register()
def build_resnet_clip(cfg, input_shape):
    """
    Create the whole CLIP instance from config.

    Returns:
        CLIP: a :class:`CLIP` instance.
    """
    # port standard ResNet config to CLIP ModifiedResNet
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = ['res5'] # includes the whole ResNet # cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    
    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [4, 6, 10, 6], # flag for ResNet50x4
    }[depth]
    vision_layers = num_blocks_per_stage
    vision_width = {
        50: 64,
        101: 64,
        200: 80, # flag for ResNet50x4
    }[depth]  # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    
    # default configs of CLIP
    embed_dim = {
        50: 1024,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    vision_heads = vision_width * 32 // 64
    image_resolution = {
        50: 224,
        101: 224,
        200: 288, # flag for ResNet50x4
    }[depth] 
    vision_patch_size = None
    context_length = 77
    vocab_size = 49408
    transformer_width = {
        50: 512,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    transformer_heads = {
        50: 8,
        101: 8,
        200: 10, # flag for ResNet50x4
    }[depth] 
    transformer_layers = 12

    model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            out_features, freeze_at
        )
    return model


@BACKBONE_REGISTRY.register()
def build_clip_resnet_backbone(cfg, input_shape):
    """
    Create a CLIP-version ResNet instance from config.

    Returns:
        ModifiedResNet: a :class:`ModifiedResNet` instance.
    """
    # port standard ResNet config to CLIP ModifiedResNet
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    # num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    # width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    # bottleneck_channels = num_groups * width_per_group
    # in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    # out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    # stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    # res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    # deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    # deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    
    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [4, 6, 10, 6], # flag for ResNet50x4
    }[depth]
    vision_layers = num_blocks_per_stage
    vision_width = {
        50: 64,
        101: 64,
        200: 80, # flag for ResNet50x4
    }[depth]  # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    
    # default configs of CLIP ModifiedResNet, but not used if only building ModifiedResNet as backbone
    embed_dim = {
        50: 1024,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    vision_heads = vision_width * 32 // 64
    image_resolution = {
        50: 224,
        101: 224,
        200: 288, # flag for ResNet50x4
    }[depth] 

    # if combine {ModifiedResNet of CLIP, C4, text emb as classifier}, then has to use att_pool to match dimension
    create_att_pool = True if (cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER)\
                           or cfg.MODEL.ROI_HEADS.NAME == 'PretrainRes5ROIHeads' else False

    return ModifiedResNet(layers=vision_layers, 
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                out_features=out_features, 
                freeze_at=freeze_at,
                depth=depth,
                pool_vec=False,
                create_att_pool=create_att_pool,
                )


class CLIPLangEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 out_features,
                 freeze_at,
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp[0].weight.dtype # torch.float32, not sure whether need to be fp16 in pretraining

    def encode_text(self, text, only_eot=True):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        if only_eot:
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x
        else:
            # return embeddings for all tokens, instead of the eot embedding as CLIP implementation below
            return x @ self.text_projection


def build_clip_language_encoder(cfg):
    """
    Create the CLIP language encoder instance from config.

    Returns:
        CLIP: a :class:`CLIP` instance.
    """
    # port standard ResNet config to CLIP ModifiedResNet
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = ['res5'] # includes the whole ResNet # cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    
    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [4, 6, 10, 6], # flag for ResNet50x4
    }[depth]
    vision_layers = num_blocks_per_stage
    vision_width = {
        50: 64,
        101: 64,
        200: 80, # flag for ResNet50x4
    }[depth]  # cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    
    # default configs of CLIP
    embed_dim = {
        50: 1024,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    vision_heads = vision_width * 32 // 64
    image_resolution = {
        50: 224,
        101: 224,
        200: 288, # flag for ResNet50x4
    }[depth] 
    vision_patch_size = None
    context_length = 77
    vocab_size = 49408
    transformer_width = {
        50: 512,
        101: 512,
        200: 640, # flag for ResNet50x4
    }[depth] 
    transformer_heads = {
        50: 8,
        101: 8,
        200: 10, # flag for ResNet50x4
    }[depth] 
    transformer_layers = 12

    model = CLIPLangEncoder(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            out_features, freeze_at
        )
    return model