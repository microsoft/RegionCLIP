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
from .clip.prompt_engineering import get_prompt_templates
from .LangEncoder import build_tokenizer, build_lang_encoder

class CLIPLangEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 autogressive: bool, 
                 ):
        super().__init__()

        text_encoder_config = {
            'NAME': 'transformer', 
            'CONTEXT_LENGTH': 77, 
            'WIDTH': transformer_width, 
            'LAYERS': transformer_layers, 
            'HEADS': transformer_heads, 
            'AUTOGRESSIVE': autogressive, 
        }

        # build up text encoder
        tokenizer = build_tokenizer({'TOKENIZER': 'clip'})
        tokenizer_type = 'clip'
        lang_encoder = build_lang_encoder(text_encoder_config, tokenizer, False)
        max_token_num = 77
        
        dim_lang = text_encoder_config['WIDTH']
        dim_projection = 512

        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = nn.Parameter(torch.empty(dim_lang, dim_projection))
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))

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
        
    
    def get_text_embeddings(self, class_names, is_eval=False, prompt=True, norm=True):
        if not is_eval:
            if prompt:
                # randomly sample one template
                arbitary_concepts = [
                    prompt_engineering(class_names[label].replace('-other','').replace('-merged',''), topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, 'text_embeddings', text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                templates = get_prompt_templates()
                clss_embeddings = []
                for clss in class_names:
                    txts = [template.format(clss.replace('-other','').replace('-merged','')) for template in templates]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, 'text_embeddings', text_emb)

    # @torch.no_grad()
    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x


def build_clip_language_encoder(cfg):
    """
    Create the CLIP language encoder instance from config.

    Returns:
        CLIP: a :class:`CLIP` instance.
    """
    # default configs of CLIP
    embed_dim = 1024
    vision_heads = 32
    image_resolution = 224
    vision_patch_size = None
    context_length = 77
    vocab_size = 49408
    transformer_width = 512
    transformer_heads = 8
    transformer_layers = 12
    autogressive = cfg.MODEL.TEXT.AUTOGRESSIVE

    model = CLIPLangEncoder(
            embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, autogressive
        )
    return model    