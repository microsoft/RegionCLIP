import torch
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

from .hfpt_tokenizer import HFPTTokenizer
from .simple_tokenizer import SimpleTokenizer
from .bert_tokenizer import BertTokenizer
from .text2image import Text2Image

def build_tokenizer(tokenizer_name):
    tokenizer = None
    if tokenizer_name == 'clip':
        tokenizer = SimpleTokenizer()
    elif 'bert' in tokenizer_name:
        tokenizer = BertTokenizer(tokenizer_name)    # Download vocabulary from S3 and cache.
    elif 'hf_' in tokenizer_name:
        tokenizer = HFPTTokenizer(pt_name=tokenizer_name[3:])
    elif 'hfc_' in tokenizer_name:
        tokenizer = HFPTTokenizer(pt_name=tokenizer_name[4:])
    else:
        raise ValueError('Unknown tokenizer')

    return tokenizer

def build_text2image(config):
    return Text2Image(config)
    
