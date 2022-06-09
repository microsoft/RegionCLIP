import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import torch
import numpy as np
from typing import Union, List

from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from .coco_zeroshot_categories import COCO_UNSEEN_CLS, COCO_SEEN_CLS, COCO_OVD_ALL_CLS, COCO_80_ALL_CLS

# https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        self.vocab = vocab
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text, return_link=False):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        str2id_links = []  #  link original sentence word to the tokenized ids of its subwords
        for token in re.findall(self.pat, text):
            this_link = [token]
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            ids = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')]
            bpe_tokens.extend(ids)
            this_link.append(ids)
            str2id_links.append(this_link)
        if return_link:
            return bpe_tokens, str2id_links
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


# https://github.com/openai/CLIP/blob/main/clip/clip.py
#_tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


# prompt_engineering.py
def get_prompt_templates():
    # prompt_templates = [
    #     'There is a {} in the scene.',
    #     'There is the {} in the scene.',
    #     'a photo of a {} in the scene.',
    #     'a photo of the {} in the scene.',
    #     'a photo of one {} in the scene.',

    #     'itap of a {}.',
    #     'itap of my {}.',  # itap: I took a picture of
    #     'itap of the {}.',
    #     'a photo of a {}.',
    #     'a photo of my {}.',
    #     'a photo of the {}.',
    #     'a photo of one {}.',
    #     'a photo of many {}.',

    #     'a good photo of a {}.',
    #     'a good photo of the {}.',
    #     'a bad photo of a {}.',
    #     'a bad photo of the {}.',
    #     'a photo of a nice {}.',
    #     'a photo of the nice {}.',
    #     'a photo of a cool {}.',
    #     'a photo of the cool {}.',
    #     'a photo of a weird {}.',
    #     'a photo of the weird {}.',

    #     'a photo of a small {}.',
    #     'a photo of the small {}.',
    #     'a photo of a large {}.',
    #     'a photo of the large {}.',

    #     'a photo of a clean {}.',
    #     'a photo of the clean {}.',
    #     'a photo of a dirty {}.',
    #     'a photo of the dirty {}.',

    #     'a bright photo of a {}.',
    #     'a bright photo of the {}.',
    #     'a dark photo of a {}.',
    #     'a dark photo of the {}.',

    #     'a photo of a hard to see {}.',
    #     'a photo of the hard to see {}.',
    #     'a low resolution photo of a {}.',
    #     'a low resolution photo of the {}.',
    #     'a cropped photo of a {}.',
    #     'a cropped photo of the {}.',
    #     'a close-up photo of a {}.',
    #     'a close-up photo of the {}.',
    #     'a jpeg corrupted photo of a {}.',
    #     'a jpeg corrupted photo of the {}.',
    #     'a blurry photo of a {}.',
    #     'a blurry photo of the {}.',
    #     'a pixelated photo of a {}.',
    #     'a pixelated photo of the {}.',

    #     'a black and white photo of the {}.',
    #     'a black and white photo of a {}.',

    #     'a plastic {}.',
    #     'the plastic {}.',

    #     'a toy {}.',
    #     'the toy {}.',
    #     'a plushie {}.',
    #     'the plushie {}.',
    #     'a cartoon {}.',
    #     'the cartoon {}.',

    #     'an embroidered {}.',
    #     'the embroidered {}.',

    #     'a painting of the {}.',
    #     'a painting of a {}.',
    # ]

    prompt_templates = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return prompt_templates

def prompt_engineering(classnames, template=""):
    return template.replace('{}', classnames.replace(',', '').replace('+', ' '))

# clip_img_tsv.py
def convert_example_to_features_bpe(text, tokenizer, sot_token, eot_token, context_length=77):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample.
    :param tokenizer: Tokenizer
    :return: List, a list containing token id, padded by 0
    """
    assert isinstance(text, str)
    input_ids = [sot_token] + tokenizer.encode(text) + [eot_token]
    if len(input_ids) > context_length:
        input_ids = input_ids[:context_length]
    input_ids = np.array(input_ids)

    pad_input_ids = np.zeros(context_length)
    pad_input_ids[:input_ids.shape[0]] = input_ids

    return pad_input_ids

def get_cls_names(filter_novel=False, coco=None, from_file=False):
    """ return a list of strings with each string as name of a class
    """
    # the names are stored in a txt file
    if from_file:
        # coco_det_cls = {COCO_80_ALL_CLS[key]: key for key in COCO_80_ALL_CLS} 
        # # not found in nouns {'skis': 31, 'sports ball': 33, 'hot dog': 53, 'potted plant': 59, 'scissors': 77, 'hair drier': 79}
        # coco_det_cls['ski'] = 81
        # coco_det_cls['scissor'] = 82
        # with open('/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/concept_pool/COCO_Caption_nouns_4688.txt','w') as g:
        #     with open(from_file, 'r') as f:
        #         cnt = 0
        #         for row in f:
        #             if row.split(",")[0] not in coco_det_cls:
        #                 g.write(row)
        #                 cnt += 1
        #             else:
        #                 coco_det_cls.pop(row.split(",")[0])
        names = []
        with open(from_file, 'r') as f:
            for row in f:
                names.append(row.split(",")[0])
        return names
    # classes' names
    if coco == 'target':
        return COCO_UNSEEN_CLS
    elif coco == 'base':
        return COCO_SEEN_CLS
    elif coco == 'all':
        return COCO_OVD_ALL_CLS
    elif coco == 'all_80':
        return [COCO_80_ALL_CLS[i+1] for i in range(80)]
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    if filter_novel:
        class_names = [cls_meta['name'] for cls_meta in lvis_categories if cls_meta['frequency'] != 'r']
    else:
        class_names = [cls_meta['name'] for cls_meta in lvis_categories]

    # remove or replace special symbols
    class_names = [cls_n.replace("_", " ") for cls_n in class_names]
    class_names = [cls_n.replace("(", "") for cls_n in class_names]
    class_names = [cls_n.replace(")", "") for cls_n in class_names]
    return class_names

def pre_tokenize(class_names):
    """
    pre-tokenize class names
    :param class_names: List, a list of class names
    :param tokenizer: Tokenizer, SimpleTokenizer()
    :return: Tensor, containing all prompts for all classes, [#cls, #prompts, context_length]
    """
    # tokenizer
    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]    

    # prompt engineering
    prompt_templates = get_prompt_templates()
    input_ids_all = []
    for k in range(len(class_names)):
        v = class_names[k]
        if isinstance(v, str):
            vs = [v]
        elif isinstance(v, list):
            vs = v
        t1s = []
        for v in vs:
            for pt in prompt_templates:
                t1s.append(prompt_engineering(v, template=pt))
        input_ids = []
        for t1 in t1s:
            this_input_ids = convert_example_to_features_bpe(t1, tokenizer, sot_token, eot_token)                                                        
            input_ids.append(torch.tensor(this_input_ids, dtype=torch.long))

        input_ids_all.append(torch.stack(input_ids, 0))

    input_ids_all_classes = torch.stack(input_ids_all, 0)
    return input_ids_all_classes


if __name__ == "__main__":
    flatten_input_ids = pre_tokenize()
