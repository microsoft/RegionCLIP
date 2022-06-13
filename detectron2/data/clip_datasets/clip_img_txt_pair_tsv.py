from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO
import json
import logging
import base64
import threading
import random
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.utils.data as data
from .clip_tsv import InputExample, convert_example_to_features
from detectron2.structures.tsv_file import TSVFile, CompositeTSVFile
from detectron2.data.clip_datasets.clip_prompt_engineering import get_prompt_templates, prompt_engineering
#import spacy

def pre_fetch(tsv_filename: str):
    logging.info('Pre-loading %s ...' % tsv_filename)
    with open(tsv_filename, 'r'):
        logging.info('Pre-loading %s ended.' % tsv_filename)

class CLIPImgTxtPairTSVDataset(data.Dataset):
    """ This class is intended for encapsulating Image/Text pair data for contrastive learning.
    """
    def __init__(self,
                 image_tsv_file: Union[str, List[str]],
                 text_tsv_file: Union[str, List[str]],
                 transforms: Callable = None,
                 tokenizer: Callable = None,
                 seq_len = 0, context_length = 77, target_offset=0, 
                 args = None, 
                 dataset_name = "", 
                 tokenizer_type = "bert",
                 is_train = True,
                 map_file = None, 
                 filtered_datasets = ''):
        self.args = args
        self.is_train = is_train    
        self.dataset_names = dataset_name
        self.tokenizer_type = tokenizer_type
        self.target_offset = target_offset
        self.seq_len = seq_len

        self.transforms = transforms
        self.tokenizer = tokenizer
        self._chunk_sizes = None
        self.context_length = context_length
        
        self.prompt_templates = get_prompt_templates() # [:2]
        self.spacy_nlp = None # spacy.load('en_core_web_sm')
        
        self.class_selector = None

        self.label2idx = {}
        self.idx2label = {}        
        self.classnames = {}
        self.dataset_target_offsets = {}; offset = 0

        self.num_classes = sum([len(val) for val in self.classnames.values()])          

        self.filtered_classnames = []

        if isinstance(image_tsv_file, str) and isinstance(text_tsv_file, str):
            # single tsv file
            if (
                os.path.splitext(image_tsv_file)[1].lower() == '.tsv'
                and os.path.splitext(text_tsv_file)[1].lower() == '.tsv'
            ):
                self.image_tsv_file = TSVFile(image_tsv_file, if_generate_lineidx=True)
                self.text_tsv_file = TSVFile(text_tsv_file, if_generate_lineidx=True)
            # multiple tsv files specified in a text file
            elif (
                os.path.splitext(image_tsv_file)[1].lower() == '.txt'
                and os.path.splitext(text_tsv_file)[1].lower() == '.txt'
            ):
                self.image_tsv_file = CompositeTSVFile(image_tsv_file)
                self.text_tsv_file = CompositeTSVFile(text_tsv_file)
                self._chunk_sizes = self.image_tsv_file.get_chunk_size()
            else:
                raise ValueError("Invalid input! Please check the tsv filenames.")
        # multiple tsv files specified in a list
        elif (
            isinstance(image_tsv_file, list)
            and isinstance(text_tsv_file, list)
        ):
            assert len(image_tsv_file) == len(text_tsv_file), \
                "Inconsistent number of Image/Text tsv files!"
            assert len(image_tsv_file) == len(text_tsv_file), \
                "Inconsistent number of Image/Text tsv files!"
            self.image_tsv_path = image_tsv_file
            self.text_tsv_path = text_tsv_file            
            self.image_tsv_file = CompositeTSVFile(image_tsv_file, class_selector=self.class_selector)
            self.text_tsv_file = CompositeTSVFile(text_tsv_file, class_selector=self.class_selector)
            self._chunk_sizes = self.image_tsv_file.get_chunk_size()
            self._accumulated_chunk_sizes = np.cumsum(self._chunk_sizes).tolist()
        else:
            raise ValueError("Invalid input! Please check the tsv filenames.")
        
        assert len(self.image_tsv_file) == len(self.text_tsv_file), \
            "Inconsistent size of Image/Text ({}/{}) data!".format(
                len(self.image_tsv_file), len(self.text_tsv_file)
            )
    
    def get_chunk_sizes(self):
        return self._chunk_sizes

    def get_class_boundaries(self):
        # The samples of each class are organized class-by-class.
        # _class_boundaries stores the lower- and upper-bound of each class.
        return self.image_tsv_file.get_class_boundaries()

    def _load_map(self, map_file: str):
        if not map_file:
            return None

        label2idx = {}
        with open(map_file) as f:
            for line in f:
                items = line.strip().split('\t')
                label2idx[items[0]] = int(items[1])

        return label2idx
    
    def _load_darknet_map(self, map_file):
        if not map_file:
            return None

        label2idx = {}
        with open(map_file) as f:
            linenum = 0
            for l in f:
                item = l.strip()
                label2idx[item] = linenum
                linenum += 1

        return label2idx    

    def _pre_tokenize(self):
        """
        pre-tokenize class names
        """
        input_ids_all = []
        input_masks_all = []
        segment_ids_all = []
        for k in range(len(self.classnames["imagenet"])):
            cur_id = 0; img_id = 0
            scale = 1.0

            v = self.classnames["imagenet"].label_to_name(k)
            if isinstance(v, str):
                vs = [v]
            elif isinstance(v, list):
                vs = v
            t1s = []
            t2s = []
            for v in vs:
                for pt in self.prompt_templates:
                    t1s.append(prompt_engineering(v, template=pt))
                    t2s.append("")
            input_ids = []
            input_masks = []
            segment_ids = []
            is_next_labels = [0] * len(t1s)
            is_img_matchs = [1] * len(t1s)
            img_feat_len = 0
            for t1, t2, is_next_label, is_img_match in zip(t1s, t2s, is_next_labels, is_img_matchs):
                if self.tokenizer_type == "bert":
                    # tokenize
                    tokens_a = self.tokenizer.tokenize(t1)
                    tokens_b = None

                    # combine to one sample
                    cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                            tokens_b=tokens_b, is_next=is_next_label,
                                            img_id=img_id, is_img_match=is_img_match)

                    # transform sample to features
                    cur_features = convert_example_to_features(self.args, cur_example,
                                                            self.seq_len, self.tokenizer,
                                                            img_feat_len)                                                        

                    input_ids.append(torch.tensor(cur_features.input_ids, dtype=torch.long))
                    input_masks.append(torch.tensor(cur_features.input_mask, dtype=torch.long))
                    segment_ids.append(torch.tensor(cur_features.segment_ids, dtype=torch.long))

                elif self.tokenizer_type == "bpe":
                    tokens_a = t1; tokens_b = None
                    # combine to one sample
                    cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                            tokens_b=tokens_b, is_next=is_next_label,
                                            img_id=img_id, is_img_match=is_img_match)

                    # transform sample to features
                    cur_features = convert_example_to_features_bpe(self.args, cur_example,
                                                            self.seq_len, self.tokenizer,
                                                            img_feat_len)                                                        

                    input_ids.append(torch.tensor(cur_features.input_ids, dtype=torch.long))
                    input_masks.append(torch.tensor(cur_features.input_mask, dtype=torch.long))
                    segment_ids.append(torch.tensor(cur_features.segment_ids, dtype=torch.long))

                else:
                    raise NotImplementedError
            input_ids_all.append(torch.stack(input_ids, 0))
            input_masks_all.append(torch.stack(input_masks, 0))
            segment_ids_all.append(torch.stack(segment_ids, 0))

        self.input_ids_all_classes = torch.stack(input_ids_all, 0)
        self.input_mask_all_classes = torch.stack(input_masks_all, 0)
        self.segment_ids_all_classes = torch.stack(segment_ids_all, 0)

    def _online_tokenize(self, text):

        # random select a prompt template
        temp_idx = np.random.randint(len(self.prompt_templates))
        pt = self.prompt_templates[temp_idx]

        names = text.split(";")
        num_names = np.random.randint(len(names)) + 1
        names_sampled = random.sample(names, num_names)
        text = ", ".join(names_sampled)    

        t1 = prompt_engineering(text, template=pt)

        cur_id = 0; img_id = 0; scale = 1.0
        is_next_label = 0; is_img_match = 1
        img_feat_len = 0

        if self.tokenizer_type == "bert":
            # tokenize
            tokens_a = self.tokenizer.tokenize(t1)
            tokens_b = None

            # combine to one sample
            cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                    tokens_b=tokens_b, is_next=is_next_label,
                                    img_id=img_id, is_img_match=is_img_match)

            # transform sample to features
            cur_features = convert_example_to_features(self.args, cur_example,
                                                    self.context_length, self.tokenizer,
                                                    img_feat_len)                                                        


        elif self.tokenizer_type == "bpe":
            tokens_a = t1; tokens_b = None
            # combine to one sample
            cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                    tokens_b=tokens_b, is_next=is_next_label,
                                    img_id=img_id, is_img_match=is_img_match)

            # transform sample to features
            cur_features = convert_example_to_features_bpe(self.args, cur_example,
                                                    self.context_length, self.tokenizer,
                                                    img_feat_len)                                                              

        return torch.tensor(cur_features.input_ids, dtype=torch.long), \
                torch.tensor(cur_features.input_mask, dtype=torch.long), \
                torch.tensor(cur_features.segment_ids, dtype=torch.long)

    def get_dataset_name(self, index):
        """
        get dataset name according to index
        """
        assert index < self._accumulated_chunk_sizes[-1], "index must in the range of accumulated data size"
        for k, boundary in enumerate(self._accumulated_chunk_sizes):
            if index < boundary:
                return self.dataset_names[k], k

    def get_target_offset(self, dataset_name):
        return self.dataset_target_offsets[dataset_name]

    def get_img_label_pair(self, items_image, index):             
        dataset_name, chunk_id = self.get_dataset_name(index)
        target_offset = self.get_target_offset(dataset_name)
        _, target, img = self._decode_data(items_image, dataset_name)   

        if self.transforms:
            img = self.transforms(img)

        if target == -1:
            input_ids, input_mask, segment_ids = \
                self._online_tokenize("uncovered image")                    
        else:
            classname = self.classnames[dataset_name].labels2names[self.idx2label[dataset_name][target]]
            if classname in self.filtered_classnames:
                # we filter these classnames for training
                target = -1
                input_ids, input_mask, segment_ids = \
                    self._online_tokenize("uncovered image")                  
            else:            
                input_ids, input_mask, segment_ids = \
                    self._online_tokenize(classname)
                target += target_offset
        return img, \
            input_ids, \
            input_mask, \
            segment_ids, \
            torch.LongTensor([target]), \
            dataset_name                          
        
    def get_img_txt_pair(self, items_image, items_text, index):
        dataset_name, chunk_id = self.get_dataset_name(index)
        assert items_text[0] == items_image[0], \
            'keys do not match for image ({}) and text ({}) for {} at chunk {}-{}'.format(
                len(items_text[0]), len(items_image[0]), dataset_name, chunk_id, self.image_tsv_path[chunk_id]
            )

        img = self._decode_image(items_image, dataset_name)
        #     print("index {}, chunk id {}, name {}".format(index, chunk_id, self.image_tsv_path[chunk_id]))
        #     raise TypeError("cannot decode current item") 
        img_width, img_height = img.size  # img_height, img_width = np.array(img).shape

        txts = self._decode_text(items_text)
        if self.spacy_nlp is not None:
            np_input_ids, np_input_masks, np_segment_ids = self.create_phrase_text(txts)

        if self.transforms:
            img = self.transforms(img)

        if isinstance(txts, str):
            input_ids, input_masks, segment_ids = \
                convert_txt_to_tokens_bpe(txts, self.tokenizer, self.context_length)
            all_str2id_links = []
        elif isinstance(txts, list):
            input_ids = []
            input_masks = []
            segment_ids = []
            all_str2id_links = []
            for txt in txts:
                input_id, input_mask, segment_id, str2id_links = \
                    convert_txt_to_tokens_bpe(txt, self.tokenizer, self.context_length, return_link=True)
                input_ids += input_id          
                input_masks += input_mask
                segment_ids += segment_id
                all_str2id_links += [str2id_links]
        scale = 1.0
        img_id = 0
        
        if self.spacy_nlp is not None:
            return img, \
                    torch.tensor(input_ids).long().view(-1), \
                    torch.tensor(input_masks).long().view(-1), \
                    torch.tensor(segment_ids).long().view(-1), \
                    torch.LongTensor([1e5]), \
                    dataset_name, \
                    torch.tensor(np_input_ids).long().view(-1), \
                    torch.tensor(np_input_masks).long().view(-1), \
                    torch.tensor(np_segment_ids).long().view(-1)
        else:
            return img, \
                    torch.tensor(input_ids).long().view(-1), \
                    torch.tensor(input_masks).long().view(-1), \
                    torch.tensor(segment_ids).long().view(-1), \
                    torch.LongTensor([1e5]), \
                    (dataset_name, items_text[0], (img_height, img_width), all_str2id_links)  # dataset name, image id, image height&width, links bet string and tokenized texts

    def create_phrase_text(self, txt_list):
        """ Use NLP tool to detect noun phrases in captions, fill each identified phrase into a random prompt to create a sentence,
            and convert each sentence to bpe tokens
        """
        if isinstance(txt_list, str):
            txt_list = [txt_list]
        # detect noun phrase
        noun_phrase = []
        for txt in txt_list:
            doc = self.spacy_nlp(txt.lower())
            this_text = [nc.text for nc in doc.noun_chunks]
            this_text = [nc.replace('a ', '').replace('the ', '') for nc in this_text]
            noun_phrase.extend(this_text)
        noun_phrase = list(set(noun_phrase))
        # fill each phrase into a random prompt
        text_list = []
        pts = random.sample(self.prompt_templates, len(noun_phrase))
        for i, np in enumerate(noun_phrase):
            text_list.append(prompt_engineering(np, pts[i]))
        # convert string into bpe tokens
        input_ids = []
        input_masks = []
        segment_ids = []
        for txt in text_list:
            input_id, input_mask, segment_id = \
                convert_txt_to_tokens_bpe(txt, self.tokenizer, self.context_length)
            input_ids += input_id          
            input_masks += input_mask
            segment_ids += segment_id
        return input_ids, input_masks, segment_ids

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if isinstance(index, tuple):
            items_image = self.image_tsv_file[index[0]]
            items_text = self.text_tsv_file[index[0]]
            if index[1] >= 0:
                tsv_filename = self.image_tsv_file.file_list[index[1]]

                # Python threads are not truly parallel. Spawn a new process instead.
                # logging.info('Pre-loading %s ...' % tsv_filename)
                # os.system('cat ' + tsv_filename + ' > /dev/null &')
                x = threading.Thread(
                   target=pre_fetch, args=(tsv_filename,), daemon=True
                )
                x.start()
            curr_index = index[0]
        else:
            items_image = self.image_tsv_file[index]
            items_text = self.text_tsv_file[index]
            curr_index = index
        
        # NOTE: since we duplicate image tsv to text tsv for image-label data,
        # we can determine whether the current instance is an image-label pair or 
        # a image-text pair data based on whether items_image is identical to items_text or not.
        if items_image == items_text:
            return self.get_img_label_pair(items_image, curr_index)
        else:
            return self.get_img_txt_pair(items_image, items_text, curr_index)
            
    def _decode_image(self, items: Tuple[str, str], dataset_name=""):
        key = items[0]
        image = Image.open(BytesIO(base64.b64decode(items[1]))).convert('RGB')
        return image

    def _decode_text(self, items: Tuple[str, Union[str, dict]]):
        key = items[0]
        text = ''
        if isinstance(items[1], str):
            try:
                str_dict = json.loads(items[1])
                # in this dict, it may contain either "tags" or "captions" or both
                keys = [key for key in str_dict.keys()]
                selected_key = random.sample(keys, 1)[0]
                if selected_key == "captions":
                    # if this is a caption, we sample a caption
                    captions = str_dict[selected_key]
                    text = captions[:5]
                    # text = random.sample(captions, 1)[0]
                elif selected_key == "tags":
                    # for tags, we randomly disorder it
                    tags = str_dict[selected_key]
                    tag_words = tags.split(' ')
                    random.shuffle(tag_words)
                    tags_shuffled = " ".join(tag_words)
                    # add prompt template
                    pt = random.sample(self.prompt_templates, 1)[0]
                    text = prompt_engineering(tags_shuffled, pt)
            except:
                text = items[1]
        elif isinstance(items[1], dict):
            assert 'captions' in items[1], '"captions" does not in {}'.format(items[1])
            captions = items[1]['captions']
            if isinstance(captions, list):
                text = random.choice(captions)
            elif isinstance(captions, str):
                text = captions
            else:
                raise ValueError('captions should be str or list')

        return text

    def _decode_data(self, items, dataset_name):
        key = items[0]
        label = self._get_label(items[1], dataset_name)
        try:
            image = Image.open(BytesIO(base64.b64decode(items[2])))
        except:
            return None

        return key, label, image.convert('RGB')

    def _get_label(self, item, dataset_name):
        if not self.label2idx[dataset_name]:
            return int(item)

        if item in self.label2idx[dataset_name]:
            return self.label2idx[dataset_name][item]

        label = json.loads(item)[0]['class']
        if label in self.label2idx[dataset_name]:
            return self.label2idx[dataset_name][label]
        else:
            return -1

    def __len__(self):
        return len(self.image_tsv_file)

def convert_txt_to_tokens_bpe(text, tokenizer, context_length, return_link=False):

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    if return_link:
        bpe_tokens, str2id_links = tokenizer.encode(text, return_link=return_link)
        str2id_links = [["<|startoftext|>", [sot_token]]] + str2id_links + [["<|endoftext|>", [eot_token]]]
    else:
        bpe_tokens = tokenizer.encode(text, return_link=return_link)
    input_ids = [sot_token] + bpe_tokens + [eot_token]

    if len(input_ids) > context_length:
        input_ids = input_ids[:context_length]
    segment_ids = [0] * len(input_ids)
    lm_label_ids = [-1] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < context_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == context_length
    assert len(input_mask) == context_length
    assert len(segment_ids) == context_length
    assert len(lm_label_ids) == context_length    
    
    if return_link:
        return input_ids, input_mask, segment_ids, str2id_links
    return input_ids, input_mask, segment_ids

def convert_example_to_features_bpe(args, example, max_seq_length, tokenizer,
                                img_feat_len, context_length=77):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    # we do not consider tokens_b for now in original CLIP
    text = example.tokens_a
    assert isinstance(text, str)

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    input_ids = [sot_token] + tokenizer.encode(text) + [eot_token]

    if len(input_ids) > context_length:
        input_ids = input_ids[:context_length]
    segment_ids = [0] * len(input_ids)
    lm_label_ids = [-1] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < context_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == context_length
    assert len(input_mask) == context_length
    assert len(segment_ids) == context_length
    assert len(lm_label_ids) == context_length

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("LM label: %s " % lm_label_ids)
        logging.info("Is next sentence label: %s " % example.is_next)

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             img_feat_len=img_feat_len,
                             is_img_match=example.is_img_match)
    return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next,
                 lm_label_ids, img_feat_len, is_img_match):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match    