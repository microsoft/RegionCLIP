# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import os
import torch
import torch.utils.data
import torch.distributed
from torch.utils.data.dataset import ConcatDataset

from .catalog import DatasetCatalog
from .clip_datasets.clip_img_txt_pair_tsv import CLIPImgTxtPairTSVDataset

from .transforms.build import build_clip_transforms

def config_tsv_dataset_args(cfg, dataset_file, factory_name=None, is_train=True):
    ############### code removecd as tsv_dataset_name = factory_name = "CLIPImgTxtPairTSVDataset" ##############
    if factory_name is not None:
        tsv_dataset_name = factory_name

    if tsv_dataset_name in ["CLIPImgTxtPairTSVDataset"]:
        # no need for extra arguments
        args = {}
        args['args'] = cfg
        args['seq_len'] = cfg.DATASETS.MAX_SEQ_LENGTH # cfg.max_seq_length

    return args, tsv_dataset_name


def build_dataset(cfg, transforms, dataset_catalog, is_train=True, is_aux=False):
    """
    Arguments:
        cfg: config file.
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """

    dataset_list = (cfg.DATASETS.TRAIN if not is_aux else cfg.DATASETS.AUX) if is_train else cfg.DATASETS.TEST
    factory_list = (cfg.DATASETS.FACTORY_TRAIN if not is_aux else cfg.DATASETS.FACTORY_AUX) if is_train else cfg.DATASETS.FACTORY_TEST
    path_list = (cfg.DATASETS.PATH_TRAIN if not is_aux else cfg.DATASETS.PATH_AUX) if is_train else cfg.DATASETS.PATH_TEST

    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
                "dataset_list should be a list of strings, got {}".format(dataset_list))
    if not isinstance(factory_list, (list, tuple)):
        raise RuntimeError(
                "factory_list should be a list of strings, got {}".format(factory_list))
    datasets = []
    target_offset = 0
    for i, dataset_name in enumerate(dataset_list):
        factory_name = factory_list[i] if i < len(factory_list) else None

        if factory_name == "CLIPImgTxtPairTSVDataset":
            dataset_names_merged = dataset_name.split('+')
            path_lists_merged = path_list[i].split('+')

            assert len(dataset_names_merged) == len(path_lists_merged), "number of datasets must match that of dataset paths"

            image_tsv_list = []
            text_tsv_list = []
            dataset_name_list = []  
            map_files = []
            max_num_tsv = 20  # maximum tsv files to load within a given folder        

            for dname, dpath in zip(dataset_names_merged, path_lists_merged):            
                args, tsv_dataset_name = config_tsv_dataset_args(
                    cfg, dataset_name, factory_name, is_train
                )
                factory = CLIPImgTxtPairTSVDataset if tsv_dataset_name in ["CLIPImgTxtPairTSVDataset"] else None
                prev_len = len(image_tsv_list)

                isFile = os.path.isfile(dpath) 
                if isFile:
                    dpath_listed_files = [os.path.basename(dpath)]
                    dpath = os.path.dirname(dpath)
                else:
                    dpath_listed_files = sorted(os.listdir(dpath))

                for filename in dpath_listed_files:
                    if ("images" in filename or "image" in filename or "img" in filename) and filename.endswith(".tsv"):
                        image_tsv_list.append(os.path.join(dpath, filename))     
                        if "images" in filename: # "images" - "text"
                            text_tsv_list.append(os.path.join(dpath, filename.replace("images", "text")))
                        elif "image" in filename: # "image"-"text"
                            text_tsv_list.append(os.path.join(dpath, filename.replace("image", "text")))
                        elif "img" in filename: # "img"-"caption"
                            text_tsv_list.append(os.path.join(dpath, filename.replace("img", "caption")))
                        if len(image_tsv_list) - prev_len == max_num_tsv:
                            break                                                        
                dataset_name_list += [dname] * (len(image_tsv_list) - prev_len)

                if dname == "imagenet22k":
                    map_files += [os.path.join(dpath, 'darknet_data_imagenet.labels.list')] * (len(image_tsv_list) - prev_len)
                else:
                    map_files += [None] * (len(image_tsv_list) - prev_len)

                assert len(image_tsv_list) == len(text_tsv_list), \
                    "the number image tsv files must be equal to that of text tsv files, otherwise check your data!"                

            args["image_tsv_file"] = image_tsv_list
            args["text_tsv_file"] = text_tsv_list
            args["dataset_name"] = dataset_name_list
            args["map_file"] = map_files                           
            args["filtered_datasets"] = cfg.DATASETS.FILTERED_CLASSIFICATION_DATASETS
            assert len(image_tsv_list) == len(text_tsv_list) == len(dataset_name_list) == len(map_files)

            print("number of image tsv files: ", len(image_tsv_list))
            print("number of text tsv fies: ", len(text_tsv_list))
                
        args["is_train"] = is_train
        args["transforms"] = transforms
        args["target_offset"] = target_offset
        if "bpe" in cfg.INPUT.TEXT_TOKENIZER:
            from detectron2.data.datasets.clip_prompt_utils import SimpleTokenizer as _Tokenizer
            tokenizer = _Tokenizer()                
            args["tokenizer_type"] = "bpe"
        args["tokenizer"] = tokenizer
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    precomputed_tokens = {}
    dataset_classes = {}
    for dataset in datasets:
        if hasattr(dataset, "input_ids_all_classes"):
            precomputed_tokens["imagenet"] = \
                [dataset.input_ids_all_classes, dataset.input_mask_all_classes, dataset.segment_ids_all_classes]
        if hasattr(dataset, "classnames"):
            if isinstance(dataset.classnames, dict):
                dataset_classes.update(dataset.classnames)
            else:
                dataset_classes[dataset.dataset_name] = dataset.classnames

    # for testing, return a list of datasets
    if not is_train:
        return datasets, precomputed_tokens, dataset_classes

    if len(datasets) == 0:
        return None, None, None

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    return [dataset], precomputed_tokens, dataset_classes


def make_clip_dataset(cfg, is_train=True, is_aux=False, transforms=None):
    if transforms is None:
        transforms = build_clip_transforms(cfg, is_train)
    print("data transforms: ")
    print(transforms)
    datasets, precomputed_tokens, dataset_classes = build_dataset(cfg, transforms, DatasetCatalog, is_train, is_aux)

    if not datasets:
        return None, None, None
    return datasets, precomputed_tokens, dataset_classes