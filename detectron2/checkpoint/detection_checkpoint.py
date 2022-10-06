# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts
from .clip_model_loading import align_and_update_state_dicts_for_CLIP

def interpolate_pos_embed(model, checkpoint_model):
    if 'backbone.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['backbone.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.backbone.patch_embed.num_patches
        num_extra_tokens = model.backbone.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens.type_as(extra_tokens)), dim=1)
            checkpoint_model['backbone.pos_embed'] = new_pos_embed

class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, bb_rpn_weights=False, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager
        self.bb_rpn_weights = bb_rpn_weights

    def load(self, path, *args, **kwargs):
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            logger = logging.getLogger(__name__)
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", "matching_heuristics": True}
        elif "OAI_CLIP" in filename:
            # assume file is from OpenAI CLIP pre-trained model
            loaded = super()._load_file(filename)  # load native pth checkpoint
            if "model" not in loaded:
                loaded = {"model": loaded}
            return {"model": loaded["model"], "__author__": "OAI_CLIP", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False) or self.bb_rpn_weights:
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            if checkpoint.get("__author__", "NA") == "OAI_CLIP" or self.bb_rpn_weights:  # for OAI_CLIP or 2nd ckpt (offline modules)
                checkpoint["model"] = align_and_update_state_dicts_for_CLIP(
                    self.model.state_dict(),
                    checkpoint["model"],
                    bb_rpn_weights=self.bb_rpn_weights,
                )
            else:  # default loading
                checkpoint["model"] = align_and_update_state_dicts(
                    self.model.state_dict(),
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )

        # mapping keys
        
        # for swin
        new_ckpt = {key.replace('image_encoder.', 'backbone.'): val for key, val in checkpoint['model'].items()}
        new_ckpt = {key.replace('text_encoder.', 'roi_heads.box_predictor.lang_encoder.lang_encoder.') if key.startswith('text_encoder.') else key: val for key, val in new_ckpt.items()}
        new_ckpt = {key.replace('text_projection', 'roi_heads.box_predictor.lang_encoder.lang_proj'): val for key, val in new_ckpt.items()}
        new_ckpt = {key.replace('logit_scale', 'roi_heads.box_predictor.lang_encoder.logit_scale') if key == 'logit_scale' else key: val for key, val in new_ckpt.items()}

        # for focalnet
        new_ckpt = {(key.replace('norm', 'backbone.norm') if key.startswith('norm.') else key): val for key, val in new_ckpt.items()}
        new_ckpt = {(key.replace('lang_encoder.lang_encoder.', 'roi_heads.box_predictor.lang_encoder.lang_encoder.') if key.startswith('lang_encoder.lang_encoder.') else key): val for key, val in new_ckpt.items()}
        new_ckpt = {key.replace('lang_encoder.logit_scale', 'roi_heads.box_predictor.lang_encoder.logit_scale') if key == 'lang_encoder.logit_scale' else key: val for key, val in new_ckpt.items()}
        new_ckpt = {key.replace('lang_encoder.lang_proj', 'roi_heads.box_predictor.lang_encoder.lang_proj') if key == 'lang_encoder.lang_proj' else key: val for key, val in new_ckpt.items()}

        # for vit
        new_ckpt = {(key.replace('lang_encoder.', 'roi_heads.box_predictor.lang_encoder.lang_encoder.') if key.startswith('lang_encoder.') else key): val for key, val in new_ckpt.items()}
        new_ckpt = {(key.replace('lang_projection', 'roi_heads.box_predictor.lang_encoder.lang_proj') if key == 'lang_projection' else key): val for key, val in new_ckpt.items()}


        checkpoint['model'] = new_ckpt

        interpolate_pos_embed(self.model, checkpoint['model'])

        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)
        del checkpoint  # try saving memory

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible