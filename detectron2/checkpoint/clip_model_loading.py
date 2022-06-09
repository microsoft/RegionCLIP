# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import re
from typing import Dict, List
import torch
from tabulate import tabulate


def convert_basic_clip_names(original_keys, add_backbone_prefix=False, use_whole_clip=False, use_fpn_arch=False, regionclip=False):
    """
    Apply some basic name conversion to names in CLIP weights.
    It only deals with typical backbone models.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    
    vit = False
    for l_k in layer_keys:
        if 'visual.transformer' in l_k:
            vit = True
    
    # load pretrained oai clip
    if not vit: # resnet
        if add_backbone_prefix: # CLIPRCNN or CLIPFastRCNN
            if use_whole_clip: # CLIPRCNN
                layer_keys = [k.replace("visual.", "clip_backbone.visual.") for k in layer_keys]
            else:  # CLIPFastRCNN
                if use_fpn_arch: # FPN
                    layer_keys = [k.replace("visual.", "backbone.bottom_up.") for k in layer_keys]
                else: # C4
                    layer_keys = [k.replace("visual.", "backbone.") for k in layer_keys] 
        else:  # GeneralizedRCNN or ProposalNetwork
            #layer_keys = [k.replace("visual.", "backbone.bottom_up.") for k in layer_keys] #
            layer_keys = [k.replace("visual.", "") for k in layer_keys] # 
            #layer_keys = [k.replace("visual.", "backbone.visual.") for k in layer_keys] #
    else: # vit
        pass

    return layer_keys, vit


def convert_clip_names(weights, add_backbone_prefix=False, use_whole_clip=False, use_fpn_arch=False, regionclip=False):
    """
    Map CLIP Detectron weight names to Detectron2 names.

    Args:
        weights (dict): name -> tensor

    Returns:
        dict: detectron2 names -> tensor
        dict: detectron2 names -> C2 names
    """
    logger = logging.getLogger(__name__)
    logger.info("Renaming CLIP weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys, use_vit = convert_basic_clip_names(layer_keys, add_backbone_prefix, use_whole_clip, use_fpn_arch, regionclip)

    # --------------------------------------------------------------------------
    # RPN hidden representation conv
    # --------------------------------------------------------------------------
    # FPN case
    # In the C2 model, the RPN hidden layer conv is defined for FPN level 2 and then
    # shared for all other levels, hence the appearance of "fpn2"
    layer_keys = [
        k.replace("conv.rpn.fpn2", "proposal_generator.rpn_head.conv") for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [k.replace("conv.rpn", "proposal_generator.rpn_head.conv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # RPN box transformation conv
    # --------------------------------------------------------------------------
    # FPN case (see note above about "fpn2")
    layer_keys = [
        k.replace("rpn.bbox.pred.fpn2", "proposal_generator.rpn_head.anchor_deltas")
        for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits.fpn2", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [
        k.replace("rpn.bbox.pred", "proposal_generator.rpn_head.anchor_deltas") for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]

    # --------------------------------------------------------------------------
    # Fast R-CNN box head
    # --------------------------------------------------------------------------
    layer_keys = [re.sub("^bbox\\.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls\\.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6\\.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7\\.", "box_head.fc2.", k) for k in layer_keys]
    # 4conv1fc head tensor names: head_conv1_w, head_conv1_gn_s
    layer_keys = [re.sub("^head\\.conv", "box_head.conv", k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # FPN lateral and output convolutions
    # --------------------------------------------------------------------------
    def fpn_map(name):
        """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
        splits = name.split(".")
        norm = ".norm" if "norm" in splits else ""
        if name.startswith("fpn.inner."):
            # splits example: ['fpn', 'inner', 'res2', '2', 'sum', 'lateral', 'weight']
            stage = int(splits[2][len("res") :])
            return "fpn_lateral{}{}.{}".format(stage, norm, splits[-1])
        elif name.startswith("fpn.res"):
            # splits example: ['fpn', 'res2', '2', 'sum', 'weight']
            stage = int(splits[1][len("res") :])
            return "fpn_output{}{}.{}".format(stage, norm, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # Mask R-CNN mask head
    # --------------------------------------------------------------------------
    # roi_heads.StandardROIHeads case
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [re.sub("^\\.mask\\.fcn", "mask_head.mask_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    # roi_heads.Res5ROIHeads case
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Keypoint R-CNN head
    # --------------------------------------------------------------------------
    # interestingly, the keypoint head convs have blob names that are simply "conv_fcnX"
    layer_keys = [k.replace("conv.fcn", "roi_heads.keypoint_head.conv_fcn") for k in layer_keys]
    layer_keys = [
        k.replace("kps.score.lowres", "roi_heads.keypoint_head.score_lowres") for k in layer_keys
    ]
    layer_keys = [k.replace("kps.score.", "roi_heads.keypoint_head.score.") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)
    assert len(original_keys) == len(layer_keys)

    new_weights = {}
    new_keys_to_original_keys = {}
    for orig, renamed in zip(original_keys, layer_keys):
        new_keys_to_original_keys[renamed] = orig
        if renamed.startswith("bbox_pred.") or renamed.startswith("mask_head.predictor."):
            # remove the meaningless prediction weight for background class
            new_start_idx = 4 if renamed.startswith("bbox_pred.") else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.info(
                "Remove prediction weight for background class in {}. The shape changes from "
                "{} to {}.".format(
                    renamed, tuple(weights[orig].shape), tuple(new_weights[renamed].shape)
                )
            )
        elif renamed.startswith("cls_score."):
            # move weights of bg class from original index 0 to last index
            logger.info(
                "Move classification weights for background class in {} from index 0 to "
                "index {}.".format(renamed, weights[orig].shape[0] - 1)
            )
            new_weights[renamed] = torch.cat([weights[orig][1:], weights[orig][:1]])
        else:
            new_weights[renamed] = weights[orig]

    return new_weights, new_keys_to_original_keys, use_vit


# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts_for_CLIP(model_state_dict, ckpt_state_dict, c2_conversion=True, bb_rpn_weights=False, regionclip=False):
    """
    Extended from ./c2_model_loading.py
    Match names between the two state-dict, and returns a new chkpt_state_dict with names
    converted to match model_state_dict with heuristics. The returned dict can be later
    loaded with fvcore checkpointer.
    If `c2_conversion==True`, `ckpt_state_dict` is assumed to be a Caffe2
    model and will be renamed at first.

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(model_state_dict.keys())
    use_whole_clip = False  # whether use the whole clip (text & visual encoders), typically in CLIPRCNN meta arch
    add_backbone_prefix = False  # convert to 'backbone.' prefix, typically in CLIPFastRCNN meta arch
    use_fpn_arch = False # if use FPN arch then convert to `bottom_up`, typically in CLIPFastRCNN meta arch with FPN backbone
    if bb_rpn_weights: # a 2nd pretrained weights to load, for offline backbone & RPN, then convert the ckpt key names and only keep the ones we need
        new_ckpt_state_dict = {}
        for original_k in ckpt_state_dict:
            if 'backbone' in original_k:
                new_key = original_k.replace('backbone', 'offline_backbone')
                new_ckpt_state_dict[new_key] = ckpt_state_dict[original_k]
            if 'proposal_generator' in original_k:
                new_key = original_k.replace('proposal_generator', 'offline_proposal_generator')
                new_ckpt_state_dict[new_key] = ckpt_state_dict[original_k]
        new_ckpt_state_dict['ignore_others'] = torch.tensor([1]) # ignore other model weights (not 'offline_*') in batch_norm.py
        ckpt_state_dict = new_ckpt_state_dict
    else: # the 1st pretrained weigths to load
        for model_key in model_keys: # if use the whole clip, then convert ckpt 'visual.' names to 'clip_backbone.visual.'
            if 'clip_backbone' in model_key:
                use_whole_clip = True
        for model_key in model_keys: # if there are backbone & offline_backbone, then convert the ckpt 'visual.' names to 'backbone.' to avoid ambiguity
            if 'offline_backbone' in model_key:
                add_backbone_prefix = True
            if 'fpn' in model_key:
                use_fpn_arch = True
    # original_keys: the name in the original dict (before renaming)
    ckpt_state_dict, original_keys, use_vit = convert_clip_names(ckpt_state_dict, add_backbone_prefix, use_whole_clip, use_fpn_arch, regionclip)
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    logger = logging.getLogger(__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    result_state_dict = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            logger.warning(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        assert key_model not in result_state_dict
        result_state_dict[key_model] = value_ckpt
        if key_ckpt in matched_keys:  # already added to matched_keys
            logger.error(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model

    # logging:
    matched_model_keys = sorted(matched_keys.values())
    mmk_list = "The following model parameters are loaded from checkpoints:\n"
    for mmk in matched_model_keys:
        mmk_list += mmk + "\n"
    if len(matched_model_keys) == 0:
        logger.warning("No weights in checkpoint matched with model.")
        return ckpt_state_dict
    common_prefix = _longest_common_prefix(matched_model_keys)
    rev_matched_keys = {v: k for k, v in matched_keys.items()}
    original_keys = {k: original_keys[rev_matched_keys[k]] for k in matched_model_keys}

    model_key_groups = _group_keys_by_module(matched_model_keys, original_keys)
    table = []
    memo = set()
    for key_model in matched_model_keys:
        if key_model in memo:
            continue
        if key_model in model_key_groups:
            group = model_key_groups[key_model]
            memo |= set(group)
            shapes = [tuple(model_state_dict[k].shape) for k in group]
            table.append(
                (
                    _longest_common_prefix([k[len(common_prefix) :] for k in group]) + "*",
                    _group_str([original_keys[k] for k in group]),
                    " ".join([str(x).replace(" ", "") for x in shapes]),
                )
            )
        else:
            key_checkpoint = original_keys[key_model]
            shape = str(tuple(model_state_dict[key_model].shape))
            table.append((key_model[len(common_prefix) :], key_checkpoint, shape))
    table_str = tabulate(
        table, tablefmt="pipe", headers=["Names in Model", "Names in Checkpoint", "Shapes"]
    )
    if len(table) != 1 and not use_vit: # do this for now; the table function has some bugs when the whole CLIP is loaded
        logger.info(
            "Following weights matched with "
            + (f"submodule {common_prefix[:-1]}" if common_prefix else "model")
            + ":\n"
            + table_str
        )
    else:
        logger.info(mmk_list)

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in set(matched_keys.keys())]
    for k in unmatched_ckpt_keys:
        result_state_dict[k] = ckpt_state_dict[k]
    return result_state_dict


def _group_keys_by_module(keys: List[str], original_names: Dict[str, str]):
    """
    Params in the same submodule are grouped together.

    Args:
        keys: names of all parameters
        original_names: mapping from parameter name to their name in the checkpoint

    Returns:
        dict[name -> all other names in the same group]
    """

    def _submodule_name(key):
        pos = key.rfind(".")
        if pos < 0:
            return None
        prefix = key[: pos + 1]
        return prefix

    all_submodules = [_submodule_name(k) for k in keys]
    all_submodules = [x for x in all_submodules if x]
    all_submodules = sorted(all_submodules, key=len)

    ret = {}
    for prefix in all_submodules:
        group = [k for k in keys if k.startswith(prefix)]
        if len(group) <= 1:
            continue
        original_name_lcp = _longest_common_prefix_str([original_names[k] for k in group])
        if len(original_name_lcp) == 0:
            # don't group weights if original names don't share prefix
            continue

        for k in group:
            if k in ret:
                continue
            ret[k] = group
    return ret


def _longest_common_prefix(names: List[str]) -> str:
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split(".") for n in names]
    m1, m2 = min(names), max(names)
    ret = [a for a, b in zip(m1, m2) if a == b]
    ret = ".".join(ret) + "." if len(ret) else ""
    return ret


def _longest_common_prefix_str(names: List[str]) -> str:
    m1, m2 = min(names), max(names)
    lcp = [a for a, b in zip(m1, m2) if a == b]
    lcp = "".join(lcp)
    return lcp


def _group_str(names: List[str]) -> str:
    """
    Turn "common1", "common2", "common3" into "common{1,2,3}"
    """
    lcp = _longest_common_prefix_str(names)
    rest = [x[len(lcp) :] for x in names]
    rest = "{" + ",".join(rest) + "}"
    ret = lcp + rest

    # add some simplification for BN specifically
    ret = ret.replace("bn_{beta,running_mean,running_var,gamma}", "bn_*")
    ret = ret.replace("bn_beta,bn_running_mean,bn_running_var,bn_gamma", "bn_*")
    return ret
