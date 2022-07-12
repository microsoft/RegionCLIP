# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from PIL import Image
import copy
from ..backbone.fpn import build_resnet_fpn_backbone
from ..backbone.clip_backbone import build_clip_language_encoder
from detectron2.utils.comm import gather_tensors, MILCrossEntropy

__all__ = ["CLIPFastRCNN", "PretrainFastRCNN"]

@META_ARCH_REGISTRY.register()
class CLIPFastRCNN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components: 
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        language_encoder: nn.Module, 
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool

    @classmethod
    def from_config(cls, cfg):
        # create independent backbone & RPN
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": 
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
            if cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg)
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        # region proposals are ground-truth boxes
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        
        backbone = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB: # extract concept embeddings
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        roi_heads = build_roi_heads(cfg, backbone.output_shape())

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
            "language_encoder": language_encoder, 
            "roi_heads": roi_heads, 
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # localization branch: offline modules to get the region proposals
        with torch.no_grad():  
            if self.clip_crop_region_type == "GT":  # from ground-truth
                proposals = []
                for r_i, b_input in enumerate(batched_inputs): 
                    this_gt = copy.deepcopy(b_input["instances"])  # Instance
                    gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                    this_gt._fields = {'proposal_boxes': gt_boxes, 'objectness_logits': torch.ones(gt_boxes.tensor.size(0)).to(self.device)}
                    proposals.append(this_gt)                
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else: # use mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, attnpool=self.backbone.bottom_up.attnpool)
            else: # use mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        #visualize_proposals(batched_inputs, proposals, self.input_format)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        
        # localization branch: offline modules to get the region proposals
        if self.clip_crop_region_type == "GT":  # from ground-truth
            proposals = []
            for r_i, b_input in enumerate(batched_inputs): 
                this_gt = copy.deepcopy(b_input["instances"])  # Instance
                gt_boxes = this_gt._fields['gt_boxes'].to(self.device)
                this_gt._fields = {'proposal_boxes': gt_boxes} #, 'objectness_logits': None}
                proposals.append(this_gt)                
        elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
    
        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Given the proposals, crop region features from 2D image features and classify the regions
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else: # use mean pool
                results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
        else:  # regular detector setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                results, _  = self.roi_heads(images, features, proposals, None, attnpool=self.backbone.bottom_up.attnpool)
            else:
                results, _  = self.roi_heads(images, features, proposals, None)
        
        #visualize_proposals(batched_inputs, proposals, self.input_format)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPFastRCNN._postprocess(results, batched_inputs)
        else:
            return results

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

@META_ARCH_REGISTRY.register()
class PretrainFastRCNN(nn.Module):
    """
    RegionCLIP: Learning visual region representation via vision-language pretraining from image-text pairs
    1. region-token level matching: learn to match the pseudo region-text pairs, provided by teacher model
    2. image-text level matching: learn to match image-text pairs, obtained from the Internet
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
        roi_heads: nn.Module,
        teacher_backbone: nn.Module,
        teacher_roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        use_clip_c4: False,
        use_clip_attpool: False,
        offline_input_format: Optional[str] = None,
        offline_pixel_mean: Tuple[float],
        offline_pixel_std: Tuple[float],
        language_encoder: nn.Module,
        matching_temp: None,
        num_regions_per_img: int = 0,
        img_txt_level: None,
        gather_gpus: False,
        concept_emb: None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.offline_backbone = offline_backbone
        self.backbone = backbone
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        
        # image-text level pretraining
        self.img_txt_level = img_txt_level[0]
        self.only_eot = img_txt_level[1]
        if self.img_txt_level:
            self.lang_encoder = language_encoder
            for p in self.lang_encoder.parameters():  # freeze language encoder
                p.requires_grad = False
        self.matching_temp = matching_temp
        self.context_length = 77 # defined in clip_img_txt_pair_tsv class
        self.num_regions_per_img = num_regions_per_img
        self.gather_gpus = gather_gpus

        # region-token level pretraining
        if concept_emb[0]:
            self.register_buffer("concept_emb", torch.load(concept_emb[0]), False) # [#concepts, d]
            self.concept_thres = concept_emb[1]
            self.teacher_backbone = teacher_backbone
            for p in self.teacher_backbone.parameters():  # freeze visual encoder of teacher model
                p.requires_grad = False
            if concept_emb[2] is None: # teacher model uses the same concept embedding as student model
                self.register_buffer("teacher_concept_emb", torch.load(concept_emb[0]), False)
            else: # teacher model uses a seperate concept embedding
                self.register_buffer("teacher_concept_emb", torch.load(concept_emb[2]), False)
            self.teacher_roi_heads = teacher_roi_heads
        else:
            self.concept_emb = None

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN": # create isolated backbone & RPN
            # create offline cfg for the pretrained backbone & RPN
            from detectron2.config import get_cfg
            offline_cfg = get_cfg()
            offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
            if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
                offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
                offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
                offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
                offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
            if cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS:
                offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS 
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH
            
            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg) # build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())
            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE in ["GRID", "RANDOM"]:
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        
        # visual encoder and roi_heads of student model
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        # language encoder of student model
        language_encoder = build_clip_language_encoder(cfg)
        # visual encoder of teacher model
        teacher_cfg = copy.deepcopy(cfg)
        teacher_cfg.defrost()
        teacher_cfg.MODEL.RESNETS.DEPTH = teacher_cfg.MODEL.CLIP.TEACHER_RESNETS_DEPTH
        teacher_backbone = build_backbone(teacher_cfg)
        teacher_cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = teacher_cfg.MODEL.CLIP.TEACHER_POOLER_RESOLUTION
        teacher_roi_heads = build_roi_heads(teacher_cfg, teacher_backbone.output_shape())

        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
            "roi_heads": roi_heads, 
            "teacher_backbone": teacher_backbone,
            "teacher_roi_heads": teacher_roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME == 'PretrainRes5ROIHeads',
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            "language_encoder": language_encoder,
            "matching_temp": cfg.MODEL.CLIP.CLSS_TEMP,
            "num_regions_per_img": cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS,
            "img_txt_level": (cfg.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL, cfg.MODEL.CLIP.PRETRAIN_ONLY_EOT),
            "gather_gpus": cfg.MODEL.CLIP.GATHER_GPUS,
            "concept_emb": (cfg.MODEL.CLIP.CONCEPT_POOL_EMB, cfg.MODEL.CLIP.CONCEPT_THRES, cfg.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        gt_instances = None
        losses = {}
        
        # localization branch: offline modules to get the region proposals
        proposals = self.get_region_proposals(batched_inputs)
        global_proposals = self.create_global_proposals(batched_inputs)

        # recognition branch: get 2D feature maps using the backbone of recognition branch and extract region features
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        region_feats = self.get_region_features(images, features, proposals, gt_instances)
        global_feats = self.get_region_features(images, features, global_proposals, gt_instances)

        # image-text level matching
        if self.img_txt_level:
            self.image_text_matching(batched_inputs, proposals, region_feats, losses, global_feats=global_feats)

        # region-concept level matching
        if self.concept_emb is not None:
            self.region_concept_matching(images, proposals, gt_instances, region_feats, losses)

        return losses

    def region_concept_matching(self, images, proposals, gt_instances, region_feats, losses, use_distill=True, use_contrastive=True):
        # get psuedo concept labels from teacher model
        concept_scores, target_inds, keep_regions, target_embs, label_mtx \
            = self.get_psuedo_concept_labels(images, proposals, gt_instances)

        # prepare region features for the kept regions
        keep_region_feats = region_feats[keep_regions]
        keep_region_feats = keep_region_feats / keep_region_feats.norm(dim=-1, keepdim=True)

        if use_distill:
            # distillation learning: learns from the predictions of teacher model
            concept_emb = self.concept_emb / self.concept_emb.norm(dim=-1, keepdim=True)
            cls_scores = keep_region_feats @ concept_emb.t()  # [#kept_regions, #concepts]
            cls_scores_temp = cls_scores / self.matching_temp
            
            # calculate loss
            cls_loss = F.kl_div(F.softmax(cls_scores_temp, dim=1).log(), concept_scores, reduction='batchmean')  # input is log-probabilities, target is probabilities
            losses.update({"loss_region_distill": cls_loss}) #  * 0.8})

        if use_contrastive:
            # contrastive learning: matching student visual features with target concept embs
            target_embs = target_embs / target_embs.norm(dim=-1, keepdim=True)
            match_scores = keep_region_feats @ target_embs.t()  # [#kept_regions, #kept_regions]
            match_scores_temp = match_scores / self.matching_temp

            # calculate loss given matching scores and label matrix
            contrastive_loss = MILCrossEntropy()(match_scores_temp, label_mtx, weights=None, avg_positives=False)
            losses.update({"loss_concept_contrastive": contrastive_loss})

    def image_text_matching(self, batched_inputs, proposals, region_feats, losses, global_feats):
        # encode text
        num_cap = int(batched_inputs[0][1].size(0) / self.context_length)
        if num_cap == 1:  # one caption per image
            text = [x[1].view(1,-1).to(self.device) for x in batched_inputs]
        else: # multiple caption pers image, then randomly pick one
            rand_ind = [randint(0, num_cap-1) for _ in range(len(batched_inputs))]
            text = [x[1].view(-1,self.context_length)[rand_ind[i]:rand_ind[i]+1].to(self.device) for i, x in enumerate(batched_inputs)]
        text = torch.cat(text, dim=0)
        text_embs = self.lang_encoder.encode_text(text, only_eot=self.only_eot)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]

        # prepare region features and text embeddings
        region_feats = global_feats
        region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        region_feats_full, min_bs = gather_tensors(region_feats) if self.gather_gpus else (region_feats, None)  #  gather across GPUs
        text_embs_full, min_bs = gather_tensors(text_embs) if self.gather_gpus else (text_embs, None)  #  gather across GPUs

        # matching visual features with text embs
        match_scores = region_feats_full @ text_embs_full.view(-1, text_embs_full.size(-1)).t()  # [#regions, img_batch * n_ctx]
        img_b = int(region_feats_full.size(0))
        pooled_score = match_scores

        pooled_score = pooled_score / self.matching_temp
        contrast_target = torch.arange(img_b).to(self.device)
        row_loss = F.cross_entropy(pooled_score, contrast_target)
        col_loss = F.cross_entropy(pooled_score.t(), contrast_target)
        losses.update({"loss_img_txt_level": (row_loss + col_loss) / 2.0}) 

    def get_psuedo_concept_labels(self, images, proposals, gt_instances, s_temp=0.01):
        """ Input images and region proposals, return matching results from teacher model
        """
        with torch.no_grad():
            # extract visual features from teacher model
            features = self.teacher_backbone(images.tensor)
            teacher_region_feats = self.teacher_roi_heads(images, features, proposals, gt_instances, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)
            
            # match teacher visual features with teacher concept embs to create pseudo labels
            teacher_region_feats = teacher_region_feats / teacher_region_feats.norm(dim=-1, keepdim=True)
            teacher_concept_emb = self.teacher_concept_emb / self.teacher_concept_emb.norm(dim=-1, keepdim=True)
            concept_scores = teacher_region_feats @ teacher_concept_emb.t()  # [#regions, #concepts]
            concept_scores = F.softmax(concept_scores / s_temp, dim=1)

            max_scores, max_inds = torch.max(concept_scores, dim=1)
            keep_regions = max_scores > self.concept_thres  # only keep the regions that have high matching score with a concept
            if keep_regions.nonzero().size(0) == 0: # if all regions can't match to any concept
                print("all regions can't match to any concept!")
                keep_regions = max_scores > 0.0 
            target_inds = max_inds[keep_regions]
            target_embs = self.concept_emb[target_inds] # the target embedding of student model
            label_mtx = (target_inds.view(-1, 1) == target_inds.view(1, -1)).type_as(teacher_region_feats)
            concept_scores = concept_scores[keep_regions]
                
        return concept_scores, target_inds, keep_regions, target_embs, label_mtx

    def get_region_features(self, images, features, proposals, gt_instances):
        """ Input images and region proposals, return region features
        """
        # Given the proposals, crop region features from 2D image features
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else: # use mean pool
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        else:  # regular detector setting
            region_feats = self.roi_heads(images, features, proposals, gt_instances)
        return region_feats

    def get_region_proposals(self, batched_inputs):
        """ Given image, return object proposals
        """
        with torch.no_grad():  
            if self.clip_crop_region_type == "RANDOM":  # from random proposals
                proposals = self.create_rand_boxes(batched_inputs)         
            elif self.clip_crop_region_type == "RPN": # from the backbone & RPN of standard Mask-RCNN, trained on base classes
                if self.offline_backbone.training or self.offline_proposal_generator.training:  #  was set to True in training script
                    self.offline_backbone.eval() 
                    self.offline_proposal_generator.eval()  
                images = self.offline_preprocess_image(batched_inputs)
                features = self.offline_backbone(images.tensor)
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)     
            #visualize_proposals(batched_inputs, proposals, self.input_format, vis_pretrain=True)
        
        # randomly select proposals
        if self.training:
            rand_inds = [torch.randperm(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
            proposals = [p[rand_inds[i]] for i, p in enumerate(proposals)]
        return proposals

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x[0].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [((x * 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x[0].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [((x * 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def create_rand_boxes(self, batched_inputs, grid_length=8):
        """ create random boxes within an image, output random self.num_regions_per_img boxes
        return a list of Boxes
        """
        images = self.preprocess_image(batched_inputs)
        image_height = images.tensor.size(2)
        image_width = images.tensor.size(3)

        left_top_x = torch.tensor([i*(grid_length) for i in range(image_width // grid_length)])
        left_top_y = torch.tensor([i*(grid_length) for i in range(image_height // grid_length)])
        right_bot_x = torch.tensor([(i+1)*(grid_length) for i in range(image_width // grid_length)])
        right_bot_y = torch.tensor([(i+1)*(grid_length) for i in range(image_height // grid_length)])
        x_inds = torch.randint(0, left_top_x.size(0), (self.num_regions_per_img,))
        y_inds = torch.randint(0, left_top_y.size(0), (self.num_regions_per_img,))

        proposals = []
        for i in range(self.num_regions_per_img):
            rb_x_candidates = right_bot_x[x_inds[i]:]
            rb_x = rb_x_candidates[torch.randperm(rb_x_candidates.size(0))[0]]
            rb_y_candidates = right_bot_y[y_inds[i]:]
            rb_y = rb_y_candidates[torch.randperm(rb_y_candidates.size(0))[0]]
            this_box = torch.cat((left_top_x[x_inds[i]].view(1,1), left_top_y[y_inds[i]].view(1,1), rb_x.view(1,1), rb_y.view(1,1)),dim=-1)
            proposals.append(this_box)
        proposals = torch.cat(proposals).float().to(self.device)
        proposals = [Boxes(proposals) for i in range(len(batched_inputs))] # a list of Boxes
        return proposals

    def create_global_proposals(self, batched_inputs):
        """ create a single global box for an image, so as to extract global image features with RoIAlign on high-resolution images.
        """
        images = self.preprocess_image(batched_inputs)
        image_height = images.tensor.size(2)
        image_width = images.tensor.size(3)

        global_box = torch.tensor([0, 0, image_width, image_height]).view(1,4).float().to(self.device)
        proposals = [Boxes(global_box) for i in range(len(batched_inputs))] # a list of Boxes
        return proposals

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        pass

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(instances, batched_inputs):
            height, width = input_per_image[-1][2] # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


def visualize_proposals(batched_inputs, proposals, input_format, vis_pretrain=False):
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 top-scoring predicted
    object proposals on the original image. Users can implement different
    visualization functions for different models.

    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
    """
    from detectron2.utils.visualizer import Visualizer

    max_vis_prop = 50
    if vis_pretrain:
        for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
            img = input[0] * 255.0
            img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = prop_img
            to_save = Image.fromarray(np.array(vis_img, np.uint8))
            to_save.save("output/regions/" + str(i) + ".png")
            #break  # only visualize one image in a batch
    else:
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            #vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            f_n = input['file_name']
            to_save = Image.fromarray(np.array(vis_img, np.uint8))
            to_save.save("output/regions/" + f_n.split("/")[-1].split(".")[0] + ".png")
            #break  # only visualize one image in a batch
