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
import torchvision
from torchvision.transforms import Resize, CenterCrop
from detectron2.data.datasets.clip_prompt_utils import get_cls_names, pre_tokenize
import copy
from ..backbone.fpn import build_resnet_fpn_backbone
from ..roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.layers import ShapeSpec
from ..backbone.clip_backbone import build_clip_language_encoder
from detectron2.utils.comm import gather_tensors, MILCrossEntropy, SoftTargetCrossEntropy

__all__ = ["CLIPRCNN", "CLIPFastRCNN", "PretrainFastRCNN"]

@META_ARCH_REGISTRY.register()
class CLIPRCNN(nn.Module):
    """
    CLIP in R-CNN format. 
    It takes the image regions as inputs and classifies each image.
    It contains the following two components:
    1. Per-image feature extraction (visual encoder)
    2. Per-image prediction (text-based classifier)
    """
    @configurable
    def __init__(
        self,
        *,
        clip: Backbone,
        offline_backbone: Backbone,
        offline_proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        clip_crop_region_type: str = 'GT',
        test_score_thresh: float = 0.0001,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: float = 300,
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
        self.clip_backbone = clip
        self.offline_backbone = offline_backbone
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # Detectron2 default pixel mean and std
        self.register_buffer("detectron_pixel_mean", torch.tensor([103.530, 116.280, 123.675]).view(-1, 1, 1), False)
        self.register_buffer("detectron_pixel_std", torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1), False)

        # CLIP image loading
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else: # default setting
            self.div_pixel = False
        n_px = 224
        self.clip_resize = Resize(n_px, interpolation=Image.BICUBIC)  # shorter side becomes n_px
        self.clip_center_crop = CenterCrop(n_px)  # crop image into n_px * n_px at the center
        self.region_crop_scales = (1.0, 1.5) # (1.0, 2.0) # (1.0, 1.2) # (1.0,) #  

        # CLIP text prompt loading
        print("Working on pre_tokenize...")
        cls_names = get_cls_names(filter_novel=False, from_file='/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/concept_pool/googlecc_nouns_filtered_100.txt') # filter_novel=True; coco='all', coco='base', coco='target'; from_file: a file path for concept pool
        # from_file='/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/concept_pool/googlecc_nouns_triplet_parser_filtered_100.txt'
        print("Got {} class names: {}\n {} class names in total.".format(len(cls_names), cls_names, len(cls_names)))
        input_ids = pre_tokenize(cls_names)
        self.num_cls = input_ids.size(0)
        self.num_prompt = input_ids.size(1)
        self.input_ids_flat = input_ids.view(-1, input_ids.size(2))  # [#cls*#prompts, #context_length]
        self.clss_emb_all = None

        # CLIP crop image configs
        self.clip_crop_region_type = clip_crop_region_type
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
            offline_backbone = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))) # build_backbone(cfg)
            offline_rpn = build_proposal_generator(cfg, offline_backbone.output_shape())
            roi_heads = None # build_roi_heads(cfg, backbone.output_shape()),
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            roi_heads = None
        clip = build_backbone(cfg)
        return {
            "clip": clip,
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "roi_heads": roi_heads, 
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_crop_region_type" : cfg.MODEL.CLIP.CROP_REGION_TYPE,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
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
        # No training mode for this arch

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

        # get the label prompt, and use CLIP.encode_text() to compute text emb only once
        if self.clss_emb_all is None:  # compute only once
            num_instances = self.input_ids_flat.size(0)
            per_split = 1000
            num_splits = num_instances // per_split
            input_ids_flat = self.input_ids_flat.to(self.device)
            #self.clss_emb_all = torch.ones((1203, 512)).to(self.device)
            clss_emb_all = []
            for i in range(num_splits+1):
                if i < num_splits:
                    clss_emb_i = self.clip_backbone.encode_text(input_ids_flat[per_split*i:per_split*(i+1)]) # per_split x D
                else:
                    clss_emb_i = self.clip_backbone.encode_text(input_ids_flat[per_split*i:]) # per_split x D
                # clss_emb_i = clip_model.encode_label(torch.arange(0, 1000).view(-1, 1).long().to(device)) # per_split x D
                clss_emb_all.append(clss_emb_i)
            self.clss_emb_all = torch.cat(clss_emb_all, 0).view(self.num_cls, self.num_prompt, -1)  # [#cls, #prompts, D]
            self.clss_emb_all = self.clss_emb_all.mean(1)  # ensemble different prompts for each class
            # torch.save(self.clss_emb_all.cpu(), "/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/lvis_cls_emb/coco_17_target_cls_emb_notnorm_rn50x4.pth")
            self.clss_emb_all = F.normalize(self.clss_emb_all, p=2.0, dim=1)  # [#cls, emb_dim]
        else:
            assert self.clss_emb_all.device == self.device

        # get the region proposals, from the backbone & RPN of standard Mask-RCNN, trained on base classes
        if self.clip_crop_region_type == "GT":
            proposals = None
        elif self.clip_crop_region_type == "RPN":
            images = self.preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            if detected_instances is None:
                if self.offline_proposal_generator is not None:
                    proposals, _ = self.offline_proposal_generator(images, features, None)   

        # crop image regions, and use CLIP.encode_image() to get the visual feature
        images, bbs, num_bbs = self.preprocess_image_crop(batched_inputs, rpn_proposals=proposals)
        img_emb = self.clip_backbone.encode_image(images.tensor)
        img_emb = img_emb.view(-1, len(self.region_crop_scales), img_emb.size(1))
        img_emb = torch.sum(img_emb, dim=1)  # ensemble different scales for each region
        img_emb = F.normalize(img_emb, p=2.0, dim=1)

        # cosine similarity as logits
        all_scores = torch.mm(img_emb, self.clss_emb_all.T)
        all_scores = F.softmax(all_scores, dim=-1)
        scores, pred_cls = torch.max(all_scores, dim=-1)  # Note: [0, #cls-1] representing the categories. The value #cls represents "background".

        # convert model outputs into regular output result format
        scores_per_img = scores.split(num_bbs)
        pred_cls_per_img = pred_cls.split(num_bbs)
        all_scores_per_img = all_scores.split(num_bbs)

        # per-class NMS
        if self.clip_crop_region_type == "GT":
            image_shapes = [x['instances']._image_size for x in batched_inputs]
            bbs = [bb.to(self.device) for bb in bbs]
            pred_instances, _ = fast_rcnn_inference(bbs, all_scores_per_img, image_shapes, \
                                        self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)
            results = pred_instances

            # results = []
            # for r_i, (b_input, bb, sc, prd) in enumerate(zip(batched_inputs, bbs, scores_per_img, pred_cls_per_img)): 
            #     this_result = copy.deepcopy(b_input["instances"])  # Instance
            #     if self.clip_crop_region_type == "GT":
            #         result_boxes = this_result._fields['gt_boxes'].to(self.device)
            #     elif self.clip_crop_region_type == "RPN":  # directly use RPN boxes without per-class NMS
            #         result_boxes = bb # result_boxes = Boxes(bb)
            #     this_result._fields = {'pred_boxes': result_boxes, 'scores': sc, 'pred_classes': prd}
            #     results.append(this_result)
            
            # sanity check: GT boxes + GT classes
            # results = []
            # for b_input in batched_inputs: 
            #     this_result = copy.deepcopy(b_input["instances"])  # Instance
            #     gt_boxes = this_result._fields['gt_boxes'].to(self.device)
            #     gt_cls =  this_result._fields['gt_classes'].to(self.device)
            #     this_result._fields = {'pred_boxes': gt_boxes, 'scores': torch.ones(gt_cls.size(0)).to(self.device), 'pred_classes': gt_cls}
            #     #this_result._fields = {'pred_boxes': gt_boxes, 'scores': sc, 'pred_classes': prd}
            #     results.append(this_result)
        elif self.clip_crop_region_type == "RPN":
            image_shapes = [x.image_size for x in proposals]
            pred_instances, _ = fast_rcnn_inference(bbs, all_scores_per_img, image_shapes, \
                                        self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)
            results = pred_instances

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CLIPRCNN._postprocess(results, batched_inputs)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.detectron_pixel_mean) / self.detectron_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image_crop(self, batched_inputs: List[Dict[str, torch.Tensor]], rpn_proposals=None, max_num_rpn=1000):
        """
        Crop image regions based on GT or RPN boxes with different scales.
        Then apply CLIP tranformation: resizing / cropping the regions into square shape (224 * 224).
        Followed by the default preprocessing in Detectron2 as follows.
        Normalize, pad and batch the input images.
        """
        def clip_crop_region(image, box, scales=(1.0, 1.5)):
            """Crop image regions based on given boxes. Return different scales of region crops. (3 hrs)"""
            img_h, img_w = image.size(1), image.size(2)
            x1, y1, x2, y2 = list(box)
            assert x1 < x2 and y1 < y2 and x2 < (img_w + 1) and y2 < (img_h + 1)
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            half_w = x_center - x1
            half_h = y_center - y1
            regions = []
            for scale in scales: # get region coordinates
                r_y1 = int(max(0, (y_center - half_h * scale).item()))
                r_y2 = int(min(img_h, (y_center + half_h * scale).item()))
                r_x1 = int(max(0, (x_center - half_w * scale).item()))
                r_x2 = int(min(img_w, (x_center + half_w * scale).item()))
                # sanity check
                if r_y2 - r_y1 <= 1:
                    r_y2 = int(min(img_h, r_y2 + 2))
                if r_y2 - r_y1 <= 1:
                    r_y1 = int(max(0, r_y1 - 2))
                if r_x2 - r_x1 <= 1:
                    r_x2 = int(min(img_w, r_x2 + 2))
                if r_x2 - r_x1 <= 1:
                    r_x1 = int(max(0, r_x1 - 2))
                regions.append(image[:, r_y1:r_y2, r_x1:r_x2])
            return regions
        
        def clip_square_crop(image, box, scales=(1.0,)):
            """Crop image regions based on given boxes. Ensure square region as much as possible. (1.75 hrs)"""
            img_h, img_w = image.size(1), image.size(2)
            x1, y1, x2, y2 = list(box)
            assert x1 < x2 and y1 < y2 and x2 < (img_w + 1) and y2 < (img_h + 1)
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            half_w = x_center - x1
            half_h = y_center - y1
            square_side = max(half_w, half_h)
            half_w = square_side
            half_h = square_side
            regions = []
            for scale in scales: # get region coordinates
                if square_side * square_side < 2500:  # crop larger context area for tiny objects
                    scale = 1.5 if scale == 1.0 else 4.0
                # elif square_side * square_side > 90000:  # crop exact area for large objects
                #     scale = 1.0 if scale == 1.0 else 1.1
                r_y1 = int(max(0, (y_center - half_h * scale).item()))
                r_y2 = int(min(img_h, (y_center + half_h * scale).item()))
                r_x1 = int(max(0, (x_center - half_w * scale).item()))
                r_x2 = int(min(img_w, (x_center + half_w * scale).item()))
                # sanity check
                if r_y2 - r_y1 <= 1:
                    r_y2 = int(min(img_h, r_y2 + 2))
                if r_y2 - r_y1 <= 1:
                    r_y1 = int(max(0, r_y1 - 2))
                if r_x2 - r_x1 <= 1:
                    r_x2 = int(min(img_w, r_x2 + 2))
                if r_x2 - r_x1 <= 1:
                    r_x1 = int(max(0, r_x1 - 2))
                #regions.append(image[:, r_y1:r_y2, r_x1:r_x2])
                # if the cropped image isn't square (due to image boundaries), pad the cropped region
                crop_image = image[:, r_y1:r_y2, r_x1:r_x2]
                r_h, r_w = crop_image.size(1), crop_image.size(2)
                pad_image = torch.zeros((3, int(2 * half_h.item() * scale) + 4 , int(2 * half_w.item() * scale) + 4)) #.fill_(torch.mean(crop_image.float()))
                p_h, p_w = pad_image.size(1), pad_image.size(2)
                pad_image[:, int(((p_h - r_h) / 2)):int(((p_h - r_h) / 2 + r_h)), int(((p_w - r_w) / 2)):int(((p_w - r_w) / 2 + r_w))] = crop_image
                regions.append(pad_image.type(torch.uint8))
            return regions
        
        def vis_crop(f_n, images):
            """visualize the crop regions to diagnose the accuracy."""
            if f_n not in ['datasets/coco/train2017/000000008691.jpg']:
                for p_i, pad_image in enumerate(images):
                    to_save = pad_image.permute(1, 2, 0).numpy()
                    to_save = Image.fromarray(np.array(to_save, np.uint8))
                    to_save.save("output/regions/" + f_n.split("/")[-1].split(".")[0] + "-{}.png".format(p_i))
                    pass

        # crop image region
        images = []
        bbs = []
        num_bbs = []
        for img_i, b_input in enumerate(batched_inputs): 
            this_img = b_input["image"]
            if self.clip_crop_region_type == "GT":
                this_boxes = b_input["instances"]._fields['gt_boxes'].tensor  # variant #bbox (eg, max 759), might lead to OOM
            elif self.clip_crop_region_type == "RPN":
                this_boxes = rpn_proposals[img_i]._fields['proposal_boxes'].tensor[:max_num_rpn]

            bbs.append(this_boxes)
            num_bbs.append(this_boxes.size(0))
            for this_box in this_boxes:
                #images.extend(clip_crop_region(this_img, this_box, self.region_crop_scales))
                images.extend(clip_square_crop(this_img, this_box, self.region_crop_scales))
        #vis_crop(batched_inputs[0]['file_name'], images)
        images = [self.clip_resize(x) for x in images]
        images = [self.clip_center_crop(x) for x in images]
        images = [x.to(self.device) for x in images]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.clip_backbone.size_divisibility) # batch images into single tensor by padding to same size
        return images, bbs, num_bbs

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

    def inference_on_cifar(self, pseudo_input):
        """ Evaluate recoginition accuracy on CIFAR-10 for sanity check """
        # get the label prompt, and use CLIP.encode_text() to compute text emb only once
        cifar_cls_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        input_ids = pre_tokenize(cifar_cls_names)
        num_cls = input_ids.size(0)
        input_ids_flat = input_ids.view(-1, input_ids.size(2))
        input_ids_flat = input_ids_flat.to(self.device)
        
        clss_emb_all = self.clip_backbone.encode_text(input_ids_flat)
        clss_emb_all = clss_emb_all.view(num_cls, self.num_prompt, -1)
        clss_emb_all = clss_emb_all.mean(1)
        clss_emb_all = F.normalize(clss_emb_all, p=2.0, dim=1)  # [#cls, emb_dim]
        
        # dataset loads images and labels
        testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                            download=False, transform=None)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
        #                                         shuffle=False, num_workers=0)
        
        # inference on each image and calculate accuracy
        correct = 0
        wrong = 0
        for idx, inputs in enumerate(testset):
            if idx % 1000 == 0:
                print(idx)
            # preprocess images
            raw_image, label = inputs
            image = np.array(raw_image)  # [h, w, 3]
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1) # [3, h, w]
            images = [image]
            images = [self.clip_resize(x) for x in images]
            images = [self.clip_center_crop(x) for x in images]
            images = [x.to(self.device) for x in images]
            if self.div_pixel:
                images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
            else:
                images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            
            # get image embedding
            img_emb = self.clip_backbone.encode_image(images[0].unsqueeze(0))
            img_emb = img_emb.view(-1, 1, img_emb.size(1))
            img_emb = torch.sum(img_emb, dim=1)  # ensemble different scales for each region
            img_emb = F.normalize(img_emb, p=2.0, dim=1)

            # cosine similarity as logits
            all_scores = torch.mm(img_emb, clss_emb_all.T)
            scores, pred_cls = torch.max(all_scores, dim=1)  # Note: [0, #cls-1] representing the categories. The value #cls represents "background".
            pred_cls = pred_cls.item()
            if pred_cls == label:
                correct += 1
            else:
                wrong += 1
        
        print("\n\nGot correct {} and wrong {}. Accuracy is {} / {} = {}\n\n".format(correct,wrong,correct,correct+wrong,correct/(correct+wrong)))
        return

@META_ARCH_REGISTRY.register()
class CLIPFastRCNN(nn.Module):
    """
    CLIP in Fast R-CNN format, where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components: 
    1. Localization modules: pretrained backbone+RPN or equivalent modules and is able to output object proposals
    2. Recognition branch: initialized by CLIP and is able to recognize zero-shot regions
    """
    @configurable
    def __init__(
        self,
        *,
        offline_backbone: Backbone,
        backbone: Backbone,
        offline_proposal_generator: nn.Module,
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
        self.offline_proposal_generator = offline_proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else: # default setting
            self.div_pixel = False

        # input format, pixel mean and std for offline modules
        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else: # default setting
                self.offline_div_pixel = False
        
        self.clip_crop_region_type = clip_crop_region_type
        self.use_clip_c4 = use_clip_c4 # if True, use C4 mode where roi_head uses the last resnet layer from backbone 
        self.use_clip_attpool = use_clip_attpool # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool


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
            if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9

            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg) # build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())
            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT":
            offline_backbone = None
            offline_rpn = None
            offline_cfg = None
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        return {
            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "backbone": backbone,
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
            else: # use default mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        else:  # default setting
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, attnpool=self.backbone.bottom_up.attnpool)
            else: # use default mean pool
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
            else: # use default mean pool
                results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
        else:  # default setting
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
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'): # the input image follows the main config format ('RGB' or 'BGR')
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
    Open-vocabulary region representation via vision-language pretraining from image-text pairs
    1. image-text level matching: weakly supervised grounding task with contrastive learning based on region-token representation
    2. region-token level matching: use pseudo text to train model, provided by teacher model
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
        grid_regions: False,
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

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else: # default setting
            self.div_pixel = False

        # input format, pixel mean and std for offline modules
        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else: # default setting
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
        if matching_temp > 0.0: # fixed temp
            self.matching_temp = matching_temp
        else: # leanable temp
            self.matching_temp = nn.Parameter(torch.ones([]) * 4.6052) # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.context_length = 77 # defined in clip_img_txt_pair_tsv class
        self.num_regions_per_img = num_regions_per_img
        self.gather_gpus = gather_gpus
        self.grid_regions = grid_regions

        # region-token level pretraining
        if concept_emb[0]:
            self.register_buffer("concept_emb", torch.load(concept_emb[0]), False) # [#concepts, 1024]
            self.concept_thres = concept_emb[1]
            self.teacher_backbone = teacher_backbone # None
            # when resume, create teacher model in advance to load ckpt
            # self.teacher_backbone = copy.deepcopy(self.backbone)
            # # # oai_clip = torch.load("/mnt/output_storage/trained_models/oai_clip_weights/RN50_OAI_CLIP.pth") #("/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/oai_clip_weights/RN50_OAI_CLIP.pth")
            # # # oai_clip_visual = {}
            # # # for key in oai_clip['model']:
            # # #     if 'visual' in key and 'num_batches_tracked' not in key:
            # # #         oai_clip_visual[key.replace('visual.','')] = oai_clip['model'][key]
            # # # self.teacher_backbone.load_state_dict(oai_clip_visual)
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
                offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
            # offline_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
            # print("\n\n Set offline RPN.NMS_THRESH to {} and ROI_HEADS.NMS_THRESH_TEST to {}.\n\n".format(offline_cfg.MODEL.RPN.NMS_THRESH, offline_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST))
            # create offline backbone and RPN
            offline_backbone = build_backbone(offline_cfg) # build_resnet_fpn_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
            offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())
            # convert to evaluation mode
            for p in offline_backbone.parameters(): p.requires_grad = False
            for p in offline_rpn.parameters(): p.requires_grad = False
            offline_backbone.eval()
            offline_rpn.eval()
        elif cfg.MODEL.CLIP.CROP_REGION_TYPE in ["GLOBAL", "GRID", "RANDOM"]:
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
            "grid_regions": cfg.MODEL.CLIP.GRID_REGIONS,
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
        if self.concept_emb is not None and self.teacher_backbone is None: # create a teacher model from an initialized student model; if resume, simply comment out this section
            self.teacher_backbone = copy.deepcopy(self.backbone)
            for p in self.teacher_backbone.parameters():  # freeze visual encoder of teacher model
                p.requires_grad = False
        gt_instances = None
        losses = {}
        
        # localization branch: offline modules to get the region proposals
        proposals = self.get_region_proposals(batched_inputs)
        global_proposals = self.create_global_proposals(batched_inputs)
        # for prop, g_prop in zip(proposals, global_proposals):  # append global proposal into each image
        #     prop.proposal_boxes.tensor = torch.cat((prop.proposal_boxes.tensor, g_prop.tensor), dim=0)

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        region_feats = self.get_region_features(images, features, proposals, gt_instances)
        global_feats = self.get_region_features(images, features, global_proposals, gt_instances)

        # image-text level matching
        if self.img_txt_level:
            self.image_text_matching(batched_inputs, proposals, region_feats, losses, global_feats=global_feats, only_global=True)

        # region-phrase level matching
        if len(batched_inputs[0]) > 6:  # controlled by dataset loading
            phrase_text_embs = self.encode_phrase_text(batched_inputs)
        else:
            phrase_text_embs = None

        # region-concept level matching
        if self.concept_emb is not None:
            self.region_concept_matching(images, proposals, gt_instances, region_feats, losses, phrase_embs=phrase_text_embs)

        return losses

    def encode_phrase_text(self, batched_inputs):
        text = [x[6].view(-1,self.context_length).to(self.device) for i, x in enumerate(batched_inputs)]
        text = torch.cat(text, dim=0)
        text_embs = self.lang_encoder.encode_text(text, only_eot=True)  # [#phrases, transformer.width]
        return text_embs

    def region_concept_matching(self, images, proposals, gt_instances, region_feats, losses, phrase_embs=None):
        use_distill = True
        use_contrastive = True
        # get psuedo concept labels from teacher model
        concept_scores, target_inds, keep_regions, target_embs, label_mtx, phrase_label_mtx, phrase_target_regions \
            = self.get_psuedo_concept_labels(images, proposals, gt_instances, phrase_embs=phrase_embs)

        # prepare region features for the kept regions
        keep_region_feats = region_feats[keep_regions]
        keep_region_feats = keep_region_feats / keep_region_feats.norm(dim=-1, keepdim=True)

        if use_distill:
            # distillation learning: learns from the predictions of teacher model
            concept_emb = self.concept_emb / self.concept_emb.norm(dim=-1, keepdim=True)
            cls_scores = keep_region_feats @ concept_emb.t()  # [#kept_regions, #concepts]
            if isinstance(self.matching_temp, float): # Typical good values are 100.0 for euclidean, 10.0 for dot, 0.01 for cosine
                cls_scores_temp = cls_scores / self.matching_temp
            else:
                cls_scores_temp = cls_scores * self.matching_temp.exp() 
            
            # loss weights
            #rpn_weights = torch.cat([torch.sigmoid(p.objectness_logits) for p in proposals])[keep_regions]
            #focal_weights = self.focal_scaling(cls_scores_temp, target_inds)
            
            # calculate loss
            cls_loss = F.kl_div(F.softmax(cls_scores_temp, dim=1).log(), concept_scores, reduction='batchmean')  # input is log-probabilities, target is probabilities
            #cls_loss = SoftTargetCrossEntropy()(cls_scores_temp, concept_scores)
            #cls_loss = F.cross_entropy(cls_scores_temp, target_inds)
            #cls_loss = (F.cross_entropy(cls_scores_temp, target_inds, reduction="none") * focal_weights).mean()
            losses.update({"loss_region_distill": cls_loss}) #  * 0.8})

        if use_contrastive:
            # contrastive learning: matching student visual features with target teacher concept embs
            target_embs = target_embs / target_embs.norm(dim=-1, keepdim=True)
            match_scores = keep_region_feats @ target_embs.t()  # [#kept_regions, #kept_regions]
            if isinstance(self.matching_temp, float): # Typical good values are 100.0 for euclidean, 10.0 for dot, 0.01 for cosine
                match_scores_temp = match_scores / self.matching_temp
            else:
                match_scores_temp = match_scores * self.matching_temp.exp() 
            
            # loss weights
            #rpn_weights = torch.cat([torch.sigmoid(p.objectness_logits) for p in proposals])[keep_regions]
            #focal_weights = (1 - torch.sigmoid(torch.diag(match_scores_temp))) ** 0.8 # 1.0 # 2.0 # 

            # calculate loss given matching scores and label matrix
            contrastive_loss = MILCrossEntropy()(match_scores_temp, label_mtx, weights=None, avg_positives=False) # SoftTargetCrossEntropy()(match_scores_temp, label_mtx)
            #contrastive_loss = (MILCrossEntropy()(match_scores, label_mtx) + MILCrossEntropy()(match_scores.t(), label_mtx)) / 2.0
            losses.update({"loss_concept_contrastive": contrastive_loss})

        if phrase_embs is not None:
            phrase_embs = phrase_embs / phrase_embs.norm(dim=-1, keepdim=True)
            phrase_scores = phrase_embs @ phrase_target_regions.t()
            if isinstance(self.matching_temp, float): # Typical good values are 100.0 for euclidean, 10.0 for dot, 0.01 for cosine
                phrase_scores_temp = phrase_scores / self.matching_temp
            else:
                phrase_scores_temp = phrase_scores * self.matching_temp.exp() 
            contrastive_loss = MILCrossEntropy()(phrase_scores_temp, phrase_label_mtx, weights=None, avg_positives=False)
            #contrastive_loss = SoftTargetCrossEntropy()(phrase_scores_temp, phrase_label_mtx)
            losses.update({"loss_phrase_contrastive": contrastive_loss})

    def image_text_matching(self, batched_inputs, proposals, region_feats, losses, global_feats=None, only_global=False):
        # encode text
        num_cap = int(batched_inputs[0][1].size(0) / self.context_length)
        if num_cap == 1:  # one caption per image
            text = [x[1].view(1,-1).to(self.device) for x in batched_inputs]
        else: # multiple caption pers image, then randomly pick one
            rand_ind = [randint(0, num_cap-1) for _ in range(len(batched_inputs))]
            text = [x[1].view(-1,self.context_length)[rand_ind[i]:rand_ind[i]+1].to(self.device) for i, x in enumerate(batched_inputs)]
        text = torch.cat(text, dim=0)
        text_embs = self.lang_encoder.encode_text(text, only_eot=self.only_eot)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]
        eot_pos = text.argmax(dim=-1) 

        # prepare region features and text embeddings
        if isinstance(proposals[0], Boxes):
            num_bbs = [len(prop) for prop in proposals]
        else: 
            num_bbs = [len(prop.proposal_boxes) for prop in proposals]
        if global_feats is not None and only_global:  # only global feature
            assert self.only_eot
            region_feats = global_feats
            region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            num_bbs = [1 for _ in num_bbs]
        elif global_feats is not None and not only_global:  # combine both global and region features
            assert self.only_eot
            keep_num = 20
            region_feats = region_feats.split(num_bbs)
            region_feats = [torch.mean(rg_f, dim=0, keepdim=True) for rg_f in region_feats]
            region_g_feats = [torch.cat((r_f[:keep_num], global_feats[i:i+1]), dim=0) for i, r_f in enumerate(region_feats)]
            region_g_feats = [torch.mean(rg_f, dim=0, keepdim=True) for rg_f in region_g_feats]
            region_g_feats = [rg_f / rg_f.norm(dim=-1, keepdim=True) for rg_f in region_g_feats]
            region_feats = torch.cat(region_g_feats)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            num_bbs = [1 for _ in num_bbs]
        else: # only region features
            num_bbs = torch.tensor(num_bbs).long().to(self.device)

        region_feats_full, min_bs = gather_tensors(region_feats) if self.gather_gpus else (region_feats, None)  #  gather across GPUs
        text_embs_full, min_bs = gather_tensors(text_embs) if self.gather_gpus else (text_embs, None)  #  gather across GPUs

        # matching visual features with text embs
        match_scores = region_feats_full @ text_embs_full.view(-1, text_embs_full.size(-1)).t()  # [#regions, img_batch * n_ctx]
        if global_feats is not None: # only global feature or combine both global and region features
            img_b = int(region_feats_full.size(0))
            pooled_score = match_scores
        else: # only region features
            eot_pos_full, min_bs = gather_tensors(eot_pos) if self.gather_gpus else (eot_pos, None)  #  gather across GPUs
            num_bbs_full, min_bs = gather_tensors(num_bbs) if self.gather_gpus else (num_bbs, None)  #  gather across GPUs
            pooled_score = []
            token_b = self.context_length
            # region_b = self.num_regions_per_img if global_feats is None else 1
            # img_b = int(region_feats_full.size(0) / region_b)
            img_b = num_bbs_full.size(0)
            rb_start = 0  # the starting index of regions
            for i in range(img_b): # for each image
                region_b = num_bbs_full[i].item()
                for j in range(img_b):  # for each text
                    if self.only_eot: # sentence level embs
                        # max pool over regions
                        this_s = torch.max(match_scores[rb_start:(rb_start+region_b), j:(j+1)], dim=0)[0]
                    else: # token level embs
                        # 3. softmax over regions as soft attention, then multiply attention with original logits, finally sum over matrix and divided by #tokens
                        # this_matrix = match_scores[rb_start:(rb_start+region_b), j*token_b:(j*token_b+eot_pos_full[j]+1)]
                        # this_att = F.softmax(this_matrix, dim=0)
                        # this_s = torch.sum(this_matrix * this_att) / (eot_pos_full[j]+1)
                        # 2. max pool over regions, and then avg over text tokens
                        # this_s = torch.sum(torch.max(match_scores[rb_start:(rb_start+region_b), j*token_b:(j*token_b+eot_pos_full[j]+1)], dim=0)[0]) / (eot_pos_full[j]+1)
                        # 1. max pool over regions, and then sum over text tokens
                        this_s = torch.sum(torch.max(match_scores[rb_start:(rb_start+region_b), j*token_b:(j*token_b+eot_pos_full[j]+1)], dim=0)[0])
                    pooled_score.append(this_s.view(1,1))
                rb_start += region_b
            assert rb_start == match_scores.size(0)
            pooled_score = torch.cat(pooled_score).view(img_b, img_b)  # diagnal elements are positive pairs and the others are negative pairs

        if isinstance(self.matching_temp,float):  # Typical good values are 100.0 for euclidean, 10.0 for dot, 0.01 for cosine
            pooled_score = pooled_score / self.matching_temp
        else:
            pooled_score = pooled_score * self.matching_temp.exp()
        contrast_target = torch.arange(img_b).to(self.device)
        row_loss = F.cross_entropy(pooled_score, contrast_target)
        col_loss = F.cross_entropy(pooled_score.t(), contrast_target)
        losses.update({"loss_img_txt_level": (row_loss + col_loss) / 2.0}) # losses.update({"loss_img_txt_level": (row_loss + col_loss) / 4.0}) # 

    def focal_scaling(self, logits, targets, gamma=1.0):
        p = F.softmax(logits, dim=1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        weights = (1 - p_t) ** gamma
        return weights

    def get_psuedo_concept_labels(self, images, proposals, gt_instances, s_temp=0.01, norm=True, phrase_embs=None):
        """ Input images and region proposals, return matching results from teacher model
        """
        with torch.no_grad():
            # extract visual features from teacher model
            features = self.teacher_backbone(images.tensor)
            teacher_region_feats = self.teacher_roi_heads(images, features, proposals, gt_instances, res5=self.teacher_backbone.layer4, attnpool=self.teacher_backbone.attnpool)
            # match teacher visual features with teacher concept embs to create pseudo labels
            if norm:
                teacher_region_feats = teacher_region_feats / teacher_region_feats.norm(dim=-1, keepdim=True)
                teacher_concept_emb = self.teacher_concept_emb / self.teacher_concept_emb.norm(dim=-1, keepdim=True)
            else:
                teacher_concept_emb = self.teacher_concept_emb
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
            # matching kept regions with phrase-text to create labels
            if phrase_embs is None:
                phrase_label_mtx = None
                phrase_target_regions = None
            else:
                if norm:
                    phrase_embs = phrase_embs / phrase_embs.norm(dim=-1, keepdim=True)
                teacher_kept_feats = teacher_region_feats[keep_regions]
                phrase_scores = phrase_embs @ teacher_kept_feats.t()  # [#phrases, #keep regions]
                phrase_scores = F.softmax(phrase_scores / s_temp, dim=1)
                _, max_region_inds = torch.max(phrase_scores, dim=1)
                phrase_label_mtx = (max_region_inds.view(-1, 1) == max_region_inds.view(1, -1)).type_as(teacher_region_feats)
                phrase_target_regions = teacher_kept_feats[max_region_inds]
                
        return concept_scores, target_inds, keep_regions, target_embs, label_mtx, phrase_label_mtx, phrase_target_regions

    def get_region_features(self, images, features, proposals, gt_instances):
        """ Input images and region proposals, return region features
        """
        # Given the proposals, crop region features from 2D image features
        if self.use_clip_c4: # use C4 + resnet weights from CLIP
            if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else: # use default mean pool
                region_feats = self.roi_heads(images, features, proposals, gt_instances, res5=self.backbone.layer4)
        else:  # default setting
            region_feats = self.roi_heads(images, features, proposals, gt_instances)
        return region_feats

    def get_region_proposals(self, batched_inputs):
        """ Given image, return object proposals
        """
        if self.grid_regions:  # use grid boxes
            proposals = self.create_grid_boxes(batched_inputs)
        else:  # use object proposals
            with torch.no_grad():  
                if self.clip_crop_region_type == "GLOBAL":  # from a global box per image
                    proposals = self.create_global_proposals(batched_inputs) 
                elif self.clip_crop_region_type == "GRID":  # from grid proposals
                    proposals = self.create_grid_boxes(batched_inputs)
                elif self.clip_crop_region_type == "RANDOM":  # from random proposals
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
        # randomly select proposals to avoid overfitting
        if self.training:
            #rand_inds = [torch.arange(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
            rand_inds = [torch.randperm(len(p))[:self.num_regions_per_img].to(self.device) for p in proposals]
            proposals = [p[rand_inds[i]] for i, p in enumerate(proposals)]
        return proposals

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        NOTE: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x[0].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'): # the input image follows the main config format ('RGB' or 'BGR')
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [((x * 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        NOTE: the image tsv in pretraining are already normalized pixel values and thus opposite to Detectron2 default input.
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
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

    def create_grid_boxes(self, batched_inputs, grid_length=32):
        """ create (image_height/32) * (image_width/32) pseudo grid boxes, and randomly sample self.num_regions_per_img boxes
        return a list of Boxes
        """
        images = self.preprocess_image(batched_inputs)
        image_height = images.tensor.size(2)
        image_width = images.tensor.size(3)

        left_top_x = torch.tensor([i*(grid_length) for i in range(image_width // grid_length)])
        left_top_y = torch.tensor([i*(grid_length) for i in range(image_height // grid_length)])
        right_bot_x = torch.tensor([(i+1)*(grid_length) for i in range(image_width // grid_length)])
        right_bot_y = torch.tensor([(i+1)*(grid_length) for i in range(image_height // grid_length)])
        left_top_x, left_top_y = torch.meshgrid(left_top_x, left_top_y)
        right_bot_x, right_bot_y = torch.meshgrid(right_bot_x, right_bot_y)
        grid_boxes = torch.cat((left_top_x.flatten().view(-1,1), left_top_y.flatten().view(-1,1),\
                                right_bot_x.flatten().view(-1,1), right_bot_y.flatten().view(-1,1),), dim=1)
        sample_ind = torch.randperm(grid_boxes.size(0))[:self.num_regions_per_img]
        grid_boxes = grid_boxes[sample_ind]
        grid_boxes = grid_boxes.float().to(self.device)
        proposals = [Boxes(grid_boxes) for i in range(len(batched_inputs))] # a list of Boxes
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
        """
        Grounding inference: map region features with sentence tokens
        return: matching scores between region features and tokenized texts, region boxes in raw image resolution, image id & raw string texts & tokenized texts
        """
        assert len(batched_inputs) == 1 # only one instance per image during inference
        gt_instances = None
        losses = {}
        
        # localization branch: offline modules to get the region proposals
        proposals = self.get_region_proposals(batched_inputs)

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        region_feats = self.get_region_features(images, features, proposals, gt_instances)

        # encode text
        num_cap = int(batched_inputs[0][1].size(0) / self.context_length)
        text = batched_inputs[0][1].view(num_cap, -1).to(self.device)  # [num_cap, context_length]
        text_embs = self.lang_encoder.encode_text(text, only_eot=False)  # [img_batch, n_ctx, transformer.width] or [img_batch, transformer.width]

        # matching visual features with text embs
        region_feats = region_feats / region_feats.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        match_scores = region_feats @ text_embs.view(-1, text_embs.size(-1)).t()  # [#regions, img_batch * n_ctx]
        # visualize_proposals(batched_inputs, proposals, self.input_format, vis_pretrain=True)

        # multiply RPN logits
        rpn_scores = [p.get('objectness_logits') for p in proposals][0]
        match_scores = (match_scores * rpn_scores[:, None]) ** 0.5
        
        # scale the object proposals back to raw image resolution
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            processed_results = PretrainFastRCNN._postprocess(proposals, batched_inputs)
            return match_scores, processed_results

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
