from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.meta_arch import CLIPFastRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import BasicStem, BottleneckBlock, ResNet, FPN, LastLevelMaxPool, ModifiedResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    Res5ROIHeads,
    CLIPRes5ROIHeads,
)

model = L(CLIPFastRCNN)(
    offline_backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ), 
        in_features=["res2", "res3", "res4", "res5"], 
        out_channels=256, 
        # norm="FrozenBN",
        top_block=LastLevelMaxPool(), 
        fuse_type="sum", 
    ),
    offline_proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(
            in_channels=256, 
            num_anchors=3,
            box_dim=4, 
            # conv_dims=[-1,-1], 
        ),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(12000, 6000),
        post_nms_topk=(2000, 1000),
        nms_thresh=0.7, 
        min_box_size=0,
        loss_weight={
            "loss_rpn_cls": 1.0, 
            "loss_rpn_loc": 1.0,
        },
        anchor_boundary_thresh=-1,
        box_reg_loss_type="smooth_l1",
        smooth_l1_beta=0.0,
    ),
    backbone=L(ModifiedResNet)(
        layers=[3, 4, 6, 3],
        output_dim=1024,
        heads=32,
        input_resolution=224, 
        width=64,
        out_features=["res4"],
        freeze_at=0,
        depth=50,
        pool_vec=False, 
        create_att_pool=True, 
        # norm_type="SyncBN", 
    ), 
    roi_heads=L(CLIPRes5ROIHeads)(
        num_classes=48,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        in_features=["res4"],
        pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 16,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        res5=None,
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=L(ShapeSpec)(channels=256*8, height=1, width=1),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes=48,
            clip_cls_emb=(
                True, 
                "/mnt/output_storage/trained_models/lvis_cls_emb/coco_48_base_cls_emb_notnorm.pth", 
                # "/home/jwyang/azureblobs/vyiwuzhong/trained_models/lvis_cls_emb/coco_48_base_cls_emb_notnorm.pth", 
                "CLIPRes5ROIHeads"
            ), 
            no_box_delta=False, 
            bg_cls_loss_weight=0.2, # 0.2 for COCO, 0.8 for LVIS 
            multiply_rpn_score=False, # False for COCO and True for LVIS
            openset_test=(
                None, 
                "/mnt/output_storage/trained_models/lvis_cls_emb/coco_65_cls_emb_notnorm.pth", 
                # "/home/jwyang/azureblobs/vyiwuzhong/trained_models/lvis_cls_emb/coco_65_cls_emb_notnorm.pth", 
                0.01, 
                False, 
                0.5
            ), 
            cls_agnostic_bbox_reg=True, 
        ),
        # mask_head=L(MaskRCNNConvUpsampleHead)(
        #     input_shape=L(ShapeSpec)(
        #         channels="${...res5.out_channels}",
        #         width="${...pooler.output_size}",
        #         height="${...pooler.output_size}",
        #     ),
        #     num_classes="${..num_classes}",
        #     conv_dims=[256],
        # ),
    ),
    input_format="RGB", 
    vis_period=0, 
    pixel_mean=[0.48145466, 0.4578275, 0.40821073],
    pixel_std=[0.26862954, 0.26130258, 0.27577711],
    clip_crop_region_type="RPN",
    use_clip_c4=True,
    use_clip_attpool=True,
    offline_input_format="BGR", 
    offline_pixel_mean=[103.530, 116.280, 123.675],
    offline_pixel_std=[1.0, 1.0, 1.0],
)