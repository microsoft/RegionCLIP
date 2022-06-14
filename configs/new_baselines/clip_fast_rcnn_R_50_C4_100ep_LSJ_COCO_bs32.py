import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from ..common.data.coco import dataloader
from ..common.models.clip_fast_rcnn_c4_coco import model
from ..common.optim import SGD as optimizer
from ..common.train import train

# train from scratch
# train.init_checkpoint = "/home/jwyang/azureblobs/vyiwuzhong/results/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k_cc6790emb/model_final.pth"
# train.bb_rpn_checkpoint = "/home/jwyang/azureblobs/vyiwuzhong/trained_models/mrcnn_coco48_FSD/fpn/model_final.pth"
train.init_checkpoint = "/mnt/output_storage/results/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k_cc6790emb/model_final.pth"
train.bb_rpn_checkpoint = "/mnt/output_storage/trained_models/mrcnn_coco48_FSD/fpn/model_final.pth"
train.amp.enabled = True
train.ddp.fp16_compression = True
# model.backbone.bottom_up.freeze_at = 0

dataloader.train.dataset.names = "coco_2017_ovd_b_train"
dataloader.test.dataset.names = "coco_2017_ovd_all_test"

# SyncBN
# fmt: off
# model.backbone.bottom_up.stem.norm = \
#     model.backbone.bottom_up.stages.norm = \
#     model.backbone.norm = "SyncBN"

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
# model.roi_heads.box_head.conv_norm = \
#     model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
#                                                                        stats_mode="N")
# fmt: on

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
# model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
# model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
# model.roi_heads.box_head.fc_dims = [1024]

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
    L(T.RandomFlip)(horizontal=True),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

# larger batch-size.
dataloader.train.total_batch_size = 32 # 64

# Equivalent to 100 epochs.
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 184375
train.eval_period = 2500
train.output_dir = "/mnt/output_storage/results/LSJ_train_CLIPfrcnn_rn50_coco48_FSDfpnrpn_02bgloss_normtemp001_gamma05_ourclip3m600kcc6790emb"

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=CosineParamScheduler(1.0, 0.0),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer.lr = 0.004 # 0.1 # 2e-3
optimizer.weight_decay = 4e-5