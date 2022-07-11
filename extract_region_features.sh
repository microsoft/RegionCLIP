# Extract region features for a folder of images

# RN50, LVIS 1203 concepts
python3 ./tools/extract_region_features.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
INPUT_DIR ./datasets/custom_images \
OUTPUT_DIR ./output/region_feats \
TEST.DETECTIONS_PER_IMAGE 100 \

# # RN50, features of RPN proposals
# python3 ./tools/extract_region_features.py \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# INPUT_DIR ./datasets/custom_images \
# OUTPUT_DIR ./output/region_feats \
# MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST 100 \

# # RN50x4, LVIS 1203 concepts
# python3 ./tools/extract_region_features.py \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml \
# MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
# MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
# MODEL.CLIP.TEXT_EMB_DIM 640 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# INPUT_DIR ./datasets/custom_images \
# OUTPUT_DIR ./output/region_feats \
# TEST.DETECTIONS_PER_IMAGE 100 \