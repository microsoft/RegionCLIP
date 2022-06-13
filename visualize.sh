
# evaluate ablation zeroshotinference our cliprn50 (from evaluate_models.yaml)
# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_testall.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k/model_final.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/coco_48_base_cls_emb_notnorm.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/coco_65_cls_emb_notnorm.pth \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.001 \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE False

# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_testall.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k/model_final.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/coco_48_base_cls_emb_notnorm.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/coco_65_cls_emb_notnorm.pth \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.001 \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True


# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k/model_final.pth \
# MODEL.CLIP.CROP_REGION_TYPE GT \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/lvis_1203_cls_emb_notnorm.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE False \

# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k/model_final.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/lvis_1203_cls_emb_notnorm.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.02


# evaluate ablation zeroshotinference our cliprn50x4 (from evaluate_models.yaml)
# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k_teacherrn50x4_studentrn50x4/model_0574999.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/lvis_1203_cls_emb_notnorm_rn50x4.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.02 \
# MODEL.RESNETS.DEPTH 200 \
# MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
# MODEL.CLIP.TEXT_EMB_DIM 640 \


# visualize FPV images with zeroshotinference our cliprn50x4 (adapted from above, changed NMS_THRESH_TEST)
# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k/model_final.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/lvis_1203_cls_emb_notnorm.pth \
# MODEL.CLIP.NO_BOX_DELTA True \
# MODEL.MASK_ON False \
# MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
# MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
# MODEL.ROI_HEADS.NMS_THRESH_TEST 0.3 \
# MODEL.CLIP.CLSS_TEMP 0.01 \
# MODEL.CLIP.MULTIPLY_RPN_SCORE True \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.02


# visualize FPV images with zeroshotinference our cliprn50x4 (adapted from above, changed NMS_THRESH_TEST)
python3 ./tools/train_net.py \
--num-gpus 2 \
--eval-only \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
MODEL.WEIGHTS /home/user/Desktop/RegionCLIP_data/data/dpretrain_frzt2_b96_glbimg_cntrstkl_exp639_lr0002_cc3m_600k_teacherrn50x4_studentrn50x4/model_final.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
MODEL.CLIP.TEXT_EMB_PATH /home/user/Desktop/RegionCLIP_data/data/lvis_cls_emb/lvis_1203_cls_emb_notnorm_rn50x4.pth \
MODEL.CLIP.NO_BOX_DELTA True \
MODEL.MASK_ON False \
MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True \
MODEL.CLIP.OFFLINE_RPN_NMS_THRESH 0.9 \
MODEL.ROI_HEADS.NMS_THRESH_TEST 0.3 \
MODEL.CLIP.CLSS_TEMP 0.01 \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.02 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.TEXT_EMB_DIM 640 \

