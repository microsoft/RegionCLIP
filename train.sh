#CUDA_LAUNCH_BLOCKING=1 \

python ./tools/train_net.py \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
--num-gpus 2 \
OUTPUT_DIR ./output \
SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.CLIP.IMS_PER_BATCH_TEST 2 \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_CLIP_R_50_C4_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/mrcnn_clip_backbone_supervised/c4/model_final.pth \
MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
MODEL.CLIP.TEXT_EMB_PATH /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth \
MODEL.CLIP.NO_BOX_DELTA False \
MODEL.MASK_ON False \
MODEL.ROI_HEADS.PROPOSAL_APPEND_GT False 

#MODEL.WEIGHTS /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/oai_clip_weights/RN50_OAI_CLIP.pth \
#MODEL.BACKBONE.FREEZE_AT 2

#--eval-only \
#MODEL.BACKBONE.FREEZE_AT 2
#SOLVER.CLIP_GRADIENTS.ENABLED True
#./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml \
#./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#./configs/LVISv1-InstanceSegmentation/mask_rcnn_CLIP_R_50_C4_1x.yaml \
#./configs/LVISv1-InstanceSegmentation/mask_rcnn_CLIP_R_50_FPN_1x.yaml \
#./configs/LVISv1-InstanceSegmentation/CLIP_rcnn_R_50.yaml \
#./configs/LVISv1-InstanceSegmentation/CLIP_rcnn_VITB32.yaml \


