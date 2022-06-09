# VISUALIZATION
python ./tools/visualize_json_results.py \
--output /home/v-yiwuzhong/projects/detectron2-open-set/output/regions/ \
--input /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/results/train_CLIP_fast_rcnn_resnet50_coco48_FSDfpnrpn_005bgloss_notnormtemp_focallosssoftmaxgamma1_ourclip/inference/coco_instances_results.json \
--dataset coco_2017_ovd_t_test

# --input /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/results/train_CLIP_fast_rcnn_resnet50_coco48_FSDfpnrpn_005bgloss_notnormtemp_ourclip_testall/inference/coco_instances_results.json \
# --dataset coco_2017_ovd_all_test
# --input /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/results/train_mask_rcnn_resnet50_baseline_gpus8_bs16_lr0.02_1x_C4_COCO/inference/coco_instances_results.json \
# --dataset coco_2017_val


# python ./tools/train_net.py \
# --config-file ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# --eval-only \
# OUTPUT_DIR ./output \
# MODEL.WEIGHTS ./pretrained_ckpt/model_final_571f7c.pkl

# python ./tools/train_net.py \
# --num-gpus 2 \
# --config-file ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# --eval-only \
# OUTPUT_DIR ./output \
# MODEL.WEIGHTS /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/mrcnn_supervised/fpn/model_final.pth

# python3 ./tools/train_net.py \
# --num-gpus 2 \
# --eval-only \
# --config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4.yaml \
# MODEL.WEIGHTS /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/results/train_CLIP_fast_rcnn_resnet50_offlinesupervisedclipc4rpn/model_0049999.pth \
# MODEL.CLIP.CROP_REGION_TYPE RPN \
# MODEL.CLIP.IMS_PER_BATCH_TEST 8 \
# MODEL.CLIP.BB_RPN_WEIGHTS /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/mrcnn_clip_backbone_supervised/c4/model_final.pth \
# MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER True \
# MODEL.CLIP.TEXT_EMB_PATH /home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/lvis_cls_emb/lvis_1203_cls_emb.pth \
# MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_CLIP_R_50_C4_1x.yaml \
# MODEL.CLIP.NO_BOX_DELTA False \
# SOLVER.IMS_PER_BATCH 2 \
# DATALOADER.NUM_WORKERS 8 \
# SOLVER.BASE_LR 0.002