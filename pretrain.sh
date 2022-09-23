# NOTE: The pre-training section is still under construction.
#       The pre-training code was already released (PretrainFastRCNN class).
#       Now we release the config files and scripts (un-tested yet), as requested by researchers.
#       We will release the pre-training data (image-text pairs) in near future.


# Distributed training across multiple nodes
# ResNet50 (default: batch 96, lr 0.002, 32 GPUs)
python3 -m launch --nnodes=2 --nproc_per_node=16 --master_port 12345 ./tools/train_net.py \
--num-gpus 16 \
--config-file ./configs/pretrain/RegionCLIP_RN50.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/coco_nouns_4764_emb.pth \
OUTPUT_DIR ./output/pretrain \


# ResNet50x4 (default: batch 96, lr 0.002, 32 GPUs)
python3 -m launch --nnodes=2 --nproc_per_node=16 --master_port 12345 ./tools/train_net.py \
--num-gpus 16 \
--config-file ./configs/pretrain/RegionCLIP_RN50x4.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50x4_student_RN50x4_OAI_CLIP.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/coco_nouns_4764_emb_rn50x4.pth \
OUTPUT_DIR ./output/pretrain \
