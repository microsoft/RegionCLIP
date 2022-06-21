# RegionCLIP: Region-based Language-Image Pretraining

This is the official PyTorch implementation of RegionCLIP (CVPR 2022).

[**Paper**](https://arxiv.org/abs/2112.09106) | [**Slides**](https://drive.google.com/file/d/1EepNVJGo_d73Glr4vNjR4Av0dNkBCGcj/view?usp=sharing)

> **RegionCLIP: Region-based Language-Image Pretraining (CVPR 2022)** <br>
> [Yiwu Zhong](https://pages.cs.wisc.edu/~yiwuzhong/), [Jianwei Yang](https://jwyang.github.io/), [Pengchuan Zhang](https://pzzhang.github.io/pzzhang/), [Chunyuan Li](https://chunyuan.li/), [Noel Codella](https://noelcodella.github.io/publicwebsite/), [Liunian Li](https://liunian-harold-li.github.io/), [Luowei Zhou](https://luoweizhou.github.io/), [Xiyang Dai](https://sites.google.com/site/xiyangdai/), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en), [Yin Li](https://www.biostat.wisc.edu/~yli/), and [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F) <br>

<p align="center">
<img src="docs/regionclip.png" width=80% height=80%
class="center">
</p>

## Overview

We propose RegionCLIP that significantly extends CLIP to learn region-level visual representations. RegionCLIP enables fine-grained alignment between image regions and textual concepts, and thus supports region-based reasoning tasks including zero shot object detection and open-vocabulary object detection.

- **Pretraining**: We leverage a CLIP model to match image regions with template captions, and then pretrain our model to align these region-text pairs.
- **Zero-shot inference**: Once pretrained, the learned region representations support zero-shot inference for object detection.
- **Transfer learning**: The learned RegionCLIP model can be further fine-tuned with additional object detection annotations, allowing our model to be used for fully supervised or open-vocabulary object detection.
- **Results**: Our method demonstrates **state-of-the-art** results for zero-shot object detection and open vocabulary object detection.

## Updates

* [06/20/2022] We released models and inference code for our RegionCLIP!

## Outline

1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Model Zoo](#Model-Zoo)
4. [Zero-shot Inference](#Zero-shot-Inference)
5. [Transfer Learning](#Transfer-Learning)
6. [Extract Region Features](#Extract-Region-Features)
7. [Citation and Acknowledgement](#Citation-and-Acknowledgement)
8. [Contributing](#Contributing)

## Installation

Check [`INSTALL.md`](docs/INSTALL.md) for installation instructions.

## Datasets

Check [`datasets/README.md`](datasets/README.md) for dataset preparation.

## Model Zoo

Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) for our pretrained models.


## Zero-shot Inference

After pretraining, RegionCLIP can directly support the challenging zero-shot object detection task **without finetuning on detection annotation**. Given an input image, our pretrained RegionCLIP can match image region features to object concept embeddings, and thus recognize image regions into many object categories. The image regions are produced by a region localizer (e.g., RPN), where the object class names come from a dictionary **specifiied by users**.


### Visualization on custom images

We provide an example below for zero-shot object detection with pretrained RegionCLIP on custom images and for visualizing the results.

<details>

<summary>
Before detecting objects, please prepare pretrained models, label files, and the custom images. See details below.
</summary>
  
- Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) to 
  - download the pretrained model checkpoint `regionclip_pretrained-cc_rn50x4.pth` (RegionCLIP with ResNet50x4) to the folder `./pretrained_ckpt/regionclip`.
  - download the class embeddings `lvis_1203_cls_emb_rn50x4.pth` to the folder `./pretrained_ckpt/concept_emb/`.
- Check [`datasets/README.md`](datasets/README.md) to download LVIS label file `lvis_v1_val.json` and put it in the folder `./datasets/lvis/lvis_v1_val.json`. The file is used to specify object class names.
- Put all custom images in the folder `./datasets/custom_images/`.

</details>
  
<details>

<summary>
After preparation, run the following script to detect objects.
</summary>
  
```
python3 ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
```

</details>

<details>

<summary>
The detection results will be stored as the file "./output/inference/lvis_instances_results.json". To visualize it, run the script below.
</summary>
 
```
 python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/regions \
--dataset lvis_v1_val_custom_img \
--conf-threshold 0.05 \
--show-unique-boxes \
--max-boxes 25 \
--small-region-px 8100\ 
```
</details> 

The visualized images will be placed at `./output/regions/`. The visualized images would look like:

<p align="center">
<img src="docs/sample_img1_vis.jpg" width=80% height=80%
class="center">
</p>

In this example, the detection results come from our pretrained RegionCLIP with ResNet50x4 architecture. The regions are proposed by an RPN trained by 866 object categories from LVIS dataset. For now, we use 1203 object class names from LVIS dataset for this visualization example. We also include an example in `visualize_zeroshot_inference.sh` with our pretrained RegionCLIP (ResNet50 architecture).


### Evaluation for zero-shot inference

We provide an example below for evaluating our pretrained RegionCLIP (ResNet50) using ground-truth boxes on COCO dataset. This will reproduce our results in Table 4 of the paper.

<details>

<summary>
Before evaluation, please prepare pretrained models and set up the dataset.
</summary>
  
- Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) to 
  - download the pretrained RegionCLIP checkpoint `regionclip_pretrained-cc_rn50.pth` to the folder `./pretrained_ckpt/regionclip`.
  - download the class embeddings `coco_65_cls_emb.pth` to the folder `./pretrained_ckpt/concept_emb/`.
- Check [`datasets/README.md`](datasets/README.md) to set up COCO dataset.

</details>
  
<details>

<summary>
After preparation, run the following script to evaluate the pretrained model in zero-shot inference setting.
</summary>
  
```
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.CLIP.MULTIPLY_RPN_SCORE False \
```

</details>

For more examples, please refer to `test_zeroshot_inference.sh`. This script covers a wide combination of pretrained models (ResNet50, ResNet50x4), datasets (COCO, LVIS) and region proposal types (ground-truth regions, RPN proposals). Also, please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for available trained models and [`datasets/README.md`](datasets/README.md) for setting up COCO and LVIS datasets.

## Transfer Learning

Our pretrained RegionCLIP can be further **fine-tuned** when human annotations of objects are available. In this transfer learning setting, we demonstrate results on **open-vocabulary object detection**, where the object detector is trained on base categories and evaluated on both base and **novel** categories.

We show an example for running a trained detector on custom images. Further, we provide scripts of training and evaluation for the benchmark of **open-vocabulary object detection**, including COCO and LVIS datasets (Table 1 & 2 in paper).


### Visualization on custom images

We provide an example below for running a trained open-vocabulary object detector on custom images and for visualizing the results. In this example, the detector is initialized using RegionCLIP (RN50x4), trained on 866 LVIS base categories, and is tasked to detect all 1203 categories on LVIS.

<details>

<summary>
Before detecting objects, please prepare the trained detectors, label files, and the custom images.
</summary>
  
- Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) to 
  - download the trained detector checkpoint `regionclip_finetuned-lvis_rn50x4.pth` to the folder `./pretrained_ckpt/regionclip`.
  - download the trained RPN checkpoint `rpn_lvis_866_lsj.pth` to the folder `./pretrained_ckpt/rpn`.
  - download the class embeddings `lvis_1203_cls_emb_rn50x4.pth` to the folder `./pretrained_ckpt/concept_emb/`.
- Check [`datasets/README.md`](datasets/README.md) to download label file `lvis_v1_val.json` and put it in the folder `./datasets/lvis/lvis_v1_val.json`.
- Put all custom images in the folder `./datasets/custom_images/`.

</details>
  
<details>

<summary>
After preparation, run the following script to detect objects and visualize the results.
</summary>

```
# for simplicity, we integrate the script in visualize_transfer_learning.sh
bash visualize_transfer_learning.sh
```

</details>

  
The visualized images will be placed at `./output/regions/`.


### Evaluate the trained detectors

We provide an example below for evaluating our open-vocabulary object detector, initialized by RegionCLIP (ResNet50) and trained on COCO dataset.

<details>

<summary>
Before evaluation, please prepare the trained detector and set up the dataset.
</summary>
  
- Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) to 
  - download the trained detector checkpoint `regionclip_finetuned-coco_rn50.pth` to the folder `./pretrained_ckpt/regionclip`, 
  - download the trained RPN checkpoint `rpn_coco_48.pth` to the folder `./pretrained_ckpt/rpn`,
  - download the class embeddings `coco_48_base_cls_emb.pth` and `coco_65_cls_emb.pth` to the folder `./pretrained_ckpt/concept_emb/`.
- Check [`datasets/README.md`](datasets/README.md) to set up COCO dataset.

</details>
  
<details>

<summary>
After preparation, run the following script to evaluate the trained open-vocabulary detector.
</summary>
  
```
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
```

</details>


For more examples, please refer to `test_transfer_learning.sh`. This script includes benchmark evaluation for various combination of trained detectors (ResNet50, ResNet50x4) and datasets (COCO, LVIS). Also, please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for available trained models and [`datasets/README.md`](datasets/README.md) for setting up COCO and LVIS datasets.


### Train detectors on your own

We provide an example below for training an open-vocabulary object detector on COCO dataset, with pretrained RegionCLIP (ResNet50) as the initialization.

<details>

<summary>
Before training, please prepare our pretrained RegionCLIP model and set up the dataset.
</summary>
  
- Check [`MODEL_ZOO.md`](docs/MODEL_ZOO.md) to 
  - download the pretrained RegionCLIP checkpoint `regionclip_pretrained-cc_rn50.pth` to the folder `./pretrained_ckpt/regionclip`, 
  - download the trained RPN checkpoint `rpn_coco_48.pth` to the folder `./pretrained_ckpt/rpn`,
  - download the class embeddings `coco_48_base_cls_emb.pth` and `coco_65_cls_emb.pth` to the folder `./pretrained_ckpt/concept_emb/`.
- Check [`datasets/README.md`](datasets/README.md) to set up COCO dataset.

</details>
  
<details>

<summary>
After preparation, run the following script to train an open-vocabulary detector.
</summary>
  
```
python3 ./tools/train_net.py \
--num-gpus 1 \
--config-file ./configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
```

</details>

For more examples, please refer to `train_transfer_learning.sh`. This script provides training scripts for various combination of detector backbones (ResNet50, ResNet50x4) and datasets (COCO, LVIS). Also, please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for available trained models and [`datasets/README.md`](datasets/README.md) for setting up COCO and LVIS datasets.


## Extract Region Features

Under construction. We're working on scripts for extracting region features from our pretrained models.


## Citation and Acknowledgement
### Citation

If you find this repo useful, please consider citing our paper:

```
@inproceedings{zhong2022regionclip,
  title={Regionclip: Region-based language-image pretraining},
  author={Zhong, Yiwu and Yang, Jianwei and Zhang, Pengchuan and Li, Chunyuan and Codella, Noel and Li, Liunian Harold and Zhou, Luowei and Dai, Xiyang and Yuan, Lu and Li, Yin and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16793--16803},
  year={2022}
}
```

### Acknowledgement

This repository was built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [CLIP](https://github.com/openai/CLIP), [OVR-CNN](https://github.com/alirezazareian/ovr-cnn), and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). We thank the effort from our community.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
