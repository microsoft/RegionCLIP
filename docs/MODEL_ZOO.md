# Model Zoo

This document describes our trained models, including pretrained RegionCLIP and finetuned open-vocabulary object detectors. We also present the benchmark results of open-vocabulary object detection.

## Benchmark of open-vocabulary object detection

The following two tables show the benchmarks on COCO and LVIS datasets, respectively. `Novel` denotes the novel object categories that haven't been seen during detector training. It's the main metric to be optimized in open-vocabulary object detection. `All` denotes both novel and base categories.

### COCO dataset (AP50)

|         Method        |   Pretraining Dataset  |  Detector Backbone   | Finetuning Schedule | Novel |  All  |
|-----------------------|------------------------|--------------|-------------------|-------|-------|
| [OVR](https://arxiv.org/abs/2011.10678) | COCO Caption | ResNet50-C4  | 1x  | 22.8 | 39.9 |
| [ViLD](https://arxiv.org/abs/2104.13921)| CLIP 400M    | ResNet50-FPN | 16x | 27.6 | 51.3 |
|  RegionCLIP (ours)                      | GoogleCC 3M  | ResNet50-C4  | 1x  | 31.4 | 50.4 |


### LVIS dataset (AP)

|         Method        |   Pretraining Dataset  |  Detector Backbone   | Finetuning Schedule | Novel |  All  |
|-----------------------|------------------------|--------------|-------------------|-------|-------|
| [ViLD](https://arxiv.org/abs/2104.13921)| CLIP 400M    |  ResNet152-FPN | 16x | 19.8 | 28.7 |
|  RegionCLIP (ours)                      | GoogleCC 3M  |  ResNet50x4-C4 | 1x  | 22.0 | 32.3 |



## Model downloading:
The pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii?usp=sharing). The folders include:

- `regionclip`: Our pretrained RegionCLIP models and the finetuned detectors.
- `concept_emb`: The embeddings for object class names, encoded by CLIP language encoder.
- `rpn`: The trained RPN checkpoints.
- `clip`: The trained CLIP models from [OpenAI CLIP](https://github.com/openai/CLIP). They were used as teacher models during our pretraining.

You can download the trained models as you need, and put them into respective folder in `pretrained_ckpt/`. The file structure should be
```
pretrained_ckpt/
  regionclip/
  concept_emb/
  rpn/
  clip/
```

The following sections introduce our trained models in each folder respectively.

### RegionCLIP models

#### Pretrained RegionCLIP

We use Google Conceptual Caption (3M image-text pairs) to pretrain our RegionCLIP. The pretrained models can be directly used for zero-shot inference (Table 4 in paper).

- `regionclip_pretrained-cc_{rn50, rn50x4}.pth`: `rn50` denotes that both teacher and student visual backbones are ResNet50. `rn50x4` represents ResNet50x4.


#### Finetuned open-vocabulary detectors

We further transfer our pretrained RegionCLIP for open-vocabulary object detection, including COCO and LVIS dataset. Initialized by pretrained RegionCLIP, the detectors are trained by base categories and tested on both novel and base categories (Table 1 & 2).

- `regionclip_finetuned-{coco,lvis}_{rn50, rn50x4}`: `coco`/`lvis` specifies the dataset and `rn50`/`rn50x4` represents the backbone architecture.


### Concept embeddings

We compute the embeddings of object names using CLIP's language encoder and export them as following local files. By default, the embeddings are obtained using CLIP with ResNet50 architecture, unless noted otherwise (eg, `rn50x4` denotes ResNet50x4)

- `{coco, lvis}_NUMBER_emb*.pth`: These files store the class embeddings of COCO and LVIS datasets. `NUMBER` denotes the number of classes. These class names can be viewd at `coco_zeroshot_categories.py` and `lvis_v1_categories.py` in folder `detectron2/data/datasets/`.
- `{coco, googlecc}_nouns_NUMBER.txt`: These text files include the nouns parsed from COCO Caption or Google Conceptual Caption. `NUMBER` denotes the number of nouns.
- `{coco, googlecc}_nouns_NUMBER.pth`: These files store the embeddings of the parsed nouns in text files above.

### RPN models

By default, all RPN use backbone as ResNet50. The RPN trained on COCO use ResNet50-C4 architecture. The RPN trained on LVIS use ResNet50-FPN architecture.

#### COCO dataset

- `rpn_coco_48`: Trained by 48 base categories. This is the category split of open-vocabulary object detectioin. We use it during detector finetuning on COCO dataset.
- `rpn_coco_65`: Trained by 65 (48 base + 17 novel) categories.
- `rpn_coco_80`: Trained by 80 categories. This is the commonly-used category split for fully-supervised object detection.


#### LVIS dataset

- `rpn_lvis_866`: Trained by 866 base categories. This is the category split of open-vocabulary object detectioin. We use it during RegionCLIP pretraining and zero-shot inference.
- `rpn_lvis_866_lsj`: Trained by 866 base categories with large-scale jittering augmentation and longer training time (400 epoches). We use it during detector finetuning on LVIS dataset.
- `rpn_lvis_1203_clipbackbone`: Trained by all categories (1203) in LVIS. This is the category split of fully-supervised object detectioin. Specially, it uses the [CLIP-version ResNet50](https://github.com/openai/CLIP/blob/main/clip/model.py) as backbone.


### CLIP models

For the purpose of loading CLIP models during pretraining, we simply make a copy of CLIP model weights to `teacher` module and thus each `.pth` file contains two identical set of model weights.

- `teacher_RN50_student_RN50_OAI_CLIP.pth`: Pretrained CLIP with ResNet50 architecture.
- `teacher_RN50x4_student_RN50x4_OAI_CLIP.pth`: Pretrained CLIP with ResNet50x4 architecture.
