# Prepare Datasets

We provide instruction for preparing datasets, including pretraining (under construction) and finetuning (COCO and LVIS).

The following instruction is adapted from [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md).

## Preliminary

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

Detectron2 has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for [COCO dataset](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
    ovd_ins_{train,val}2017_{all,b,t}.json # for open-vocabulary object detection
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

**Note**: `ovd_ins_{train,val}2017_{all,b,t}.json` is obtained from [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb) for creating the open-vocabulary COCO split. `b` represents 48 base categories, `t` represents 17 novel categories and `all ` denotes both base and novel categories. You can also download them from [this Google Drive](https://drive.google.com/drive/folders/1Ew24Rua-LAuNeK6OsaglrYwkpLSG8Xmt?usp=sharing).

Since the folder `coco/` is large in size, you could soft link it in dataset directory. For example, run `ln -s DIR_to_COCO datasets/coco`.


## Expected dataset structure for [LVIS dataset](https://www.lvisdataset.org/dataset):
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v1_{train,val}.json
  lvis_v1_image_info_test{,_challenge}.json
```

Since the folder `lvis/` is large in size, you could soft link it in dataset directory. For example, run `ln -s DIR_to_LVIS datasets/lvis`.

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

