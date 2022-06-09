# COCO categories for zero-shot setting
# 65 categories in total, 48 base categories for training, 17 unseen categories are only used in testing
# from http://ankan.umiacs.io/files/mscoco_seen_classes.json, http://ankan.umiacs.io/files/mscoco_unseen_classes.json

# 17 class names in order, obtained from load_coco_json() function
COCO_UNSEEN_CLS = ['airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', \
    'tie', 'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard', \
    'sink', 'scissors']

# 48 class names in order, obtained from load_coco_json() function
COCO_SEEN_CLS = ['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', \
    'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', \
    'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', \
    'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', \
    'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', \
    'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', \
    'refrigerator', 'book', 'clock', 'vase', 'toothbrush']

# 65 class names in order, obtained from load_coco_json() function
COCO_OVD_ALL_CLS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', \
    'bus', 'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 'horse', \
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', \
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'kite', 'skateboard', \
    'surfboard', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', \
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', \
    'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', \
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', \
    'scissors', 'toothbrush']

# 80 class names
COCO_80_ALL_CLS = {1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 12: 'stop sign',
 13: 'parking meter',
 14: 'bench',
 15: 'bird',
 16: 'cat',
 17: 'dog',
 18: 'horse',
 19: 'sheep',
 20: 'cow',
 21: 'elephant',
 22: 'bear',
 23: 'zebra',
 24: 'giraffe',
 25: 'backpack',
 26: 'umbrella',
 27: 'handbag',
 28: 'tie',
 29: 'suitcase',
 30: 'frisbee',
 31: 'skis',
 32: 'snowboard',
 33: 'sports ball',
 34: 'kite',
 35: 'baseball bat',
 36: 'baseball glove',
 37: 'skateboard',
 38: 'surfboard',
 39: 'tennis racket',
 40: 'bottle',
 41: 'wine glass',
 42: 'cup',
 43: 'fork',
 44: 'knife',
 45: 'spoon',
 46: 'bowl',
 47: 'banana',
 48: 'apple',
 49: 'sandwich',
 50: 'orange',
 51: 'broccoli',
 52: 'carrot',
 53: 'hot dog',
 54: 'pizza',
 55: 'donut',
 56: 'cake',
 57: 'chair',
 58: 'couch',
 59: 'potted plant',
 60: 'bed',
 61: 'dining table',
 62: 'toilet',
 63: 'tv',
 64: 'laptop',
 65: 'mouse',
 66: 'remote',
 67: 'keyboard',
 68: 'cell phone',
 69: 'microwave',
 70: 'oven',
 71: 'toaster',
 72: 'sink',
 73: 'refrigerator',
 74: 'book',
 75: 'clock',
 76: 'vase',
 77: 'scissors',
 78: 'teddy bear',
 79: 'hair drier',
 80: 'toothbrush'}

if __name__ == "__main__":
    # from https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/001.ipynb
    # Create zero-shot setting data split in COCO
    import json
    import ipdb

    with open('./datasets/coco/annotations/instances_train2017.json', 'r') as fin:
        coco_train_anno_all = json.load(fin)

    with open('./datasets/coco/annotations/instances_train2017.json', 'r') as fin:
        coco_train_anno_seen = json.load(fin)

    with open('./datasets/coco/annotations/instances_train2017.json', 'r') as fin:
        coco_train_anno_unseen = json.load(fin)

    with open('./datasets/coco/annotations/instances_val2017.json', 'r') as fin:
        coco_val_anno_all = json.load(fin)

    with open('./datasets/coco/annotations/instances_val2017.json', 'r') as fin:
        coco_val_anno_seen = json.load(fin)

    with open('./datasets/coco/annotations/instances_val2017.json', 'r') as fin:
        coco_val_anno_unseen = json.load(fin)
    
    labels_seen = COCO_SEEN_CLS
    labels_unseen = COCO_UNSEEN_CLS
    labels_all = [item['name'] for item in coco_val_anno_all['categories']]  # 80 class names
    # len(labels_seen), len(labels_unseen)
    # set(labels_seen) - set(labels_all)
    # set(labels_unseen) - set(labels_all)
    
    class_id_to_split = {}  # {1: 'seen', 2: 'seen', 3: 'seen', 4: 'seen', 5: 'unseen',...}
    class_name_to_split = {}  # {'person': 'seen', 'bicycle': 'seen', 'car': 'seen', 'motorcycle': 'seen', 'airplane': 'unseen',...}
    for item in coco_val_anno_all['categories']:
        if item['name'] in labels_seen:
            class_id_to_split[item['id']] = 'seen'
            class_name_to_split[item['name']] = 'seen'
        elif item['name'] in labels_unseen:
            class_id_to_split[item['id']] = 'unseen'
            class_name_to_split[item['name']] = 'unseen'
    
    # class_name_to_emb = {}
    # with open('../datasets/coco/zero-shot/glove.6B.300d.txt', 'r') as fin:
    #     for row in fin:
    #         row_tk = row.split()
    #         if row_tk[0] in class_name_to_split:
    #             class_name_to_emb[row_tk[0]] = [float(num) for num in row_tk[1:]]
    # len(class_name_to_emb), len(class_name_to_split)

    def filter_annotation(anno_dict, split_name_list):
        """
        COCO annotations have fields: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        This function (1) filters the category metadata (list) in 'categories'; 
        (2) filter instance annotation in 'annotations'; (3) filter image metadata (list) in 'images
        """
        filtered_categories = []
        for item in anno_dict['categories']:
            if class_id_to_split.get(item['id']) in split_name_list:
                #item['embedding'] = class_name_to_emb[item['name']]
                item['split'] = class_id_to_split.get(item['id'])
                filtered_categories.append(item)
        anno_dict['categories'] = filtered_categories
        
        filtered_images = []
        filtered_annotations = []
        useful_image_ids = set()
        for item in anno_dict['annotations']:
            if class_id_to_split.get(item['category_id']) in split_name_list:
                filtered_annotations.append(item)
                useful_image_ids.add(item['image_id'])
        for item in anno_dict['images']:
            if item['id'] in useful_image_ids:
                filtered_images.append(item)
        anno_dict['annotations'] = filtered_annotations
        anno_dict['images'] = filtered_images
    
    filter_annotation(coco_train_anno_seen, ['seen'])
    filter_annotation(coco_train_anno_unseen, ['unseen'])
    filter_annotation(coco_train_anno_all, ['seen', 'unseen'])
    filter_annotation(coco_val_anno_seen, ['seen'])
    filter_annotation(coco_val_anno_unseen, ['unseen'])
    filter_annotation(coco_val_anno_all, ['seen', 'unseen'])

    with open('./datasets/coco/annotations/ovd_ins_train2017_b.json', 'w') as fout:
        json.dump(coco_train_anno_seen, fout)
    with open('./datasets/coco/annotations/ovd_ins_train2017_t.json', 'w') as fout:
        json.dump(coco_train_anno_unseen, fout)
    with open('./datasets/coco/annotations/ovd_ins_train2017_all.json', 'w') as fout:
        json.dump(coco_train_anno_all, fout)
    with open('./datasets/coco/annotations/ovd_ins_val2017_b.json', 'w') as fout:
        json.dump(coco_val_anno_seen, fout)
    with open('./datasets/coco/annotations/ovd_ins_val2017_t.json', 'w') as fout:
        json.dump(coco_val_anno_unseen, fout)
    with open('./datasets/coco/annotations/ovd_ins_val2017_all.json', 'w') as fout:
        json.dump(coco_val_anno_all, fout)