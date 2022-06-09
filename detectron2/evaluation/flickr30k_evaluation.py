import logging
import numpy as np
import os
from collections import OrderedDict
from detectron2.config import global_cfg as cfg
import torch
from fvcore.common.file_io import PathManager
from detectron2.structures.boxes import pairwise_iou

from detectron2.utils.comm import all_gather, is_main_process, synchronize
import pickle
from .evaluator import DatasetEvaluator
import json
from detectron2.structures import Boxes
import html
import ftfy
import regex as re

PATTN = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class FLICKR30KEvaluator(DatasetEvaluator):

    """
    Evaluate semantic segmentation
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.gt_boxes = json.load(open("/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/bounding_boxes_test.json"))
        self.gt_sents = json.load(open("/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/sentences_test.json"))

    def reset(self):
        self._predictions = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        assert len(inputs) == 1  # batch = 1 during inference
        dataset_name, img_id, (img_height, img_width), all_str2id_links = inputs[0][-1]
        img_id = img_id.split('/')[-1]
        match_scores, processed_results = outputs
        match_scores = match_scores.to(self._cpu_device)
        pred_boxes = processed_results[0]['instances'].proposal_boxes.to(self._cpu_device)

        self._predictions.update({img_id: [img_height, img_width, all_str2id_links, match_scores, pred_boxes]})

    def merge_gt_boxes(self, box_anno):
        gt_boxes = []
        phrase_ids = []
        scene_box_ids = box_anno['scene']
        for k, v in box_anno['boxes'].items():
            if k in scene_box_ids: # important: remove scene boxes, otherwise the number of each phrase type cannot match paper
                continue
            phrase_ids.append(k)
            if len(v) == 1:
                gt_boxes.append(v[0])
            else:
                # when a phrase respond to multiple regions, we take the union of them as paper given
                v = np.array(v)
                box = [v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()]
                gt_boxes.append(box)
        gt_boxes = np.array(gt_boxes)
        return phrase_ids, gt_boxes

    def find_ground_box(self, match_scores, all_str2id_links, sentences, gt_phrase_ids):
        """ Given matching matrix between region feats and token feats, find the box that grounds a phrase
        """
        num_box = match_scores.size(0)
        num_cap = int(match_scores.size(1) / 77)
        all_phrase_score = []
        all_phrase_ids = []
        for i in range(num_cap): # per sentence
            this_score = match_scores[:, i*77:(i+1)*77]  # [#boxes, 77]
            input_ids = [iitem for item in all_str2id_links[i] for iitem in item[1]]
            input_tokens = [item[0] for item in all_str2id_links[i]]
            phrases = sentences[i]['phrases']
            for j, phrase in enumerate(phrases):  # per phrase
                if phrase['phrase_id'] not in gt_phrase_ids:  #  no gt box for this phrase, skip
                    continue
                # locate the word
                words = whitespace_clean(basic_clean(phrase['phrase'])).lower() # phrase['phrase'].lower().replace("-"," ").split()
                words = re.findall(PATTN, words)
                first_word_index = None  #  phrase['first_word_index']
                for idx in range(len(input_tokens) - len(words) + 1):  # search start word of this phrase
                    if input_tokens[idx : idx + len(words)] == words:  # NOTE: key step for alignment btw model prediction and annotation
                        first_word_index = idx 
                        break
                if first_word_index is None:
                    print("Fail to find phrase [{}] in input tokens [{}]".format(words, input_tokens))
                start_wd_ind = first_word_index
                end_wd_ind = first_word_index + len(words)
                if len(words) != len(phrase['phrase'].split()):
                    pass # print('tokens: {} <--> phrase: {}'.format(words, phrase['phrase']))
                # locate the token
                start_tk_ind = 0
                for k_i, k in enumerate(range(0, start_wd_ind)):
                    start_tk_ind += len(all_str2id_links[i][k][1])
                token_cnt = 0
                for k_i, k in enumerate(range(start_wd_ind, end_wd_ind)):
                    if all_str2id_links[i][k][0] != words[k_i]:
                        print("Word not matched: {} in model output but {} in annotation".format(all_str2id_links[i][k][0], words[k_i]))
                    else:
                        token_cnt += len(all_str2id_links[i][k][1]) # ith sentence, kth word, and its tokens
                end_tk_ind = start_tk_ind + token_cnt
                # sanity check
                phrase_ids1 = [iitem for item in all_str2id_links[i][start_wd_ind:end_wd_ind] for iitem in item[1]]  # way 1: use word index to accumulate token ids in a phrase
                phrase_ids2 = input_ids[start_tk_ind:end_tk_ind] # way 2: use token index to directly index token ids in a phrase
                if phrase_ids1 != phrase_ids2:
                    print("Santity check: {} from word {} in token".format(phrase_ids1, phrase_ids2))
                # index similarity score
                phrase_score = this_score[:, start_tk_ind:end_tk_ind]
                phrase_score = phrase_score.mean(dim=1) # phrase_score.max(dim=1)[0] # 
                all_phrase_score.append(phrase_score)
                all_phrase_ids.append(phrase['phrase_id'])
        phrase_score_tensor = torch.cat(all_phrase_score)
        phrase_score_tensor = phrase_score_tensor.view(len(all_phrase_ids), num_box) # NOTE: this should be [#phrases, #object proposals]

        return phrase_score_tensor, all_phrase_ids

    def evaluate(self):
        """
        Evaluates Referring Segmentation IoU:
        """

        if self._distributed:
            synchronize()

            self._predictions = all_gather(self._predictions)

            if not is_main_process():
                return

            all_prediction = {}
            for p in self._predictions:
                all_prediction.update(p)
        else:
            all_prediction = self._predictions
        
        if len(all_prediction) < 30:  # resume inference results
            save_path = "/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/grounding_results/grounding_{}_imgs.npy".format(1000)
            all_prediction = np.load(save_path, allow_pickle=True).tolist()
            self._logger.info('Resume from {}'.format(save_path))
        else:  # new run
            save_path = "/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/grounding_results/grounding_{}_imgs.npy".format(len(all_prediction))
            np.save(save_path, all_prediction)
            self._logger.info('Save results to {}'.format(save_path))
        self._logger.info('Got {} images!'.format(len(all_prediction)))
        
        image_unique_ids = list(all_prediction.keys())
        image_evaled = []

        total_num = 0
        recall_num = 0
        num_type = {}
        recall_type = {}
        acc_type = {}
        recall_topk_num = {5:0, 10:0}
        point_recall_num = 0
        EVAL_THRESH = 0.5
        type_cnts = {}

        for img_sent_id in image_unique_ids:
            if img_sent_id not in self.gt_boxes:
                continue
            else:
                image_evaled.append(img_sent_id)
            # results from model
            result = all_prediction[img_sent_id]
            phrase_ids = None 
            phrase_types = []  #  phrase type: each phrase belongs to a coarse object concept
            pred_boxes = None  #  an object proposal selected by model for each phrase
            img_height, img_width, all_str2id_links = result[0], result[1], result[2]  # all_str2id_links: each word and its tokenized ids
            match_scores = result[3]  # matching score [#object proposals, #tokens]
            precomp_boxes = result[4]  # object proposals from offline module
            # annotation from dataset
            sentences = self.gt_sents[img_sent_id]
            box_anno = self.gt_boxes[img_sent_id]
            # sanity check and box merging
            assert box_anno['height'] == img_height, box_anno['width'] == img_width
            gt_phrase_ids, gt_boxes = self.merge_gt_boxes(box_anno)  # merged if multiple boxes for the same phrase
            if len(gt_phrase_ids) == 0: # no gt box for this image
                continue
            for sent_item in sentences:
                for phrase_item in sent_item['phrases']:
                    if phrase_item['phrase_id'] in gt_phrase_ids:
                        phrase_types.append(phrase_item['phrase_type']) 

            # merge similarity scores from token level to phrase level, and find the box that grounds the phrase
            phrase_score_tensor, all_phrase_ids = self.find_ground_box(match_scores, all_str2id_links, sentences, gt_phrase_ids)  
            pred_boxes_ind = torch.argmax(phrase_score_tensor, dim=1)
            pred_boxes = precomp_boxes[pred_boxes_ind]
            pred_similarity = phrase_score_tensor # .t() #  pred_similarity: matching score [#phrases, #object proposals]
            
            # get single target/gt box for each phrase
            # 1. any gt box that can be matched as target 
            # refer to (https://github.com/BigRedT/info-ground/blob/22ae6d6ec8b38df473e73034fc895ebf97d39897/exp/ground/eval_flickr_phrase_loc.py#L90)
            phrase_boxes = [box_anno['boxes'][p_id] for p_id in all_phrase_ids]
            targets = []
            for pr_b, pd_b in zip(phrase_boxes, pred_boxes):
                matched = False
                for single_b in pr_b:
                    this_iou = pairwise_iou(Boxes(torch.from_numpy(np.array([single_b])).float()), Boxes(pd_b.view(1,-1)))
                    if (this_iou >= EVAL_THRESH).sum() > 0:
                        targets.append(single_b)
                        matched = True
                        break
                if not matched:
                    targets.append(single_b)
            targets = Boxes(torch.from_numpy(np.array(targets)).float())
            # 2. union box as target
            # target_ind = np.array([gt_phrase_ids.index(p_id) for p_id in all_phrase_ids])
            # targets = gt_boxes[target_ind] # ground-truth boxes for each phrase in each sentence
            # targets = Boxes(torch.from_numpy(targets).float())
            assert len(phrase_types) == len(targets)

            # single predicted box for each phrase
            ious = pairwise_iou(targets, pred_boxes)  # this function will change the target_boxes into cuda mode
            iou = ious.numpy().diagonal()
            total_num += iou.shape[0]
            recall_num += int((iou >= EVAL_THRESH).sum())  # 0.5

            # metric of point (can be ignored)
            pred_boxes_tensor = pred_boxes.tensor
            pred_center = (pred_boxes_tensor[:, :2] + pred_boxes_tensor[:, 2:]) / 2.0
            pred_center = pred_center.repeat(1, 2)  ## x_c, y_c, x_c, y_c
            targets_tensor = targets.tensor
            fall_tensor = targets_tensor - pred_center
            fall_tensor = (fall_tensor[:, :2] <= 0).float().sum(1) + (fall_tensor[:, 2:] >= 0).float().sum(1)
            point_recall_num += (fall_tensor == 4).float().numpy().sum()

            # detailed accuracy across different phrase types
            for pid, p_type in enumerate(phrase_types):
                p_type = p_type[0]
                num_type[p_type] = num_type.setdefault(p_type, 0) + 1
                recall_type[p_type] = recall_type.setdefault(p_type, 0) + (iou[pid] >= EVAL_THRESH)
            
            # metric of recall when multiple predicted boxes for each phrase
            ious_top = pairwise_iou(targets, precomp_boxes).cpu()
            for k in [5, 10]:
                top_k = torch.topk(pred_similarity, k=k, dim=1)[0][:, [-1]]
                pred_similarity_topk = (pred_similarity >= top_k).float()
                ious_top_k = (ious_top * pred_similarity_topk).numpy()
                recall_topk_num[k] += int(((ious_top_k >= EVAL_THRESH).sum(1) > 0).sum())

        acc = recall_num / total_num
        acc_top5 = recall_topk_num[5] / total_num
        acc_top10 = recall_topk_num[10] / total_num
        point_acc = point_recall_num / total_num
        
        # details about each coarse type of phrase
        for type, type_num in num_type.items():
            acc_type[type] = recall_type[type] / type_num

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "prediction_{}.pkl".format(str(acc).replace('.', '_')[:6]))
        #     with PathManager.open(file_path, "wb") as f:
        #         pickle.dump(all_prediction, f)

        del all_prediction
        self._logger.info('evaluation on {} expression instances, detailed_iou: {}'.format(len(image_evaled), acc_type))
        self._logger.info('Evaluate Pointing Accuracy: PointAcc:{}'.format(point_acc))
        results = OrderedDict({"acc": acc, "acc_top5": acc_top5, "acc_top10": acc_top10})
        self._logger.info(results)
        self._logger.info(num_type)
        return results