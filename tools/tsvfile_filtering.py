import logging
import os
import os.path as op
from io import BytesIO
import base64
from PIL import Image
import argparse
import numpy as np
import cv2
import time
import pdb
import json

def oai_pretrained_weights_dict():
    """ Load OAI pretrained model weights, merge teacher model weights and save them into pth file
    """
    import torch
    import clip  #  the clip folder in official repo
    student_name = 'RN50x4' # 'RN50'
    teacher_name = 'RN50x4' # 'RN101' # 
    folder_path = "/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/oai_clip_weights/"
    ckpt_paths = {'RN50': folder_path+'RN50.pt', 'RN101': folder_path+'RN101.pt', 'RN50x4': folder_path+'RN50x4.pt'}

    student = clip.load(ckpt_paths[student_name], device="cpu")
    teacher = clip.load(ckpt_paths[teacher_name], device="cpu")
    student_dict = student[0].state_dict()
    teacher_dict = teacher[0].state_dict()
    new_student_dict = {} # student model
    for key in student_dict:
        if 'visual.' in key: # visual encoder
            new_key = key.replace('visual.','backbone.')
        elif 'transformer.' in key or key in ['positional_embedding', 'text_projection', 'token_embedding.weight', 'ln_final.weight', 'ln_final.bias']: # language encoder
            new_key = 'lang_encoder.' + key
        else:
            print("Keep the name of {} the same.".format(key))
            new_key = key
        new_student_dict[new_key] = student_dict[key]
    new_teacher_dict = {} # teacher model
    for key in teacher_dict:
        if 'visual.' in key: # visual encoder
            new_key = key.replace('visual.','teacher_backbone.')
        elif 'transformer.' in key or key in ['positional_embedding', 'text_projection', 'token_embedding.weight', 'ln_final.weight', 'ln_final.bias']: # language encoder
            new_key = 'teacher_lang_encoder.' + key
        else:
            new_key = 'teacher.' + key
        new_teacher_dict[new_key] = teacher_dict[key]
    # merge student and teacher into one ckpt
    for key in new_student_dict:
        if key in new_teacher_dict:
            print('Got conflict name of {}!'.format(key))
        else:
            new_teacher_dict[key] = new_student_dict[key]
    # save merged ckpt
    torch.save({'model': new_teacher_dict}, folder_path+'teacher_{}_student_{}_OAI_CLIP.pth'.format(teacher_name, student_name))

def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None

def parse_nouns(caption_tsv_path="/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/CC3M-filtered/googlecc_sbu.tsv", \
    output_file="/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/concept_pool/googlecc_nouns.txt"):
    """ Gather all captions and parse nouns from them.
    """
    caption_tsv = TSVFile(caption_tsv_path)
    all_captions = []
    for row_i in range(len(caption_tsv)):
        this_row = caption_tsv[row_i]
        img_info = this_row[0].split("_") # 'cc_cap_0_00000001', 'sbu_cap_3_00999006'
        if img_info[0] == 'cc':
            this_cap = this_row[2].lower()
            if isinstance(this_cap, str):
                all_captions.append(this_cap)
            else:
                print("Not a string text in row {}".format(row_i))        
    print("In total, we got {} captions in Google CC.".format(len(all_captions)))

    all_nouns = {}
    import spacy
    nlp = spacy.load('en_core_web_sm')
    for cap in all_captions:
        doc = nlp(cap)
        # reference from https://github.com/vacancy/SceneGraphParser/blob/master/sng_parser/backends/spacy_parser.py
        for entity in doc.noun_chunks:
            # Ignore pronouns such as "it".
            if entity.root.lemma_ == '-PRON-':
                continue

            ent = dict(
                span=entity.text,
                lemma_span=entity.lemma_,
                head=entity.root.text,
                lemma_head=entity.root.lemma_,
            )
         
            for x in entity.root.children:
                if x.dep_ == 'compound':  # compound nouns
                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']
                    # print(ent['head'])
                    # print(ent['lemma_head'])
            
            if ent['lemma_head'] in all_nouns:
                all_nouns[ent['lemma_head']] += 1
            else:
                all_nouns[ent['lemma_head']] = 1
    print("In total, we got {} unique nouns in Google CC.".format(len(all_nouns)))
    
    all_nouns_list = sorted(all_nouns.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, "w") as f:
        for item in all_nouns_list:
            f.write(item[0]+','+str(item[1])+'\n')
    print("Wrote unique nouns and their frequency into file {}".format(output_file))

def filter_nouns(input_file="/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/trained_models/concept_pool/googlecc_nouns_triplet_parser.txt", thres=100):
    """Filter some concepts"""
    remove_list = ['i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them', \
                   'my,', 'mine', 'myself', 'our', 'ours', 'ourselves', 'your', 'yours', 'yourself', 'yourselves', \
                   'his', 'himself', 'hers', 'herself', 'its', 'itself', 'their', 'theirs', 'themself', 'themselves', \
                   '#', '@', '$', '%', '^', '&', '*', '(', ')', '\'s', 'm', '1930', ]
    with open(input_file, 'r') as f, open(input_file.split('.')[0]+'_filtered_{}.txt'.format(thres), 'w') as g:
        for item in f:
            if int(item.split(',')[1].strip()) < thres:
                break
            this_concept = item.split(',')[0].lower()
            if this_concept in remove_list:
                print("Removed: {}".format(item))
                continue
            else:
                g.write(item)

def tsv_writer(values, tsv_file, sep='\t'):
    tsv_file_tmp = tsv_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            # this step makes sure python2 and python3 encoded img string are the same.
            # for python2 encoded image string, it is a str class starts with "/".
            # for python3 encoded image string, it is a bytes class starts with "b'/".
            # v.decode('utf-8') converts bytes to str so the content is the same.
            # v.decode('utf-8') should only be applied to bytes class type. 
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
    os.rename(tsv_file_tmp, tsv_file)

def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)

def regenerate_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)

class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file. 
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            regenerate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            # print('loading lineidx: {}'.format(self.lineidx))

            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            # print('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--folder", 
                        default="/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/CC3M-filtered", type=str, required=False,
                        help="Meta info folder.")                                                
    parser.add_argument("--nthreads", 
                        default=62, type=int, required=False,
                        help="Number of threads")
    parser.add_argument("--thread", 
                        default=1, type=int, required=False,
                        help="ID of current thread")

    return parser.parse_args()


def main():
    """
    This file splits the Google CC caption tsv file into 12 chunks and save them into corresponding folder.
    The captions in each folder correspond to the images that are not corrupted.
    So we re-generate the image lineidx file to align with the caption lineidx such that the corrupted images are removed.
    """
    # parse nouns from all Google CC captions, and filter these nouns
    # #parse_nouns()
    # filter_nouns()
    # pdb.set_trace()

    ########################################################################################################################
    # generate image and caption tsv files for Flickr30k-Entities test split
    # image_rows = []
    # label_rows = []
    # split = 'val' # 'test' 

    # fname = "/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/sentences_{}.json".format(split)
    # data_path = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_images/flickr30k_images/"
    # output_path = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_entities_tsv/"
    # miss_flist = []
   
    # with open(fname, "r") as fid:
    #     data = json.load(fid)
    #     for img_id in data:
    #         # caption
    #         anns = []
    #         for sent in data[img_id]:
    #             cap = sent['sentence']
    #             anns.append(cap)
    #         # image
    #         img_name = img_id + '.jpg'
    #         img_path = os.path.join(data_path, img_name)
    #         try:
    #             img_old = cv2.imread(img_path)
    #             img_encoded_str_wrong = base64.b64encode(cv2.imencode('.jpg', img_old)[1])
    #             cv2_im = cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB)
    #             with open(img_path, 'rb') as f:
    #                 image_data = f.read()
    #             img_encoded_str = base64.b64encode(image_data)
    #             img = img_from_base64(img_encoded_str)
    #             if img is None:
    #                 img_encoded_str = img_encoded_str_wrong
    #                 img = img_old
    #         except Exception as e:
    #             miss_flist.append(img_name)
    #             print("miss image {}".format(img_name))
    #             continue
    #         # accumulate data
    #         image_rows.append(["flickr30k_images/"+img_id, img_encoded_str])
    #         label_rows.append(["flickr30k_images/"+img_id, json.dumps({'captions': anns})])

    # img_file = os.path.join(output_path, 'image.tsv')
    # label_file = os.path.join(output_path, 'text.tsv')

    # tsv_writer(image_rows, img_file)
    # tsv_writer(label_rows, label_file)

    # # generate linelist file to filter out objects that don't have labels
    # temp_tsv = TSVFile(img_file, generate_lineidx=True) # generate the lineidx given a tsv file
    # temp_tsv = TSVFile(label_file, generate_lineidx=True) # generate the lineidx given a tsv file

    # with open(os.path.join(output_path, "missing_flist.txt"), 'w') as fid:
    #     fid.write("\n".join(miss_flist))

    # img_file = os.path.join(output_path, 'image.tsv')
    # label_file = os.path.join(output_path, 'text.tsv')
    # img_tsv = TSVFile(img_file)
    # label_tsv = TSVFile(label_file)
    # for i, (img_i, label_i) in enumerate(zip(img_tsv, label_tsv)):
    #     if img_i[0] != label_i[0]:
    #         pdb.set_trace()

    # pdb.set_trace()

    ########################################################################################################################
    """
    This file splits the Google CC caption tsv file into 12 chunks and save them into corresponding folder.
    The captions in each folder correspond to the images that are not corrupted.
    So we re-generate the image lineidx file to align with the caption lineidx such that the corrupted images are removed.
    """
    folder = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/CC3M-filtered" # "/mnt/output_storage/CC3M-filtered" # 
    chunks_num = 12
    
    # # 1. split the captions based on 1st column in each row
    # caption_tsv_path = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/CC3M-filtered/googlecc_sbu.tsv"
    # caption_tsv = TSVFile(caption_tsv_path)
    # cap_per_chunk = {i:[] for i in range(chunks_num)}
    # cnt = 0
    # for row_i in range(len(caption_tsv)):
    #     this_row = caption_tsv[row_i]
    #     img_info = this_row[0].split("_") # 'cc_cap_0_00000001', 'sbu_cap_3_00999006'
    #     if img_info[0] == 'cc':
    #         kept_info = this_row[1:]  # only keep last 2 columns: image name and caption string
    #         assert len(kept_info) == 2  #  format check
    #         kept_info[0] = kept_info[0].split("_")[-1] # only keep img name: ['cc_cap_00000000', 'a very typical bus station'] --> ['00000000', 'a very typical bus station']
    #         cap_per_chunk[int(img_info[2])].append(kept_info)  
    #         cnt += 1
    # print("In total, we got {} captions in Google CC.".format(cnt))
    # pdb.set_trace()

    # # 2. write data into tsv file and save to corresponding chunk folder
    # for chunk_i in cap_per_chunk:
    #     cap_save_path = os.path.join(folder, str(chunk_i), "text.tsv")
    #     this_cap = cap_per_chunk[chunk_i]
    #     tsv_writer(this_cap, cap_save_path)
    #     temp_tsv = TSVFile(cap_save_path, generate_lineidx=True) # generate the lineidx given a tsv file
    #     print("Wrote {} line idx of captions in folder {}.".format(len(this_cap), cap_save_path))
    # pdb.set_trace()
    
    ########################################################################################################################
    
    # 3. generate lineidx file for image tsv by removing some lineidx that has no captioins
    chunks_num = [3,8,9] # [3,2,1,0] # [7,6,5,4] # [11,10,9,8]
    for chunk_i in chunks_num:
        cap_tsv_file = os.path.join(folder, str(chunk_i), "text.tsv")
        cap_tsv = TSVFile(cap_tsv_file)
        cap_img_names = {cap_tsv[i][0]: 1 for i in range(len(cap_tsv))}
        img_tsv_file = os.path.join(folder, str(chunk_i), "images.tsv")
        img_tsv = TSVFile(img_tsv_file)
        img_lineidx = None
        new_img_lineidx = []
         
        for row_id in range(len(img_tsv)):
            this_img = img_tsv[row_id]
            if this_img[0] in cap_img_names:
                if img_lineidx is None:
                    img_lineidx = img_tsv._lineidx
                new_img_lineidx.append(img_lineidx[row_id])

        assert len(new_img_lineidx) == len(cap_img_names)
        
        img_lineidx_file = op.splitext(img_tsv_file)[0] + '.lineidx'
        print("Kept {} line idx of images in folder {}.".format(len(new_img_lineidx), img_lineidx_file))
        os.rename(img_lineidx_file, op.splitext(img_tsv_file)[0] + '.oldlineidx') # keep old files
        with open(img_lineidx_file, 'w') as f:
            for value in new_img_lineidx:
                f.write(str(value) + '\n')
        print("Used time: {}".format(time.time() - start))
        start = time.time()

    ########################################################################################################################

    # 4. verify the images are all valid to be loaded by Pillow, and the images are aligned with captions
    start = time.time()
    chunks_num = [0,1,2,3,4,5,6,7,8,9,10,11]
    for chunk_i in chunks_num:
        print(chunk_i)
        error_cnt = 0
        cap_tsv_file = os.path.join(folder, str(chunk_i), "text.tsv")
        cap_tsv = TSVFile(cap_tsv_file)
        img_tsv_file = os.path.join(folder, str(chunk_i), "images.tsv")
        img_tsv = TSVFile(img_tsv_file)
        new_cap_lineidx = []
        new_img_lineidx = []

        for row_id in range(len(img_tsv)):
            this_cap = cap_tsv[row_id]
            this_img = img_tsv[row_id]
            assert this_cap[0] == this_img[0]
            try:
                image = Image.open(BytesIO(base64.b64decode(this_img[1])))
                new_cap_lineidx.append(cap_tsv._lineidx[row_id])
                new_img_lineidx.append(img_tsv._lineidx[row_id])
            except: 
                print("error")
                error_cnt += 1
                continue

        assert len(new_cap_lineidx) == len(new_img_lineidx)
        
        cap_lineidx_file = op.splitext(cap_tsv_file)[0] + '.lineidx'
        img_lineidx_file = op.splitext(img_tsv_file)[0] + '.lineidx'
        print("Kept {} line idx of images in folder {}.".format(len(new_img_lineidx), img_lineidx_file))
        print("Found {} errors in folder {}.".format(error_cnt, img_lineidx_file))
        with open(cap_lineidx_file, 'w') as f:
            for value in new_cap_lineidx:
                f.write(str(value) + '\n')
        with open(img_lineidx_file, 'w') as f:
            for value in new_img_lineidx:
                f.write(str(value) + '\n')
        print("Used time: {}".format(time.time() - start))
        start = time.time()

    pdb.set_trace()

    ########################################################################################################################
    # # Not used at the end
    # # re-organize Flickr30k-Entities split
    # gt_boxes = json.load(open("/home/v-yiwuzhong/projects/azureblobs/vyiwuzhong_phillytools/flickr30k_processed/bounding_boxes_test.json"))
    # required_imgid = {item: 'test' for item in list(gt_boxes.keys())}

    # # split the captions and images based on 1st column in each row
    # tsv_types = ['text', 'image']
    # for tsv_t in tsv_types:
    #     caption_tsv_path = ["/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_entities_tsv/test/test_{}.tsv".format(tsv_t),
    #                         "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_entities_tsv/val/val_{}.tsv".format(tsv_t),
    #                         "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_entities_tsv/train/train_{}.tsv".format(tsv_t)]
    #     output_caption_tsv_path = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/flickr30k_entities_tsv/test_{}.tsv".format(tsv_t)
    #     kept_caption_tsv = []
    #     kept_imgid = {}
    #     for cap_path in caption_tsv_path:
    #         caption_tsv = TSVFile(cap_path)
    #         print("len of file is {}".format(len(caption_tsv)))
    #         for row_i in range(len(caption_tsv)):
    #             this_row = caption_tsv[row_i]
    #             img_id = this_row[0].split("/")[1] # 'flickr30k_images/1007129816'
    #             pdb.set_trace()
    #             if img_id not in required_imgid:
    #                 continue
    #             else:
    #                 if tsv_t == 'image':
    #                     pdb.set_trace()
    #                 kept_caption_tsv.append(this_row)
    #                 kept_imgid[img_id] = 1

    #     print("In total, we got {} instances from {} images in Flickr30k.".format(len(kept_caption_tsv), len(kept_imgid)))
    #     tsv_writer(kept_caption_tsv, output_caption_tsv_path)
    #     temp_tsv = TSVFile(output_caption_tsv_path, generate_lineidx=True) # generate the lineidx given a tsv file
    #     print("Wrote {} line idx of instances in folder {}.".format(len(kept_caption_tsv), output_caption_tsv_path))
    # pdb.set_trace()

    ########################################################################################################################

    # # generate lineidx for the caption tsv in each foler, based on 1st column in each row
    # caption_tsv_path = "/home/v-yiwuzhong/projects/azureblobs/vlpdatasets/CC3M-filtered/googlecc_sbu.tsv"
    # caption_tsv = TSVFile(caption_tsv_path)
    # cap_lineidx_per_chunk = {i:[] for i in range(chunks_num)}
    # cnt = 0
    # for row_i in range(len(caption_tsv)):
    #     img_info = caption_tsv[row_i][0].split("_") # 'cc_cap_0_00000001', 'sbu_cap_3_00999006'
    #     if img_info[0] == 'cc':
    #         chunk_id = int(img_info[2])
    #         cap_lineidx_per_chunk[chunk_id].append(row_i)
    #         cnt += 1
    # print("In total, we got {} captions in Google CC.".format(cnt))
    
        
    # # write the lineidx into the folder corresponding to its chunk
    # for chunk_i in cap_lineidx_per_chunk:
    #     cap_lineidx_path = os.path.join(folder, str(chunk_i), "text.lineidx")
    #     with open(cap_lineidx_path, 'w') as f:
    #         new_lineidx = cap_lineidx_per_chunk[chunk_i]
    #         for value in new_lineidx:
    #             f.write(str(value) + '\n')
    #         print("Wrote {} line idx of captions in folder {}.".format(len(new_lineidx), cap_lineidx_path))
    # import pdb; pdb.set_trace()

    # # in each folder/chunk, go through each image and only keep the images whose captions are present in "text.lineidx"
    

    
    # args = get_args()

    # # load image tsv

    # folder = args.folder

    # # files = os.listdir(folder)
    # # txt_tsvs = [tsv for tsv in files if tsv.endswith("caption.tsv")]
    # # img_tsvs = [tsv for tsv in files if tsv.endswith("img.tsv")]
    # # txt_tsvs = sorted(txt_tsvs)
    # # img_tsvs = sorted(img_tsvs)
    # # for txt_tsv_name, img_tsv_name in zip(txt_tsvs, img_tsvs):

    # txt_tsv_name = "train.{}.{}.caption.tsv".format(args.thread, args.nthreads)
    # img_tsv_name = "train.{}.{}.img.tsv".format(args.thread, args.nthreads)

    # txt_tsv_path = os.path.join(folder, txt_tsv_name)
    # img_tsv_path = os.path.join(folder, img_tsv_name)

    # img_tsv = TSVFile(img_tsv_path)
    # txt_tsv = TSVFile(txt_tsv_path)

    # import pdb; pdb.set_trace()
    
    # print(img_tsv.num_rows())
    # print(txt_tsv.num_rows())
    # assert img_tsv.num_rows() == txt_tsv.num_rows()

    # new_lineidx = []
    # for k in range(img_tsv.num_rows()):
    #     lineidx = img_tsv._lineidx[k]
    #     try:
    #         image = Image.open(BytesIO(base64.b64decode(img_tsv[k][1])))
    #         new_lineidx.append(lineidx)
    #     except: 
    #         print("error")
    #         continue

    # print("number of samples after filtering: {}".format(len(new_lineidx)))
    # with open(img_tsv_path.replace('.tsv', '.lineidx'), 'w') as fpidx_img, open(txt_tsv_path.replace('.tsv', '.lineidx'), 'w') as fpidx_txt:
    #     for value in new_lineidx:
    #         fpidx_img.write(str(value) + '\n')
    #         fpidx_txt.write(str(value) + '\n')

if __name__ == "__main__":
    main()