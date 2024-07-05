#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
import h5py
from torch.functional import F
import os
import math
from configs.dataset_class import activity_dict
import yaml
import tqdm
import sys
from utils.arguments import handle_args, modify_config

import pdb

# TODO: the 3 classes SPOTDataset, SPOTDatasetPretrain, SPOTDatasetUnlabeled should be made into one class

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class SPOTDataset(data.Dataset):
    def __init__(self, subset="train", mode="train"):
        # super().__init__()
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode

        # HACK: this is just for trying out this fix, really there should not be self.mode and self.split (until we want kfold). 
        self.split = {'train': 'training', 'validation': 'testing', 'testing': 'testing'}[self.subset]

        self.feature_path = config[self.split]['feature_path'] # FIX: something like config[self.mode] instead, and the same for other instances of config['training'] in this file.
        self.unlabel_percent = config['dataset']['training']['unlabel_percent']
        self.video_info_path_unlabeled = config['dataset']['training']['video_info_path_unlabeled']
        self.video_info_path = os.path.join(self.video_info_path_unlabeled,"video_info_new_"+str(self.unlabel_percent)+".csv")
        # self.video_info_path = config['dataset']['training']['video_info_path']
        self.video_anno_path = config['dataset']['training']['video_anno_path']
        self.num_frame = config['dataset']['training']['num_frame']
        self.num_classes = config['dataset']['num_classes']
        self.class_to_idx = activity_dict
        video_infos = self.get_video_info(self.video_info_path)
        self.info = video_infos
        video_annos = self.get_video_anno(video_infos, self.video_anno_path)
        #print("len(video_annos)", len(video_annos))
        """
        Issue: it is looking for a CSV file within self.feature_path, rather than expecting a .npy
        (Pdb) self.getVideoData(0)
        *** FileNotFoundError: [Errno 2] No such file or directory: '/data/i5O/ActivityNet1.3/train/v_---9CpRcKoU.csv'
        """
        self.subset_mask = self.getVideoMask(video_annos,self.temporal_scale)
        self.subset_mask_list = list(self.subset_mask.keys())
        #print("len(self.subset_mask_list)", len(self.subset_mask_list))


        
    def get_video_anno(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)

        #breakpoint()

        # print(anno_database)
        video_dict = {}
        for video_name in video_infos.keys():
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]['subset']
            if self.subset in video_subset:
                video_info.update({'subset': video_subset})
                video_dict[video_name] = video_info


        return video_dict

    

    def get_video_info(self,video_info_path):
        """ Gets the relevant info (duration, subset) out of the video_info_new* dataframe.
        """

        df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        video_infos = {}
        self.v_list ={}
        for info in df_info:
            video_infos[info[0]] = {
                'duration': info[2],
                'subset': info[5]
            }
            
        return video_infos

    def getAnnotation(self, subset, anno):
        """ Filters based on the temporal scale.
        """

        temporal_dict={}
        for idx in anno.keys():

            #breakpoint()

            labels = anno[idx]['annotations']
            subset_vid  = anno[idx]['subset']
            num_frame = anno[idx]['feature_frame']
            vid_frame = anno[idx]['duration_frame']
            num_sec = anno[idx]['duration_second']
            corr_sec = float(num_frame) / vid_frame * num_sec # The number of seconds in the video frames that are actually used
            label_list= []
            if subset in subset_vid:
                if 'unlabel' not in subset_vid:
                    #breakpoint()
                    for j in range(len(labels)):
                        tmp_info = labels[j]
                        clip_factor = self.temporal_scale / ( corr_sec * (self.num_frame+1) )
                        action_start = tmp_info['segment'][0]*clip_factor
                        snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                        action_end = tmp_info['segment'][1]*clip_factor
                        snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                        gt_label = tmp_info["label"]
                        if action_end - action_start > 1:
                            label_list.append([snip_start,snip_end,gt_label])    
            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}

        return temporal_dict

    def getVideoMask(self,video_annos,clip_length=100):

        self.video_mask = {}
        idx_list = self.getAnnotation(self.subset,video_annos)
        self.anno_final = idx_list
        self.anno_final_idx = list(idx_list.keys())

        print('Loading '+self.subset+' Video Information ...')

        #breakpoint()

        for idx in tqdm.tqdm(list(video_annos.keys()),ncols=0):
            if os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")) and idx in list(idx_list.keys()):
                cur_annos = idx_list[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    mask_label_idx = self.class_to_idx[cur_annos[l_id][2]]
                    mask_list.append([mask_start,mask_end,mask_label_idx])
                self.video_mask[idx] = mask_list

        # print("Total vid", len(self.video_mask.keys()))
        return self.video_mask
             

    def loadFeature(self, idx):

        #feat_path = os.path.join(self.feature_path,self.subset)
        ## feat = np.load(os.path.joina(feat_path, idx+".npy"))
        ## print(idx)
        #feat = pd.read_csv(os.path.join(self.feature_path, idx+".csv"))
        #feat = feat.values[:, :]
        feat = np.load(os.path.join(self.feature_path, idx+".npy"))

        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data_ = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data_big = F.interpolate(video_data.unsqueeze(0), size=200, mode='linear',align_corners=False)[0,...]
        video_data_small = F.interpolate(video_data.unsqueeze(0), size=50, mode='linear',align_corners=False)[0,...]

        return video_data_, video_data_big, video_data_small


    def getVideoData(self,index):

        mask_idx = self.subset_mask_list[index]
        mask_data , mask_data_big, mask_data_small = self.loadFeature(mask_idx)
        mask_label = self.video_mask[mask_idx]

        bbox = np.array(mask_label)
        start_id = bbox[:,0]
        end_id = bbox[:,1]
        label_id = bbox[:,2]

        cls_mask = np.zeros([self.num_classes+1, self.temporal_scale]) ## dim : 201x100
        temporary_mask = np.zeros([self.temporal_scale])
        action_mask = np.zeros([self.temporal_scale,self.temporal_scale]) ## dim : 100 x 100
        cas_mask = np.zeros([self.num_classes])
    
        start_indexes = []
        end_indexes = []
        tuple_list =[]

        for idx in range(len(start_id)):
          lbl_id = label_id[idx]
          start_indexes.append(start_id[idx]+1)
          end_indexes.append(end_id[idx]-1)
          tuple_list.append([start_id[idx]+1, end_id[idx]-1,lbl_id])
        temp_mask_cls = np.zeros([self.temporal_scale])

        for idx in range(len(start_id)):
            temp_mask_cls[tuple_list[idx][0]:tuple_list[idx][1]]=1
            lbl_idx = int(tuple_list[idx][2])

            cls_mask[lbl_idx,:]= temp_mask_cls
            
        for idx in range(len(start_id)):
          temporary_mask[tuple_list[idx][0]:tuple_list[idx][1]] = 1
 
        background_mask = 1 - temporary_mask

        v_label = np.zeros([1])

        new_mask = np.zeros([self.temporal_scale])
        
        for p in range(self.temporal_scale):
            new_mask[p] = -1 

        cls_mask[self.num_classes,:] = background_mask

        filter_lab = list(set(label_id))

        for j in range(len(filter_lab)):
            label_idx = filter_lab[j]
            cas_mask[label_idx] = 1

        for idx in range(len(start_indexes)):
          len_gt = int(end_indexes[idx] - start_indexes[idx])

          mod_start = tuple_list[idx][0]
          mod_end = tuple_list[idx][1]
          new_lab = tuple_list[idx][2]

          new_mask[mod_start:mod_end] = new_lab


        for p in range(self.temporal_scale):
            if new_mask[p] == -1:
                new_mask[p] = self.num_classes

        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)

        for idx in range(len(start_indexes)):
            len_gt = int(end_indexes[idx] - start_indexes[idx])
            mod_start = tuple_list[idx][0]
            mod_end = tuple_list[idx][1]
            action_mask[mod_start:(mod_end), mod_start:(mod_end)] = 1

        global_mask_branch = torch.Tensor(action_mask)

        cas_mask = torch.Tensor(cas_mask)
        mask_top = torch.Tensor(cls_mask)
        v_label = torch.Tensor()
        b_mask = torch.Tensor(temporary_mask)

        
        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask


    def __getitem__(self, index):
        
        mask_data, top_branch, bottom_branch, mask_top, cas_mask, mask_data_big, mask_data_small, b_mask = self.getVideoData(index)

        if self.mode == "train":
            return mask_data,top_branch,bottom_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask
            # return mask_data,top_branch,bottom_branch,mask_top,cas_mask
        else:
            return index, mask_data , mask_data_big, mask_data_small, b_mask
            # return index, mask_data

    def __len__(self):
        return len(self.subset_mask_list)





class SPOTDatasetUnlabeled(data.Dataset):
    # NOTE: only used in data_loader_unlabeled
    def __init__(self, subset="train", mode="train"):
        # super().__init__()
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = config['training']['feature_path']
        self.unlabel_percent = config['dataset']['training']['unlabel_percent']
        self.video_info_path_unlabeled = config['dataset']['training']['video_info_path_unlabeled']
        self.video_info_path = os.path.join(self.video_info_path_unlabeled,"video_info_new_"+str(self.unlabel_percent)+".csv")
        self.video_anno_path = config['dataset']['training']['video_anno_path']
        self.num_frame = config['dataset']['training']['num_frame']
        self.num_classes = config['dataset']['num_classes']
        self.class_to_idx = activity_dict
        video_infos = self.get_video_info(self.video_info_path)
        self.info = video_infos
        video_annos = self.get_video_anno(video_infos, self.video_anno_path)
        self.subset_mask = self.getVideoMask(video_annos,self.temporal_scale)
        self.subset_mask_list = list(self.subset_mask.keys())
        #print("len(self.subset_mask_list)", len(self.subset_mask_list))
        


        
    def get_video_anno(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)
        # print(anno_database)
        video_dict = {}
        for video_name in video_infos.keys():
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]['subset']
            if self.subset in video_subset:
                video_info.update({'subset': video_subset})
                video_dict[video_name] = video_info


        return video_dict

    

    def get_video_info(self,video_info_path):

        df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        video_infos = {}
        self.v_list ={}
        for info in df_info:
            video_infos[info[0]] = {
                'duration': info[2],
                'subset': info[5]
            }
            
        return video_infos

    def getAnnotation(self, subset, anno):

        temporal_dict={}
        for idx in anno.keys():
            labels = anno[idx]['annotations']
            subset_vid  = anno[idx]['subset']
            num_frame = anno[idx]['feature_frame']
            vid_frame = anno[idx]['duration_frame']
            num_sec = anno[idx]['duration_second']
            corr_sec = float(num_frame) / vid_frame * num_sec
            label_list= []
            if subset in subset_vid:      
                for j in range(len(labels)):
                    tmp_info = labels[j]
                    clip_factor = self.temporal_scale / ( corr_sec * (self.num_frame+1) )
                    action_start = tmp_info['segment'][0]*clip_factor
                    snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                    action_end = tmp_info['segment'][1]*clip_factor
                    snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                    gt_label = tmp_info["label"]
                    if action_end - action_start > 1:
                        label_list.append([snip_start,snip_end,gt_label])    
            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}

        return temporal_dict

    def getVideoMask(self,video_annos,clip_length=100):

        self.video_mask = {}
        idx_list = self.getAnnotation(self.subset,video_annos)
        self.anno_final = idx_list 
        self.anno_final_idx = list(idx_list.keys())

        print('Loading '+self.subset+' Video Information ...') 
        for idx in tqdm.tqdm(list(video_annos.keys()),ncols=0):
            if os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")) and idx in list(idx_list.keys()):
                cur_annos = idx_list[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    mask_label_idx = self.class_to_idx[cur_annos[l_id][2]]
                    mask_list.append([mask_start,mask_end,mask_label_idx])
                self.video_mask[idx] = mask_list
        # print("Tot Videos", len(self.video_mask.keys()))
        return self.video_mask
                
    def loadFeature(self, idx):

        #feat_path = os.path.join(self.feature_path,self.subset)
        ## feat = np.load(os.path.join(feat_path, idx+".npy"))
        #feat = pd.read_csv(os.path.join(self.feature_path, idx+".csv"))
        #feat = feat.values[:, :]
        feat = np.load(os.path.join(self.feature_path, idx+".npy"))

        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data_ = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data_big = F.interpolate(video_data.unsqueeze(0), size=200, mode='linear',align_corners=False)[0,...]
        video_data_small = F.interpolate(video_data.unsqueeze(0), size=50, mode='linear',align_corners=False)[0,...]

        return video_data_, video_data_big, video_data_small


    def getVideoData(self,index):

        mask_idx = self.subset_mask_list[index]
        mask_data , mask_data_big, mask_data_small = self.loadFeature(mask_idx)
        mask_label = self.video_mask[mask_idx]

        bbox = np.array(mask_label)
        start_id = bbox[:,0]
        end_id = bbox[:,1]
        label_id = bbox[:,2]

        cls_mask = np.zeros([self.num_classes+1, self.temporal_scale]) ## dim : 201x100
        temporary_mask = np.zeros([self.temporal_scale])
        action_mask = np.zeros([self.temporal_scale,self.temporal_scale]) ## dim : 100 x 100
        cas_mask = np.zeros([self.num_classes])
    
        start_indexes = []
        end_indexes = []
        tuple_list = []

        for idx in range(len(start_id)):
          lbl_id = label_id[idx]
          start_indexes.append(start_id[idx]+1)
          end_indexes.append(end_id[idx]-1)
          tuple_list.append([start_id[idx]+1, end_id[idx]-1,lbl_id])
        temp_mask_cls = np.zeros([self.temporal_scale])

        for idx in range(len(start_id)):
            temp_mask_cls[tuple_list[idx][0]:tuple_list[idx][1]]=1
            lbl_idx = int(tuple_list[idx][2])

            cls_mask[lbl_idx,:]= temp_mask_cls
            
        for idx in range(len(start_id)):
          temporary_mask[tuple_list[idx][0]:tuple_list[idx][1]] = 1
 
        background_mask = 1 - temporary_mask

        v_label = np.zeros([1])

        new_mask = np.zeros([self.temporal_scale])
        
        for p in range(self.temporal_scale):
            new_mask[p] = -1 

        cls_mask[self.num_classes,:] = background_mask

        filter_lab = list(set(label_id))

        for j in range(len(filter_lab)):
            label_idx = filter_lab[j]
            cas_mask[label_idx] = 1

        for idx in range(len(start_indexes)):
          len_gt = int(end_indexes[idx] - start_indexes[idx])

          mod_start = tuple_list[idx][0]
          mod_end = tuple_list[idx][1]
          new_lab = tuple_list[idx][2]

          new_mask[mod_start:mod_end] = new_lab


        for p in range(self.temporal_scale):
            if new_mask[p] == -1:
                new_mask[p] = self.num_classes

        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)

        for idx in range(len(start_indexes)):
            len_gt = int(end_indexes[idx] - start_indexes[idx])
            mod_start = tuple_list[idx][0]
            mod_end = tuple_list[idx][1]
            action_mask[mod_start:(mod_end), mod_start:(mod_end)] = 1

        global_mask_branch = torch.Tensor(action_mask)

        cas_mask = torch.Tensor(cas_mask)
        mask_top = torch.Tensor(cls_mask)
        v_label = torch.Tensor()
        b_mask = torch.Tensor(temporary_mask)


        
        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask


    def __getitem__(self, index):
        
        mask_data, top_branch, bottom_branch, mask_top, cas_mask, mask_data_big, mask_data_small, b_mask = self.getVideoData(index)

        if self.mode == "train":
            return mask_data,top_branch,bottom_branch,mask_top,cas_mask,mask_data_big, mask_data_small
        else:
            return index, mask_data , mask_data_big, mask_data_small


    def __len__(self):
        return len(self.subset_mask_list)





class SPOTDatasetPretrain(data.Dataset):
    # NOTE: this class is never used
    def __init__(self, subset="train", mode="train"):
        # super().__init__()
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = config['training']['feature_path']
        self.unlabel_percent = config['dataset']['training']['unlabel_percent']
        self.video_info_path_unlabeled = config['dataset']['training']['video_info_path_unlabeled']
        # self.video_info_path = os.path.join(self.video_info_path_unlabeled,"video_info_new_"+str(self.unlabel_percent)+".csv")
        self.video_info_path = config['dataset']['training']['video_info_path']
        self.video_anno_path = config['dataset']['training']['video_anno_path']
        self.num_frame = config['dataset']['training']['num_frame']
        self.num_classes = config['dataset']['num_classes']
        self.class_to_idx = activity_dict
        video_infos = self.get_video_info(self.video_info_path)
        self.info = video_infos
        video_annos = self.get_video_anno(video_infos, self.video_anno_path)
        self.subset_mask = self.getVideoMask(video_annos,self.temporal_scale)
        self.subset_mask_list = list(self.subset_mask.keys())
        


        
    def get_video_anno(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)
        # print(anno_database)
        video_dict = {}
        for video_name in video_infos.keys():
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]['subset']
            if self.subset in video_subset:
                video_info.update({'subset': video_subset})
                video_dict[video_name] = video_info


        return video_dict

    

    def get_video_info(self,video_info_path):

        df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        video_infos = {}
        self.v_list ={}
        for info in df_info:
            video_infos[info[0]] = {
                'duration': info[2],
                'subset': info[5]
            }
            
        return video_infos

    def getAnnotation(self, subset, anno):

        temporal_dict={}
        for idx in anno.keys():
            labels = anno[idx]['annotations']
            subset_vid  = anno[idx]['subset']
            num_frame = anno[idx]['feature_frame']
            vid_frame = anno[idx]['duration_frame']
            num_sec = anno[idx]['duration_second']
            corr_sec = float(num_frame) / vid_frame * num_sec
            label_list= []
            if subset in subset_vid:      
                for j in range(len(labels)):
                    tmp_info = labels[j]
                    clip_factor = self.temporal_scale / ( corr_sec * (self.num_frame+1) )
                    action_start = tmp_info['segment'][0]*clip_factor
                    snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                    action_end = tmp_info['segment'][1]*clip_factor
                    snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                    gt_label = tmp_info["label"]
                    if action_end - action_start > 1:
                        label_list.append([snip_start,snip_end,gt_label])    
            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}

        return temporal_dict

    def getVideoMask(self,video_annos,clip_length=100):

        self.video_mask = {}
        idx_list = self.getAnnotation(self.subset,video_annos)
        self.anno_final = idx_list
        self.anno_final_idx = list(idx_list.keys())

        print('Loading '+self.subset+' Video Information ...') 
        for idx in tqdm.tqdm(list(video_annos.keys()),ncols=0):
            if os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")) and idx in list(idx_list.keys()):
                cur_annos = idx_list[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    mask_label_idx = self.class_to_idx[cur_annos[l_id][2]]
                    mask_list.append([mask_start,mask_end,mask_label_idx])
                self.video_mask[idx] = mask_list
        # print("Tot Videos", len(self.video_mask.keys()))
        return self.video_mask
                
    def loadFeature(self, idx):

        #feat_path = os.path.join(self.feature_path,self.subset)
        ## feat = np.load(os.path.join(feat_path, idx+".npy"))
        #feat = pd.read_csv(os.path.join(self.feature_path, idx+".csv"))
        #feat = feat.values[:, :]
        feat = np.load(os.path.join(self.feature_path, idx+".npy"))
        
        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data_ = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data_big = F.interpolate(video_data.unsqueeze(0), size=200, mode='linear',align_corners=False)[0,...]
        video_data_small = F.interpolate(video_data.unsqueeze(0), size=50, mode='linear',align_corners=False)[0,...]
       
        return video_data_, video_data_big, video_data_small


    def getVideoData(self,index):

        mask_idx = self.subset_mask_list[index]
        mask_data, mask_data_big, mask_data_small = self.loadFeature(mask_idx)
        mask_label = self.video_mask[mask_idx]

        bbox = np.array(mask_label)

        start_id = np.random.randint(0,self.temporal_scale-1,size=1)
        end_id = np.random.randint(0,self.temporal_scale-1,size=1)
        start_id = bbox[:,0]
        end_id = bbox[:,1]
        label_id = bbox[:,2]

        cls_mask = np.zeros([self.num_classes+1, self.temporal_scale]) ## dim : 201x100
        temporary_mask = np.zeros([self.temporal_scale])
        action_mask = np.zeros([self.temporal_scale,self.temporal_scale]) ## dim : 100 x 100
        cas_mask = np.zeros([self.num_classes])
        # for i in range(self.num_classes):
        #     cas_mask[i] = 200
        
        start_indexes = []
        end_indexes = []
        tuple_list = []

        for idx in range(len(start_id)):
          lbl_id = label_id[idx]
          start_indexes.append(start_id[idx]+1)
          end_indexes.append(end_id[idx]-1)
          tuple_list.append([start_id[idx]+1, end_id[idx]-1,lbl_id])
        temp_mask_cls = np.zeros([self.temporal_scale])

        for idx in range(len(start_id)):
            temp_mask_cls[tuple_list[idx][0]:tuple_list[idx][1]]=1
            lbl_idx = int(tuple_list[idx][2])

            cls_mask[lbl_idx,:]= temp_mask_cls
            
        for idx in range(len(start_id)):
          temporary_mask[tuple_list[idx][0]:tuple_list[idx][1]] = 1
 
        background_mask = 1 - temporary_mask

        v_label = np.zeros([1])

        new_mask = np.zeros([self.temporal_scale])
        
        for p in range(self.temporal_scale):
            new_mask[p] = -1 

        cls_mask[self.num_classes,:] = background_mask

        filter_lab = list(set(label_id))

        for j in range(len(filter_lab)):
            label_idx = filter_lab[j]
            cas_mask[label_idx] = 1

        for idx in range(len(start_indexes)):
          len_gt = int(end_indexes[idx] - start_indexes[idx])

          mod_start = tuple_list[idx][0]
          mod_end = tuple_list[idx][1]
          new_lab = tuple_list[idx][2]

          new_mask[mod_start:mod_end] = new_lab


        for p in range(self.temporal_scale):
            if new_mask[p] == -1:
                new_mask[p] = self.num_classes

        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)

        for idx in range(len(start_indexes)):
            len_gt = int(end_indexes[idx] - start_indexes[idx])
            mod_start = tuple_list[idx][0]
            mod_end = tuple_list[idx][1]
            action_mask[mod_start:(mod_end), mod_start:(mod_end)] = 1

        global_mask_branch = torch.Tensor(action_mask)

        cas_mask = torch.Tensor(cas_mask)
        mask_top = torch.Tensor(cls_mask)
        v_label = torch.Tensor()
        b_mask = torch.Tensor(temporary_mask)


        
        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask


    def __getitem__(self, index):
        
        mask_data, top_branch, bottom_branch, mask_top, cas_mask, mask_data_big, mask_data_small, b_mask = self.getVideoData(index)

        if self.mode == "train":
            return mask_data,top_branch,bottom_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask
        else:
            return index, mask_data , mask_data_big, mask_data_small, b_mask


    def __len__(self):
        return len(self.subset_mask_list)


if __name__ == '__main__':
    from gsm_lib import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(GSMDataset(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a,b,c,d,e in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break
