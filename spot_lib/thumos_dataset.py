"""
Status (July 5th)
- This dataloader does work. However, it should probably be made more like the SPOT_dataloader.
"""
# NOTE: for THUMOS, I might want to increase the temporal_scale since I noticed that if we use temporal_scale = 100, the clips are short. However, this might have effects on the runtimes, and we don't really care too much about the high IoU.


import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from spot_lib import videotransforms # Will likely not be used 
import random
import math

import yaml
import sys
from utils.arguments import handle_args, modify_config
from torch.functional import F
import copy
import pdb
from configs.dataset_class import thumos_dict

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))

def get_class_index_map(class_info_path=config['dataset']['training']['class_info']):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class


def get_video_info(video_info_path):
    # todo: see below
    """corresponds to self.get_video_info() and get_video_anno() in spotdataset, except for the "segment" and "label" stuff
    spotdataset.get_video_info() produces dictionary of the form {'v_---9cprckou': {'duration': 14.07, 'subset': 'training_unlabel'}}
    todo: inclusion of duration and unlabel, like in spotdataset's version (ie. the 'subset' key) for semi-supervised learning

    corresponds to spot_dataloader.get_video_anno (and getvideomask)
    in contrast, spotdataset.get_video_anno produces {video_name: {duration_second:float, duration_frame:int, annotations: [{segment: [start, end], label:'string label'}], feature_frame:int, subset:training_unlabel}
    - based on this, todo?: include the details about the segment and the unlabel vs. label for semi-supervised?

    args:
        video_info_path (string): path to the dir with the numpy files

    returns:
        dictionary of the form {'video_validation_0000051': {'fps':30.0, 'sample_fps':10.0, 'count':5091.0, 'sample_count':1697, 'duration':169.7, 'subset':'training'}}
            - details of the video, and a version of the video that was downsampled by a factor of 3 (same duration, lower frame rate)
    
    does not differ between regular and unlabeled
    """
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4],
            'duration': info[3] / info[1], # sample and non-sample have same length in seconds
            'subset': 'training' if 'val' in video_info_path else 'testing'} # HACK: i will need to base this off of a video_info_new_*.csv-type file that has the training set (incl. unlabel); this should be in (training, training_unlabel, validation, testing)
    return video_infos


def get_video_anno(video_infos,
                   video_anno_path, temporal_scale=config['model']['temporal_scale']):
    # TODO see comment
    """ corresponds to getVideoMask (the "segment" and "label" stuff missing from THUMOS_Dataset's version of SPOTDataset's get_video_info/get_video_anno)
    This is probably the function that is most similar to its counterpart in SPOTDataset, although start/end should still be put in terms of temporal scale.

    Args:
        video_infos (dictionary): the get_video_info dictionary; contains fps/count and sample_fps/sample_count
        video_anno_path (string): [location of *_Annotation_ours.csv, a .csv where each row represents an annotation/action. Each row has the video, the action (indexed as a string and numerically*), and the start/end of the frame in both seconds and frame (non-downsampled)
        *the index in *_Annotation_ours.csv is not what is used here, since *_Annotation_ours indexes based on the 200 classes in THUMOS, but since we only use 20 classes the index must be in range(20) or else PyTorch complains.]
        temporal_scale (int): the temporal scale of the video, so that the scale can be fixed.

    Returns:
        dictionary of form {'video_validation_0000051':  ...} [[39, 44, 3], [50, 53, 3], [82, 87, 3]], where in each [mask_start, mask_end, class_idx]:
        - mask_start is the start frame of the annotation after adjusting for the sample_count (the sample_count being the number of frames in the video after downsampling), where times are represented as a proportion of the video (which has been put into temporal_scale)
        - mask_end: start_gt but for the last frame of the annotation
        - class_idx: the class of this annotation, indexed numerically]

    Does not differ between regular and unlabeled
    """
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map() 
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]['count']
        sample_count = video_infos[video_name]['sample_count']
        sample_fps = video_infos[video_name]['sample_fps']
        duration = video_infos[video_name]['duration']
        ratio = sample_count * 1.0 / count # the downsampling ratio.
 
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        start_gt_time = (start_gt / sample_fps)
        end_gt_time = (end_gt / sample_fps)
        class_idx = originidx_to_idx[originidx]

        # maintaining temporal scale
        clip_factor = temporal_scale / (duration * (sample_count+1))
        action_start = start_gt_time * clip_factor # scale the start time (seconds) by clip factor
        action_end = end_gt_time * clip_factor
        snip_start = max(min(1, start_gt_time / duration), 0) # puts the annotation in range (0, 1); ie. what proportion of the video has gone by before this annotation starts?
        snip_end = max(min(1, end_gt_time / duration), 0)
        mask_start = int(math.floor(temporal_scale * snip_start)) # puts annotation in range(0, temporal_scale); we have resized the video to be temporal_scale seconds long
        mask_end = int(math.floor(temporal_scale * snip_end))
 
        if mask_end - mask_start > 1: # NOTE: we only use on large-enough actions for the temporal scale
            if video_annos.get(video_name) is None:
                video_annos[video_name] = [[mask_start, mask_end, class_idx]]
            else:
                video_annos[video_name].append([mask_start, mask_end, class_idx])
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


def split_videos(video_infos,
                 video_annos,
                 clip_length=config['dataset']['training']['clip_length'],
                 stride=config['dataset']['training']['clip_stride']):
    # video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    # video_annos = get_video_anno(video_infos,
    #                              config['dataset']['training']['video_anno_path'])
    training_list = []
    min_anno_dict = {}
    for video_name in video_annos.keys():
        min_anno = clip_length
        sample_count = video_infos[video_name]['sample_count']
        annos = video_annos[video_name]
        if sample_count <= clip_length:
            offsetlist = [0]
            min_anno_len = min([x[1] - x[0] for x in annos])
            if min_anno_len < min_anno:
                min_anno = min_anno_len
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]
        for offset in offsetlist:
            left, right = offset + 1, offset + clip_length
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa >= 0.5:
                    cur_annos.append([max(anno[0] - offset, 1),
                                      min(anno[1] - offset, clip_length),
                                      anno[2]])
            if len(cur_annos) > 0:
                min_anno_len = min([x[1] - x[0] for x in cur_annos])
                if min_anno_len < min_anno:
                    min_anno = min_anno_len
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                for anno in cur_annos:
                    s, e, id = anno
                    d = max((e - s) / 10.0, 2.0)
                    start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
                    start[start_s: start_e] = 1
                    end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
                    end[end_s: end_e] = 1
                training_list.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': cur_annos,
                    'start': start,
                    'end': end
                })
        min_anno_dict[video_name] = math.ceil(min_anno)
    return training_list, min_anno_dict


def load_video_data(video_infos, npy_data_path, temporal_scale=config['model']['temporal_scale']):
    # TODO
    """ A dictionary that only contains the video name  and their features. TODO: Would most closely resemble SPOTDataset.loadFeature, although SPOTDataset's function loads one video at a time, and loads different versions of the feature data (big and small). Could this method be worse for memory requirement, especially with multiple sizes of the data? For example, SPOTDataloader does not keep the entire dataset in a list, they have loadFeature to np.load for a single video, SPOTDataset.getVideoData calls loadFeature to get extra versions of the data (different branches and sizes) but only for one video at a time, and the must-implement __getitem__ for a PyTorch dataloader (which again only wants one video) simply calls getVideoData. 
    THUMOS_Dataset appears to do work in the __getitem__ method; since I want to match what the SPOT model needs, I will need this load_video_data to load the different versions of the feature data, and the __getitem__ method will need to do work on different versions of the feature data. The idea is to make sure that the work of AFSD and SPOT is united; AFSD authors would have determined what to do for this particular THUMOS dataset, but SPOT also has its own needs (different versions of the feature data). I must take from both.

    Args
        video_infos (dictionary): form {video_name : {'fps': float, 'sample_fps': float, 'count': float, 'sample_count': float}}
            fps/count are the fps and frame number for the original video; fps ~= 30
            sample_fps/sample_count are fps/count divided by 3; the same video (same length), but downsampled to lower frame rate.
            created from get_video_info() 
        npy_data_path (string): path to the location where the .npy features are stored. (see 'video_info_path') in the config 

    Returns:
        dictionary of form {video_name : data ...}, where:
            video_name (string): the name of the video. eg. 'video_validation_0000051'
            data (np.array): the features, shape (T, D)
                T: the number of clips. T = num_of_frames_in_video / (16 + 3) (each clip is len 16, step_size 3)
                D: the dimension of the clip's feature vector. eg. 2048
    """
    data_dict = {}
    print('loading video frame data ...')
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        #data = np.transpose(data, [3, 0, 1, 2])
        feat_tensor = torch.Tensor(data)
        video_data = torch.transpose(feat_tensor, 0, 1) # torch.Size([T, D]) -> torch.Size([D, T])
        video_data_ = F.interpolate(video_data.unsqueeze(0), size=temporal_scale, mode='linear',align_corners=False)[0,...] # torch.Size([T, D]) -> torch.Size([D, temporal_scale]) (puts the video in terms of temporal_scale many clips)
        video_data_big = F.interpolate(video_data.unsqueeze(0), size=2*temporal_scale, mode='linear',align_corners=False)[0,...]
        video_data_small = F.interpolate(video_data.unsqueeze(0), size=temporal_scale//2, mode='linear',align_corners=False)[0,...]
        data_dict[video_name] = (video_data_, video_data_big, video_data_small) # [torch.Size([2048, 100]), torch.Size([2048, 200]), torch.Size([2048, 50])]
    return data_dict


class THUMOS_Dataset(Dataset):
    def __init__(self, 
                 #data_dict,
                 #video_infos,
                 #video_annos,
                 rgb_norm=True,
                 training=True,
                 origin_ratio=0.5,
                 subset='train',
                 mode='train',
                 unlabeled=False,
                 labeled=False):

        self.training = training
        self.unlabeled = unlabeled
        self.labeled = labeled
        self.num_classes = config['dataset']['num_classes']
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.class_to_idx = thumos_dict

        self.split = {'train': 'training', 'validation': 'testing', 'testing': 'testing'}[self.subset]

        self.video_info_path_unlabeled = config['dataset']['training']['video_info_path_unlabeled'] # NOTE: unlabel_percent only matters for training. We always use the full dataset for testing.
        self.unlabel_percent = config['dataset']['training']['unlabel_percent']
        self.video_info_path = os.path.join(self.video_info_path_unlabeled, "val_video_info_"+str(self.unlabel_percent)+".csv") if self.training else config['dataset'][self.split]['video_info_path']
        self.video_anno_path = config['dataset'][self.split]['video_anno_path']
        self.npy_data_path = config[self.split]['feature_path'] 
 
        self.video_infos = self.get_video_info()
        self.subset_mask_list = list(self.video_infos.keys())
        self.video_annos = self.get_video_anno()
        self.data_dict = self.load_video_data()
        self.clip_length = config['dataset'][self.split]['clip_length']
        self.crop_size = config['dataset'][self.split]['crop_size']
        self.stride = config['dataset'][self.split]['clip_stride']
        self.random_crop = videotransforms.RandomCrop(self.crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(self.crop_size)
        self.rgb_norm = rgb_norm 

        self.origin_ratio = origin_ratio

        self.training_list, self.th = split_videos(
            self.video_infos,
            self.video_annos,
            self.clip_length,
            self.stride
        )
        # np.random.shuffle(self.training_list)


    def get_video_info(self):
        # TODO: see below
        """Corresponds to self.get_video_info() and get_video_anno() in SPOTDataset, except for the "segment" and "label" stuff
        SPOTDataset.get_video_info() produces dictionary of the form {'v_---9CpRcKoU': {'duration': 14.07, 'subset': 'training_unlabel'}}
        TODO: inclusion of duration and unlabel, like in SPOTDataset's version (ie. the 'subset' key) for semi-supervised learning

        corresponds to SPOT_dataloader.get_video_anno (and getVideoMask)
        in contrast, SPOTDataset.get_video_anno produces {video_name: {duration_second:float, duration_frame:int, annotations: [{segment: [start, end], label:'string label'}], feature_frame:int, subset:training_unlabel}
        - Based on this, TODO?: include the details about the segment and the unlabel vs. label for semi-supervised?

        Args:
            video_info_path (string): Path to the dir with the numpy files

        Returns:
            dictionary of the form {'video_validation_0000051': {'fps':30.0, 'sample_fps':10.0, 'count':5091.0, 'sample_count':1697, 'duration':169.7, 'subset':'training'}}
                - details of the video, and a version of the video that was downsampled by a factor of 3 (same duration, lower frame rate)
        
        Does not differ between regular and unlabeled
        """
        if False: #self.unlabeled:
 
            video_infos = {}

            def foo(r):
                #global video_infos
                video_infos[r['video']] = {'fps':r['fps'], 'sample_fps':r['sample_fps'], 'count':r['count'], 'sample_count':r['sample_count'], 'duration':r['count']/r['fps'], 'subset':r['subset']}
     
            df_info = pd.read_csv(self.video_info_path)
            df_info.apply(foo, axis=1)

            return video_infos

        else:

            df_info = pd.DataFrame(pd.read_csv(self.video_info_path)).values[:]
            video_infos = {}
            for info in df_info:
                if self.subset in info[5] and ((self.subset == 'testing') or (self.labeled and 'unlabel' not in info[5]) or (self.unlabeled and 'unlabel' in info[5])):
                    video_infos[info[0]] = {
                        'fps': info[1],
                        'sample_fps': info[2],
                        'count': info[3],
                        'sample_count': info[4],
                        'duration': info[3] / info[1], # sample and non-sample have same length in seconds
                        'subset': 'training' if 'val' in self.video_info_path else 'testing'} # HACK: I will need to base this off of a video_info_new_*.csv-type file that has the training set (incl. unlabel); this should be in (training, training_unlabel, validation, testing)
            return video_infos


    def get_video_anno(self):
        # TODO see comment
        """ corresponds to getVideoMask (the "segment" and "label" stuff missing from THUMOS_Dataset's version of SPOTDataset's get_video_info/get_video_anno)
        This is probably the function that is most similar to its counterpart in SPOTDataset, although start/end should still be put in terms of temporal scale.

        Args:
            video_infos (dictionary): the get_video_info dictionary; contains fps/count and sample_fps/sample_count
            video_anno_path (string): [location of *_Annotation_ours.csv, a .csv where each row represents an annotation/action. Each row has the video, the action (indexed as a string and numerically*), and the start/end of the frame in both seconds and frame (non-downsampled)
            *the index in *_Annotation_ours.csv is not what is used here, since *_Annotation_ours indexes based on the 200 classes in THUMOS, but since we only use 20 classes the index must be in range(20) or else PyTorch complains.]
            temporal_scale (int): the temporal scale of the video, so that the scale can be fixed.

        Returns:
            dictionary of form {'video_validation_0000051':  ...} [[39, 44, 3], [50, 53, 3], [82, 87, 3]], where in each [mask_start, mask_end, class_idx]:
            - mask_start is the start frame of the annotation after adjusting for the sample_count (the sample_count being the number of frames in the video after downsampling), where times are represented as a proportion of the video (which has been put into temporal_scale)
            - mask_end: start_gt but for the last frame of the annotation
            - class_idx: the class of this annotation, indexed numerically]

        Does not differ between regular and unlabeled
        """
        # if self.unlabeled:
            # breakpoint()

        #breakpoint()
        df_anno = pd.DataFrame(pd.read_csv(self.video_anno_path))#.values[:]
        df_anno = df_anno[df_anno['video'].isin(self.video_infos.keys())]
        df_anno = df_anno.values[:]
        originidx_to_idx, idx_to_class = get_class_index_map() # NOTE: should "Ambiguous" be used as a class? No, use val_Annotation_ours.csv
        video_annos = {}
        for anno in df_anno:

            video_name = anno[0]
            originidx = anno[2]
            start_frame = anno[-2]
            end_frame = anno[-1]
            count = self.video_infos[video_name]['count']
            sample_count = self.video_infos[video_name]['sample_count']
            sample_fps = self.video_infos[video_name]['sample_fps']
            duration = self.video_infos[video_name]['duration']
            ratio = sample_count * 1.0 / count # the downsampling ratio.
     
            start_gt = start_frame * ratio
            end_gt = end_frame * ratio
            start_gt_time = (start_gt / sample_fps)
            end_gt_time = (end_gt / sample_fps)
            class_idx = self.class_to_idx[idx_to_class[originidx_to_idx[originidx]]] #originidx_to_idx[originidx]

            # maintaining temporal scale
            clip_factor = self.temporal_scale / (duration * (sample_count+1))
            action_start = start_gt_time * clip_factor # scale the start time (seconds) by clip factor # TODO: might want to look at using action_start and action_end
            action_end = end_gt_time * clip_factor
            snip_start = max(min(1, start_gt_time / duration), 0) # puts the annotation in range (0, 1); ie. what proportion of the video has gone by before this annotation starts?
            snip_end = max(min(1, end_gt_time / duration), 0)
            mask_start = int(math.floor(self.temporal_scale * snip_start)) # puts annotation in range(0, temporal_scale); we have resized the video to be temporal_scale seconds long
            mask_end = int(math.ceil(self.temporal_scale * snip_end)) # NOTE: I've changed this to ceil.
     
            if mask_end - mask_start > 1: # we focus on large-enough actions
                if video_annos.get(video_name) is None:
                    video_annos[video_name] = [[mask_start, mask_end, class_idx]]
                else:
                    video_annos[video_name].append([mask_start, mask_end, class_idx])
        return video_annos


    def load_video_data(self):
        # TODO
        """ A dictionary that only contains the video name  and their features. TODO: Would most closely resemble SPOTDataset.loadFeature, although SPOTDataset's function loads one video at a time, and loads different versions of the feature data (big and small). Could this method be worse for memory requirement, especially with multiple sizes of the data? For example, SPOTDataloader does not keep the entire dataset in a list, they have loadFeature to np.load for a single video, SPOTDataset.getVideoData calls loadFeature to get extra versions of the data (different branches and sizes) but only for one video at a time, and the must-implement __getitem__ for a PyTorch dataloader (which again only wants one video) simply calls getVideoData. 
        THUMOS_Dataset appears to do work in the __getitem__ method; since I want to match what the SPOT model needs, I will need this load_video_data to load the different versions of the feature data, and the __getitem__ method will need to do work on different versions of the feature data. The idea is to make sure that the work of AFSD and SPOT is united; AFSD authors would have determined what to do for this particular THUMOS dataset, but SPOT also has its own needs (different versions of the feature data). I must take from both.

        Args
            video_infos (dictionary): form {video_name : {'fps': float, 'sample_fps': float, 'count': float, 'sample_count': float}}
                fps/count are the fps and frame number for the original video; fps ~= 30
                sample_fps/sample_count are fps/count divided by 3; the same video (same length), but downsampled to lower frame rate.
                created from get_video_info() 
            npy_data_path (string): path to the location where the .npy features are stored. (see 'video_info_path') in the config 

        Returns:
            dictionary of form {video_name : data ...}, where:
                video_name (string): the name of the video. eg. 'video_validation_0000051'
                data (np.array): the features, shape (T, D)
                    T: the number of clips. T = num_of_frames_in_video / (16 + 3) (each clip is len 16, step_size 3)
                    D: the dimension of the clip's feature vector. eg. 2048
        """
        data_dict = {}
        print('loading video frame data ...')
        for video_name in tqdm.tqdm(list(self.video_infos.keys()), ncols=0):
            data = np.load(os.path.join(self.npy_data_path, video_name + '.npy'))
            #data = np.transpose(data, [3, 0, 1, 2])
            feat_tensor = torch.Tensor(data)
            video_data = torch.transpose(feat_tensor, 0, 1) # torch.Size([T, D]) -> torch.Size([D, T])
            video_data_ = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...] # torch.Size([T, D]) -> torch.Size([D, temporal_scale]) (puts the video in terms of temporal_scale many clips)
            video_data_big = F.interpolate(video_data.unsqueeze(0), size=2*self.temporal_scale, mode='linear',align_corners=False)[0,...]
            video_data_small = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale//2, mode='linear',align_corners=False)[0,...]
            data_dict[video_name] = (video_data_, video_data_big, video_data_small) # [torch.Size([2048, 100]), torch.Size([2048, 200]), torch.Size([2048, 50])]
        return data_dict


    def __len__(self):
        return len(self.video_annos) #len(self.training_list)


    def get_bg(self, annos, min_action):
        annos = [[anno[0], anno[1]] for anno in annos]
        times = []
        for anno in annos:
            times.extend(anno)
        times.extend([0, self.clip_length - 1])
        times.sort()
        regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
        regions = list(filter(
            lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action, regions))
        # regions = list(filter(lambda x:x not in annos, regions))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]


    def augment_(self, input, annos, th):
        '''
        input: (c, t, h, w)
        target: (N, 3)
        '''
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
            # gt = random.choice(annos)
        except IndexError:
            return input, annos, False
        gt_len = gt[1] - gt[0]
        region = range(math.floor(th), math.ceil(gt_len - th))
        t = random.choice(region) + math.ceil(gt[0])
        l_len = math.ceil(t - gt[0])
        r_len = math.ceil(gt[1] - t)
        try:
            bg = self.get_bg(annos, th)
        except IndexError:
            return input, annos, False
        start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
        end_idx = start_idx + th

        new_input = input.clone()
        # annos.remove(gt)
        if gt[1] < start_idx:
            new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
            new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]

            new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
            # new_annos = [[t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t+th-math.ceil(th/5), t+th+math.ceil(th/5)],
            #            [t+1, t+th-1]]

        else:
            new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
            new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]

            new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
            # new_annos = [[t-th-math.ceil(th/5), t-th+math.ceil(th/5)],
            #            [t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t-th+1, t-1]]

        return new_input, new_annos, True


    def augment(self, input, annos, th, max_iter=10):
        flag = True
        i = 0
        while flag and i < max_iter:
            new_input, new_annos, flag = self.augment_(input, annos, th)
            i += 1
        return new_input, new_annos, flag


    def getVideoData(self, index):
        """performs the role of SPOTDataset.getVideoData, ie. gets all of the inputs to the various branches.

        Args:
            index ([TODO:parameter]): [TODO:description]

        Returns:
            [TODO:return]
        """
        #print("index =", index)

        #breakpoint()

        mask_idx = list(self.video_annos.keys())[index] #self.training_list[index]['video_name']
        video_infos = self.video_infos[mask_idx]

        #print("mask_idx", mask_idx)
        #print("video_infos", video_infos)

        mask_label = self.video_annos[mask_idx]
        mask_data, mask_data_big, mask_data_small = self.data_dict[mask_idx]

        bbox = np.array(mask_label)
        start_id = bbox[:,0]
        end_id = bbox[:,1]
        label_id = bbox[:,2]

        #print(bbox)
    
        cls_mask = np.zeros([self.num_classes+1, self.temporal_scale]) ## dim : 21x100 (ie. one-hot encoding for each clip)
        temporary_mask = np.zeros([self.temporal_scale]) # dim: 100 (ie. one value per clip)
        action_mask = np.zeros([self.temporal_scale,self.temporal_scale]) ## dim : 100 x 100
        cas_mask = np.zeros([self.num_classes]) # dim: 20 (ie. one value per class)
    
        start_indexes = []
        end_indexes = []
        tuple_list =[]

        for idx in range(len(start_id)): # one iteration per annotation for this video
          lbl_id = label_id[idx] # get the class
          start_indexes.append(start_id[idx]+1)
          end_indexes.append(end_id[idx]-1)
          tuple_list.append([start_id[idx]+1, end_id[idx]-1, lbl_id]) # NOTE: reduce the segment's size by 2, but I could do away with this. 

        temp_mask_cls = np.zeros([self.temporal_scale])

        for idx in range(len(start_id)):
            temp_mask_cls[tuple_list[idx][0]:tuple_list[idx][1]]=1 # create a mask of annotation, where the clips contained in the annotation get a 1, otherwise 0
            lbl_idx = int(tuple_list[idx][2])

            cls_mask[lbl_idx,:]= temp_mask_cls # one-hot encode to track when an action occurs, and (via one-hot) track the action label. 

        temporary_mask = copy.deepcopy(temp_mask_cls)
 
        background_mask = 1 - temporary_mask # a class mask, but where the class is "background"
       
        new_mask = np.zeros([self.temporal_scale]) + self.num_classes

        cls_mask[self.num_classes,:] = background_mask # the background class has been assigned the number self.num_classes = 20; so, cls_mask[self.num_classes] shows the 100 locations in time, and has a 1 iff it is a background clip.

        filter_lab = list(set(label_id))

        for j in range(len(filter_lab)):
            label_idx = filter_lab[j]
            cas_mask[label_idx] = 1 # cas_mask lists the classes that are present in the video in a kind of "multi-hot" setup.

        len_gt = np.array(end_indexes) - np.array(start_indexes)


        for idx in range(len(start_indexes)):
          mod_start = tuple_list[idx][0]
          mod_end = tuple_list[idx][1]
          new_lab = tuple_list[idx][2]

          new_mask[mod_start:mod_end] = new_lab # for each clip, the class index (-1 for background)

        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)

        for idx in range(len(start_indexes)): # can we not combine this with the loop on line 254? 
            mod_start = tuple_list[idx][0]
            mod_end = tuple_list[idx][1]
            action_mask[mod_start:(mod_end), mod_start:(mod_end)] = 1 # a kind of 2D version of the one-hot temp_mask_cls

        global_mask_branch = torch.Tensor(action_mask) # temporary_mask but expanded to 2D
        cas_mask = torch.Tensor(cas_mask) # multi-hot encoding of the classes present in the video
        mask_top = torch.Tensor(cls_mask) # for each clip, a one-hot encoding of the class
        b_mask = torch.Tensor(temporary_mask) # aka temp_mask_cls; 1 if clip is part of action, 0 otherwise

        #print(classifier_branch)

        #breakpoint()
        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask

    """
    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_data = self.data_dict[sample_info['video_name']]
        offset = sample_info['offset']
        annos = sample_info['annos']
        th = self.th[sample_info['video_name']]

        input_data = video_data[:, offset: offset + self.clip_length]
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            # padding t to clip_length
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)

        # random crop and flip
        if self.training:
            input_data = self.random_flip(self.random_crop(input_data))
        else:
            input_data = self.center_crop(input_data)

        # import pdb;pdb.set_trace()
        input_data = torch.from_numpy(input_data).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        ssl_input_data, ssl_annos, flag = self.augment(input_data, annos, th, 1)
        annos = annos_transform(annos, self.clip_length)
        target = np.stack(annos, 0)
        ssl_target = np.stack(ssl_annos, 0)

        scores = np.stack([
            sample_info['start'],
            sample_info['end']
        ], axis=0)
        scores = torch.from_numpy(scores.copy()).float()

        return input_data, target, scores, ssl_input_data, ssl_target, flag
    """
    def __getitem__(self, index):
        
        mask_data, top_branch, bottom_branch, mask_top, cas_mask, mask_data_big, mask_data_small, b_mask = self.getVideoData(index)

        if self.mode == "train":
            return mask_data,top_branch,bottom_branch,mask_top,cas_mask, mask_data_big, mask_data_small, b_mask
            # return mask_data,top_branch,bottom_branch,mask_top,cas_mask
        else:
            return index, mask_data, mask_data_big, mask_data_small, b_mask
            # return index, mask_data


def detection_collate(batch):
    targets = []
    clips = []
    scores = []

    ssl_targets = []
    ssl_clips = []
    flags = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        scores.append(sample[2])

        ssl_clips.append(sample[3])
        ssl_targets.append(torch.FloatTensor(sample[4]))
        flags.append(sample[5])
    return torch.stack(clips, 0), targets, torch.stack(scores, 0), \
           torch.stack(ssl_clips, 0), ssl_targets, flags
