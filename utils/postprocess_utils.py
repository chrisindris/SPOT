import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from configs.dataset_class import activity_dict
# from gsm_lib import opts
import yaml
import sys
from utils.arguments import handle_args, modify_config

import pdb

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
        temporal_scale = config['model']['temporal_scale']
        num_classes = config['dataset']['num_classes']
        dataset_name = config['dataset']['name']


vid_info = config['dataset']['testing']['video_info_path']
vid_anno = config['dataset']['testing']['video_anno_path_json']
vid_path = config['testing']['feature_path']
nms_thresh = config['testing']['nms_thresh']

if dataset_name == 'anet':
    testing_subset = 'validation'
elif dataset_name == 'thumos':
    testing_subset = 'testing'

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict():

    #breakpoint()

    df = pd.read_csv(vid_info)
    json_data = load_json(vid_anno)
    database = json_data
    video_dict = {}
    video_label_dict={}
    for i in range(len(df)):
        video_name = df.video.values[i]
        # if os.path.exists(os.path.join(vid_path+"/validation",video_name+".npy")):
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['duration_frame']
        video_subset = df.subset.values[i]
        video_anno = video_info['annotations']
        video_new_info['annotations'] = video_info['annotations']
        if len(video_anno) > 0:
            video_label = video_info['annotations'][0]['label']
            if video_subset == testing_subset:
                    video_dict[video_name] = video_new_info
                    video_label_dict[video_name] = video_label
    return video_dict , video_label_dict



def Soft_NMS(df, nms_threshold=1e-5, num_prop=temporal_scale):
 
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    tlabel = list(df.label.values[:])

    rstart = []
    rend = []
    rscore = []
    rlabel = []


    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * (np.exp(-np.square(tmp_iou)*10) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rlabel.append(tlabel[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    newDf['label'] = rlabel

    return newDf



def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - min(s1, s2)
    return float(Aand) / (Aor - Aand + (e2-s2))



def multithread_detection(video_name, video_cls, video_info, label_dict, pred_prop, best_cls, num_prop=num_classes, topk = 2):

    if dataset_name == 'anet':
        modified_video_name = "v_"+video_name
    else:
        modified_video_name = video_name
    
    old_df = pred_prop[pred_prop.video_name == modified_video_name]
    # print(df)
    
    df = pd.DataFrame()
    df['score'] = old_df.reg_score.values[:]*old_df.clr_score.values[:]
    df['label'] = old_df.label.values[:]
    df['xmin'] = old_df.xmin.values[:]
    df['xmax'] = old_df.xmax.values[:]

    best_score = best_cls[modified_video_name]["score"]
    best_label = best_cls[modified_video_name]["class"]

    if len(df) > 1:
        df = Soft_NMS(df, nms_thresh)
    df = df.sort_values(by="score", ascending=False)
    video_duration=float(video_info["feature_frame"])/video_info["duration_frame"]*video_info["duration_second"]
    proposal_list = []

    for j in range(min(temporal_scale, len(df))):
        
        tmp_proposal = {}
        if float(df.score.values[j]) > best_score:
            tmp_proposal["label"] = str(df.label.values[j])
        else:
            tmp_proposal["label"] = str(df.label.values[j])

        tmp_proposal["score"] = float(df.score.values[j])
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)

    return {video_name: proposal_list}






