import os
import math
import numpy as np
import pandas as pd
import torch.nn.parallel
import itertools,operator
# from gsm_lib import opts
from spot_model import SPOT
import spot_lib.spot_dataloader as spot_dataset
from scipy import ndimage
from scipy.special import softmax
import torch.nn.functional as F
from collections import Counter
import cv2
import json
from configs.dataset_class import activity_dict
import yaml
from utils.postprocess_utils import multithread_detection , get_infer_dict, load_json
from joblib import Parallel, delayed
from spot_lib.tsne import viusalize
import sys
from utils.arguments import handle_args, modify_config

import pdb

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
print(config)


################## fix everything ##################
fix_seed = config['training']['random_seed']
import random
seed = fix_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################


if __name__ == '__main__':
    mode = "semi"  ## "semi", "semi_ema" ,""
    output_path = config['dataset']['testing']['output_path']
    # im_fig_path = config['testing']['fig_path']
    is_postprocess = True
    if not os.path.exists(output_path + "/results"):
        os.makedirs(output_path + "/results")
    ### Load Model ###
    model = SPOT()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # NOTE GPU: This works with an arbitrary CUDA_VISIBLE_DEVICES=X (as long as X is 1 GPU). To make this work with more, change it to range(config['training']['num_gpus'])
    ### Load Checkpoint ###
    checkpoint = torch.load(output_path + "/SPOT_best_semi.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    ### Load Dataloader ###
    test_loader = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset='validation', mode='inference'), # to get validation set (which is /data/i5O/ActivityNet1.3/test/): subset='validation', mode='inference'
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)

    print("len(test_loader),", len(test_loader))

    # im_path = os.path.join(im_fig_path,"SOLO_PIC")
   
    # the classes: key_list are the string names, val_list are the indexes 
    key_list = list(activity_dict.keys())
    val_list = list(activity_dict.values())

    nms_thres = config['testing']['nms_thresh'] # 0.6 by default
    
    def save_plot(x,save_path):
        """[TODO:description]

        Args:
            x ([TODO:parameter]): [TODO:description]
            save_path ([TODO:parameter]): [TODO:description]
        """
        fig = plt.figure()
        ax = plt.axes()
        fig = plt.figure()
        ax = plt.axes()
        plt.grid(False)
        plt.plot(x,color='red', linewidth=8);
        plt.xlim(0, 100)
        plt.ylim(0, 1);
        plt.xticks([])
        plt.yticks([])
        fig = plt.gcf()
        fig.set_size_inches(25.5, 2.5)
        plt.savefig(save_path)

    def post_process_multi(detection_thread,get_infer_dict):
        """[TODO:description]

        Args:
            detection_thread ([TODO:parameter]): [TODO:description]
            get_infer_dict ([TODO:parameter]): [TODO:description]
        """
        mode="semi"
        infer_dict , label_dict = get_infer_dict() # infer_dict and label_dict both have the video name as keys; infer_dict has the information, label_dict has the string name of the class of the first annotation to appear.
        pred_data = pd.read_csv("spot_output_"+mode+".csv")
        pred_videos = list(pred_data.video_name.values[:])
        cls_data_score, cls_data_cls = {}, {}
        best_cls = load_json("spot_best_score.json")
        
        for idx, vid in enumerate(infer_dict.keys()):
            if vid in pred_videos:
                vid = vid[2:] 
                cls_data_cls[vid] = best_cls["v_"+vid]["class"] 

        parallel = Parallel(n_jobs=15, prefer="processes")
        detection = parallel(delayed(detection_thread)(vid, video_cls, infer_dict['v_'+vid], label_dict, pred_data, best_cls)
                            for vid, video_cls in cls_data_cls.items())
        detection_dict = {}
        [detection_dict.update(d) for d in detection]
        output_dict = {"version": "ANET v1.3, SPOT", "results": detection_dict, "external_data": {}}

        with open(output_path + '/detection_result_nms{}.json'.format(nms_thres), "w") as out:
            json.dump(output_dict, out)


    #breakpoint()

    
    file = "spot_output_"+mode+".csv"
    if(os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)
    print("Inference start")
    with torch.no_grad():
        vid_count=0
        match_count=0
        vid_label_dict = {}
        results = {}
        result_dict = {}
        class_thres = config['testing']['cls_thresh']
        num_class = config['dataset']['num_classes']
        top_k_snip = config['testing']['top_k_snip']
        class_snip_thresh = config['testing']['class_thresh']
        mask_snip_thresh = config['testing']['mask_thresh']
        temporal_scale = config['model']['temporal_scale']
        full_label = True
        

        new_props = list()

        for idx, input_data, input_data_big, input_data_small,f_mask in test_loader: # Loads the features that get passed to the model, and loads the binary mask (no classes given, just the binary classes). 
            video_name = test_loader.dataset.subset_mask_list[idx[0]]
            vid_count+=1
            input_data = input_data.cuda()
            input_data_s = input_data_small.cuda()
            input_data_b = input_data_big.cuda()
            if not os.path.exists(output_path + "/fig/"+video_name):
                os.makedirs(output_path + "/fig/"+video_name)

            # forward pass
            top_br_pred, bottom_br_pred, feat = model(input_data.cuda())
            # - top_br_pred: [1, 201, 100]; aims to match a one-hot encode of the class at each time step
            # - bottom_br_pred: [1, 100, 100]; binary mask for where actions occur
            # - feat: [1, 400, 100]: Spot determines a feature for each of the temporal locations (the clips)


            ### global mask prediction ####
            props = bottom_br_pred[0].detach().cpu().numpy() # The proposals for where actions occur

            ### classifier branch prediction ###

            if full_label:
                best_cls = load_json("spot_best_score.json")
                full_cls = best_cls[video_name]["class"]
                full_cls_score = best_cls[video_name]["score"]


            soft_cas = torch.softmax(top_br_pred[0],dim=0) # a "softmax" of the top branch; this makes the prediction closer to the one-hot. 
            soft_cas_topk,soft_cas_topk_loc = torch.topk(soft_cas[:num_class],2,dim=0) # from top_branch, the top k=2 classes and their likelihood.
            top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class] # softmax among the non-background classes 

            label_pred = torch.softmax(torch.mean(top_br_pred[0][:num_class,:],dim=1),axis=0).detach().cpu().numpy() # The mean score for each class (shape [200])
            vid_label_id = np.argmax(label_pred) # the whole-video max-predicted class
            vid_label_sc = np.amax(label_pred) # the max score associated with that class
            props_mod = props[props>0] # Never used
            top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class]

            top_br_mean = np.mean(top_br_np,axis=1) # The mean score for each class after the softmax
            top_br_mean_max = np.amax(top_br_np,axis=1) # the maximum of those mean scores (ie. the score leading to the most likely class, for the full video)
            top_br_mean_id = np.argmax(top_br_mean) # the class of it
            
            
            soft_cas_np = soft_cas[:num_class].detach().cpu().numpy() # soft_cas, but no background


            ### for each snippet (ie. temporal location), store the max score and predicted class for that snippet ####
            seg_score = np.zeros([temporal_scale])
            seg_cls = []
            seg_mask = np.zeros([temporal_scale]) 

            for j in range(temporal_scale):
                seg_score[j] =  np.amax(soft_cas_np[:,j])
                seg_cls.append(np.argmax(soft_cas_np[:,j]))

            # seg_score[seg_score < class_thres] = 0

            thres = class_snip_thresh

            #breakpoint()
            cas_tuple = []
            for k in thres:
                filt_seg_score = seg_score > k
                integer_map1 = map(int,filt_seg_score)
                filt_seg_score_int = list(integer_map1)
                filt_seg_score_int = ndimage.binary_fill_holes(filt_seg_score_int).astype(int).tolist() # will be a 1 in locations where the score for a particular class exceeds a certain value.
                if 1 in filt_seg_score_int:
                    start_pt1 = filt_seg_score_int.index(1)
                    end_pt1 = len(filt_seg_score_int) - 1 - filt_seg_score_int[::-1].index(1)

                    if end_pt1 - start_pt1 > 1:
                        scores = np.amax(seg_score[start_pt1:end_pt1])
                        label = max(set(seg_cls[start_pt1:end_pt1]), key=seg_cls.count)
                        cas_tuple.append([start_pt1,end_pt1,scores,label])

            max_score, score_idx  = torch.max(soft_cas[:num_class],0)
            soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
            score_map = {}

            top_np = top_br_pred[0][:num_class].detach().cpu().numpy()  
            top_np_max = np.mean(top_np,axis=1)
            max_score_np = max_score.detach().cpu().numpy()
            score_idx = score_idx.detach().cpu().numpy()

            for ids in range(len(score_idx)):
                score_map[max_score_np[ids]]= score_idx[ids]

            
            k = top_k_snip ## more fast inference
            max_idx = np.argpartition(max_score_np, -k)[-k:]

            ### indexes of top K scores ###

            top_k_idx = max_idx[np.argsort(max_score_np[max_idx])][::-1].tolist()

            for locs in top_k_idx:

                seq = props[locs,:]
                thres = mask_snip_thresh

                for j in thres:
                    filtered_seq = seq > j
                
                    integer_map = map(int,filtered_seq)
                    filtered_seq_int = list(integer_map)
                    filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                    
                    if 1 in filtered_seq_int:

                        #### getting start and end point of mask from mask branch ####

                        start_pt1 = filtered_seq_int2.index(1)
                        end_pt1 = len(filtered_seq_int2) - 1 - filtered_seq_int2[::-1].index(1) 
                        r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int)),operator.itemgetter(1)) if x == 1), key=len)
                        start_pt = r[0][0]
                        end_pt = r[-1][0]
                        if (end_pt - start_pt)/temporal_scale > 0.02 : 
                        #### get (start,end,cls_score,reg_score,label) for each top-k snip ####

                            score_ = max_score_np[locs]
                            cls_score = score_
                            lbl_id = score_map[score_]
                            reg_score = np.mean(seq[start_pt+1:end_pt-1])
                            label = key_list[val_list.index(lbl_id)]
                            vid_label = key_list[val_list.index(vid_label_id)]
                            score_shift = np.amax(soft_cas_np[vid_label_id,start_pt:end_pt])
                            prop_start = start_pt1/temporal_scale
                            prop_end = end_pt1/temporal_scale
                            if full_label:
                                new_props.append([video_name, prop_start , prop_end , score_shift*reg_score, full_cls_score,full_cls])
                            else:
                                new_props.append([video_name, prop_start , prop_end , score_shift*reg_score, score_shift*cls_score,vid_label])     
                            
                            for m in range(len(cas_tuple)):
                                start_m = cas_tuple[m][0]
                                end_m = cas_tuple[m][1]
                                score_m = cas_tuple[m][2]
                                reg_score = np.amax(seq[start_m:end_m])
                                prop_start = start_m/temporal_scale
                                prop_end = end_m/temporal_scale
                                cls_score = score_m
                                if full_label:
                                    new_props.append([video_name, prop_start,prop_end,reg_score,full_cls_score,full_cls])
                                else:
                                    new_props.append([video_name, prop_start,prop_end,reg_score,cls_score,vid_label])

                                    
        ### filter duplicate proposals --> Less Time for Post-Processing #####
        new_props = np.stack(new_props)
        b_set = set(map(tuple,new_props))  
        result = map(list,b_set) 

        ### save the proposals in a csv file ###
        col_name = ["video_name","xmin", "xmax", "clr_score", "reg_score","label"]
        new_df = pd.DataFrame(result, columns=col_name)
        new_df.to_csv("spot_output_"+mode+".csv", index=False)   

        print("Inference finished")

    ###### Post-Process #####
    print("Start Post-Processing")
    #breakpoint()
    post_process_multi(multithread_detection,get_infer_dict)
    print("End Post-Processing")
    
        
