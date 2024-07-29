import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from scipy.sparse import data
from configs.dataset_class import activity_dict, thumos_dict
# from gsm_lib import opts
import yaml 
from utils.arguments import handle_args, modify_config
from evaluation.eval_detection import ANETdetection, THUMOSdetection
import pdb

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
        dataset_name = config['dataset']['name']
#print(config)


################## fix everything ##################
fix_seed = config['training']['random_seed']
import random
seed = fix_seed
random.seed(seed)
np.random.seed(seed)
#######################################################

output_path = config['dataset']['testing']['output_path']
nms_thresh = config['testing']['nms_thresh']

if dataset_name == 'anet':
    detection = ANETdetection(
    ground_truth_filename="./evaluation/activity_net_1_3_new.json",
    prediction_filename=os.path.join(output_path, "detection_result_nms{}.json".format(nms_thresh)),
    subset='validation', verbose=False, check_status=False)
elif dataset_name == 'thumos':
    detection = THUMOSdetection(
    ground_truth_filename="./data/thumos_annotations/test_Annotation_ours.csv",
    prediction_filename=os.path.join(output_path, "detection_result_nms{}.json".format(nms_thresh)),
    subset='testing', verbose=False, check_status=False)

detection.evaluate()

mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(detection.tiou_thresholds, detection.mAP)]
results = f'Detection: average-mAP {detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
print(results)
with open(os.path.join(output_path, 'results.txt'), 'a') as fobj:
    fobj.write(f'{results}\n')
