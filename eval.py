import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from configs.dataset_class import activity_dict
# from gsm_lib import opts
import yaml 
from utils.arguments import handle_args, modify_config

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
print(config)

output_path = config['dataset']['testing']['output_path']
nms_thresh = config['testing']['nms_thresh']

from evaluation.eval_detection import ANETdetection

# from evaluation.eval_proposal import ANETproposal
anet_detection = ANETdetection(
    ground_truth_filename="./evaluation/activity_net_1_3_new.json",
    prediction_filename=os.path.join(output_path, "detection_result_nms{}.json".format(nms_thresh)),
    subset='validation', verbose=False, check_status=False)
anet_detection.evaluate()

mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
print(results)
with open(os.path.join(output_path, 'results.txt'), 'a') as fobj:
    fobj.write(f'{results}\n')
