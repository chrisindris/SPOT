# Usage: python ./SANDBOX_spot_train.py </path/to/config.yaml> [<config_argument> ...]
import torch
import numpy as np
import random
import sys
import yaml
from utils.arguments import handle_args, modify_config
import subprocess
from spot_lib.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))

print(config)

batch_size = config['training']['batch_size']

GLOBAL_SEED = 1
def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
train_video_annos = get_video_anno(train_video_infos,
                                   config['dataset']['training']['video_anno_path'])
train_data_dict = load_video_data(train_video_infos,
                                  config['training']['feature_path'])
train_dataset = THUMOS_Dataset(train_data_dict,
                               train_video_infos,
                               train_video_annos)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, worker_init_fn=worker_init_fn,
                               collate_fn=detection_collate, pin_memory=True, drop_last=True)
epoch_step_num = len(train_dataset) // batch_size
