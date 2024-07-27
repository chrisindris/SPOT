# Usage: python ./SANDBOX_spot_train.py </path/to/config.yaml> [<config_argument> ...]
import torch
import numpy as np
import random
import sys
import yaml
from utils.arguments import handle_args, modify_config
import subprocess
from torch.utils.data import DataLoader
import spot_lib.spot_dataloader as spot_dataset
import pdb

import gc

torch.cuda.empty_cache()
gc.collect()
print(torch.cuda.mem_get_info())

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))

print(config)
dataset_name = config['dataset']['name']
output_path = config['dataset']['training']['output_path']
num_gpu = config['training']['num_gpu']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
decay = config['training']['weight_decay']
epoch = config['training']['max_epoch']
num_batch = config['training']['batch_size']
step_train = config['training']['step']
gamma_train = config['training']['gamma']
fix_seed = config['training']['random_seed']
use_semi = config['dataset']['training']['use_semi']
unlabel_percent = config['dataset']['training']['unlabel_percent']



if 'anet' in sys.argv[1]:
    dataset_name = 'anet'
else:
    dataset_name = 'thumos'


if dataset_name == 'anet':
    train_dataset = spot_dataset.SPOTDataset(subset="train")
    train_unlabel_dataset = spot_dataset.SPOTDatasetUnlabeled(subset="unlabel")
    test_dataset = spot_dataset.SPOTDataset(subset='validation')
elif dataset_name == 'thumos':
    from spot_lib.thumos_dataset import THUMOS_Dataset
    train_dataset = THUMOS_Dataset(training=True, subset='train', labeled=True)
    train_unlabel_dataset = THUMOS_Dataset(training=True, subset='train', unlabeled=True)
    test_dataset = THUMOS_Dataset(training=False, subset='testing')


print('train_loader')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=num_batch, shuffle=False,
                                           num_workers=1, pin_memory=False, drop_last=True)

print('train_loader_pretrain')
if config['pretraining']['unlabeled_pretrain']:
    train_loader_pretrain = torch.utils.data.DataLoader(train_unlabel_dataset,
                                           #batch_size=num_batch, shuffle=True,
                                           batch_size=num_batch, shuffle=False,drop_last=True,
                                                    num_workers=1, pin_memory=False) # NOTE pretrain_unlabel: we see what happens when we pretrain with the unlabeled data
else:
    train_loader_pretrain = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=num_batch, shuffle=False,
                                          num_workers=1, pin_memory=False, drop_last=True)

if use_semi and unlabel_percent > 0.:
    print('train_loader_unlabel')
    train_loader_unlabel = torch.utils.data.DataLoader(train_unlabel_dataset,
                                        #    batch_size=num_batch, shuffle=True,
                                           batch_size=min(max(round(num_batch*unlabel_percent/(4*(1.-unlabel_percent)))*4, 4), 24), 
                                           shuffle=False,drop_last=True,
                                           num_workers=1, pin_memory=False)
    print("training: len(train_loader_unlabel)", len(train_loader_unlabel))
    print("batch_size", batch_size)
    print("adjusted batch_size", min(max(round(num_batch*unlabel_percent/(4*(1.-unlabel_percent)))*4, 4), 24))
 
    
#breakpoint()
print('test_loader')
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=num_batch, shuffle=False,
                                          num_workers=2, pin_memory=False, drop_last=True)


print("training: len(train_loader)", len(train_loader))
print("training: len(train_loader_pretrain)", len(train_loader_pretrain)) 
print("training: len(test_loader)", len(test_loader))
