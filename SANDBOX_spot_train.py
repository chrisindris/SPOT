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


if sys.argv[1] == "configs/thumos.yaml":
    from spot_lib.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno

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

    #train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    #test_video_infos = get_video_info(config['dataset']['testing']['video_info_path'])

    #train_video_annos = get_video_anno(train_video_infos, config['dataset']['training']['video_anno_path'])
    #test_video_annos = get_video_anno(test_video_infos, config['dataset']['testing']['video_anno_path'])

    #train_data_dict = load_video_data(train_video_infos, config['training']['feature_path'])
    #test_data_dict = load_video_data(test_video_infos, config['testing']['feature_path'])
    print('train')
    train_dataset = THUMOS_Dataset(training=True, subset='train', labeled=True)

    print('train_unlabel')
    train_unlabel_dataset = THUMOS_Dataset(training=True, subset='train', unlabeled=True)

    print('test')
    test_dataset = THUMOS_Dataset(training=False, subset='testing')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=False, drop_last=True)
 
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=False, drop_last=True)
 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1, pin_memory=False, drop_last=True)

    #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                               num_workers=4, worker_init_fn=worker_init_fn,
    #                               collate_fn=detection_collate, pin_memory=True, drop_last=True)


    count = 0
    for n_iter, (mask_data,classifier_branch,global_mask_branch,mask_top,cas_mask,mask_data_big,mask_data_small,b_mask) in enumerate(train_loader):
        count += 1
        #break#breakpoint()
        #print(cas_mask[0])
        #if count == 8:
        #    break
    print(count)

    count = 0
    for n_iter, (mask_data,classifier_branch,global_mask_branch,mask_top,cas_mask,mask_data_big,mask_data_small,b_mask) in enumerate(train_unlabel_loader):
        #breakpoint()
        count += 1
    print(count)

    count = 0
    for n_iter, (index, mask_data, mask_data_big, mask_data_small, b_mask) in enumerate(test_loader):
        #breakpoint()
        count += 1
    print(count)

    epoch_step_num = len(train_dataset) // batch_size


elif sys.argv[1] == "configs/anet.yaml":

    # This is ANet mode; keep this as-is, just use it as a guide for THUMOS

    num_batch = config['training']['batch_size']
    unlabel_percent = config['dataset']['training']['unlabel_percent']

    print('train_loader')
    train_loader = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="train"),
                                                   batch_size=1, shuffle=False,
                                                   num_workers=1, pin_memory=False, drop_last=True)

    print('train_loader_pretrain')
    if config['pretraining']['unlabeled_pretrain']:
        train_loader_pretrain = torch.utils.data.DataLoader(spot_dataset.SPOTDatasetUnlabeled(subset="unlabel"),
                                               #batch_size=num_batch, shuffle=True,
                                               batch_size=1, shuffle=False,drop_last=True,
                                                        num_workers=1, pin_memory=False) # NOTE pretrain_unlabel: we see what happens when we pretrain with the unlabeled data
    else:
        train_loader_pretrain = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="train"),
                                             batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=False, drop_last=True)


    print('train_loader_unlabel')
    train_loader_unlabel = torch.utils.data.DataLoader(spot_dataset.SPOTDatasetUnlabeled(subset="unlabel"),
                                            #    batch_size=num_batch, shuffle=True,
                                               batch_size=1, shuffle=False,drop_last=True,
                                               num_workers=1, pin_memory=False)
    print("training: len(train_loader_unlabel)", len(train_loader_unlabel))
     
        
#breakpoint()
    print('test_loader')
    test_loader = torch.utils.data.DataLoader(spot_dataset.SPOTDataset(subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=False, drop_last=True)

    count = 0
    for n_iter, (mask_data,classifier_branch,global_mask_branch,mask_top,cas_mask,mask_data_big,mask_data_small,b_mask) in enumerate(train_loader):
        print(count)
        count += 1
        #break#breakpoint()
        #print(cas_mask[0])
        if count > 0:
            break
    print(count)

    del train_loader
