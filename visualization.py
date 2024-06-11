""" Pseudocode for parsing and analyzing the output files, perhaps through visual graphs:
for example, outfile_main.txt


f = open('output/experiments/outfile_main.txt', 'r')
content = f.readlines() # puts all lines into a list

# ---- get the program run: ----
>>> content[0].strip()
'./spot_train_eval.sh outfile_main.txt'

# ---- get the full arguments (ready for use in with json.loads) ----
>>> content[1].strip()
"{'dataset': {'name': 'anet', 'num_classes': 200, 'training': {'video_info_path': './data/activitynet_annotations/video_info_new.csv', 'video_info_path_unlabeled': './data/activitynet_annotations/', 'video_anno_path': './data/activitynet_annotations/anet_anno_action.json', 'num_fr
ame': 16, 'output_path': '/root/models/SPOT/output/', 'unlabel_percent': 0.9, 'use_semi': True}, 'testing': {'video_info_path': './data/activitynet_annotations/video_info_new.csv', 'video_info_path_unlabeled': './data/activitynet_annotations/', 'video_anno_path': './data/activityn
et_annotations/anet_anno_action.json', 'num_frame': 16, 'output_path': '/root/models/SPOT/output/', 'unlabel_percent': 0.9, 'use_semi': True}}, 'model': {'embedding_head': 4, 'feat_dim': 400, 'temporal_scale': 100}, 'pretraining': {'warmup_epoch': 30}, 'training': {'batch_size': 2
5, 'learning_rate': 0.0004, 'weight_decay': 0.005, 'max_epoch': 25, 'checkpoint_path': '/root/models/SPOT/output/', 'random_seed': 1, 'step': 10, 'gamma': 0.2, 'feature_path': '/data/i5O/ActivityNet1.3/train/', 'num_gpu': 1}, 'loss': {'lambda_1': 0.5, 'lambda_2': 0.4}, 'testing': 
{'feature_path': '/data/i5O/ActivityNet1.3/test/', 'cls_thresh': 0.01, 'mask_thresh': [0, 0.2, 0.4, 0.6, 0.8], 'class_thresh': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'top_k_snip': 10, 'top_k': 500, 'nms_thresh': 0.6}}"

# ---- to FIND desired rows (ie. when searching through the content) we can search for a substring and see if it is included ----
>>> content[68]
'n_iter  0 : loss (0.771401) + tot_loss (0.983506) + tot_loss_crop (0.983506) + loss_clip_order (0.755561) = final_loss = 3.493973\n'
>>> type(re.fullmatch('.*n_iter.*', content[68].strip(), flags=re.DOTALL))
<class 're.Match'>
>>> type(re.fullmatch('.*wrogiwroignoeign.*', content[68].strip(), flags=re.DOTALL))
<class 'NoneType'>

# ---- to PARSE desired rows once found ----:

# - pretraining minibatch
>>> content[68]
'n_iter  0 : loss (0.771401) + tot_loss (0.983506) + tot_loss_crop (0.983506) + loss_clip_order (0.755561) = final_loss = 3.493973\n'
>>> r = re.search('n_iter  (\d+).*\((.*)\).*\((.*)\).*\((.*)\).*\((.*)\).*(\d+\.\d+)', content[68].strip()); [r.group(i) for i in range(1, 7)]
['0', '0.771401', '0.983506', '0.983506', '0.755561', '3.493973']
>>> r = re.search('n_iter  (\d+).*loss.*\((.*)\).*tot_loss.*\((.*)\).*tot_loss_crop.*\((.*)\).*loss_clip_order.*\((.*)\).*final_loss.*(\d+\.\d+)', content[68].strip()); [r.group(i) for i in range(1, 7)] # not necessary but easier to read
['0', '0.771401', '0.983506', '0.983506', '0.755561', '3.493973']

# - other cases TODO...

"""
