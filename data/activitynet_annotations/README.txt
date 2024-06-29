ActivityNet: 

Number of videos with a given number of annotations. Some videos have no class instances, most have just 1, and there is a long tail (some have 23!)
>>> Counter([len(anet_anno_action[k]['annotations']) for k in anet_anno_action])
Counter({1: 10848, 0: 4851, 2: 1873, 3: 686, 4: 401, 5: 213, 6: 133, 7: 74, 8: 54, 9: 31, 10: 23, 11: 14, 12: 7, 18: 4, 15: 4, 13: 4, 17: 3, 23: 2, 19: 1, 20: 1, 16: 1})

in ../../evaluation/activity_net_1_3_new.json, there is a taxonomy file for hierarchical classes.


./video_info_new.csv -------------------------------------------------------
In video_info_new_X.csv, X = the proportion of the training set (50% of the videos are training set) that is considered as training_unlabeled. This would be for the purpose of trying different proportions of unlabeled training data.
Note that video_info_new_0.0 = video_info_new (the fully-supervised case)

entries look like:
(venv_SPOT) root@6c970ba04703:~/models/SPOT/data/activitynet_annotations# cat video_info_new.csv | grep v_QOlSCBRmfWY
v_QOlSCBRmfWY,2067,82.73,25.0,24.984890608,training,2064

./anet_anno_action.json: ---------------------------------------------------
one entry per video, each entry looks like this:
>>> list(anet_anno_action.keys())[0]
'v_QOlSCBRmfWY'
>>> anet_anno_action[list(anet_anno_action.keys())[0]]
{'duration_second': 82.73, 'duration_frame': 2067, 'annotations': [{'segment': [6.195294851794072, 77.73085420904837], 'label': 'Ballet'}], 'feature_frame': 2064}

./per_label_num.json: ------------------------------------------------------
200 entries, where each entry is class_name: num of that class



Video Distribution Breakdown:
Total Videos: 19228
- training: 9649
- - training_unlabeled: X * 9649, where X is the proportion of unlabeled (eg. 0.9)
- - train_loader_pretrain in spot_train.py uses (1 - X) * 9649 (ie. the labeled portion of the training set), or 80-ish% of it
- - train_loader_unlabel in spot_train.py uses X * 9649 (ie. the unlabled portion of the training set), or 80-ish% of it (uses an adjusted batch size; 24 when unlabeled=0.9, 4 when unlabeled=0.0)
- validation: 4728
- testing: 4851 # the test_loader in spot_train.py

The length of the data loader is the number of videos divided by the number of videos per batch (the batch_size)

The setup seems to be:
We only have the train and test sets 
- train is where we get the training
- - X is the proportion that gets used for train_loader 
- - (1 - X) is the proportion that gets used for train_loader_unlabel
- test set we use as validation in train_semi (no backprop) and we use this same test set in its entirety as a test set in spot_inference; hence, it is okay to use the same dataset in train_semi and spot_inference 

# TODOs:
- ensure that the train_loader (regular one) does what it is supposed to
- ensure that the spot_inference loader uses the test set, whereas the validation uses the validation set; currently, it seems as though both use the validation set
