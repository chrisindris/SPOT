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
