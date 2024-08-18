# Serves to divide into testing, train_unsupervised, test_unsupervised, validation or any other label you so desire.
# for i in {0..9}; do p=$(bc <<< "scale=1; $i / 10"); python labeller.py -u $p; done

import argparse
import os
import numpy as np
import re
import pandas as pd
import pdb


def argument_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--operation", type=str, help="operation to run")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="/root/models/SPOTworktree/data/thumos_annotations/val_video_info.csv",
        help=".csv file used as the base (lists each video).",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="/root/models/SPOTworktree/data/thumos_annotations/",
        help="Directory to save the resulting file.",
    )
    parser.add_argument(
        "-a",
        "--annotations",
        type=str,
        default="/root/models/SPOTworktree/data/thumos_annotations/val_Annotation_ours.csv",
        help=".csv file listing each annotation.",
    )
    parser.add_argument(
        "-u",
        "--percent_unlabel",
        type=float,
        default=0.9,
        help="The percent of the training set to be marked as unlabeled",
    )
    parser.add_argument("--save_dir", type=str, default="/root/models/SPOTworktree/data/thumos_annotations/", help="output directory")
    return parser

subset = []
def build_subset_lambda(count):
    global subset
    count_unlabel = int(round(count * args.percent_unlabel))
    subset += ['training_unlabel'] * count_unlabel + ['training'] * (count - count_unlabel)


def build_subset(count):
    count_unlabel = int(round(count * args.percent_unlabel))
    subset = np.array(['training_unlabel'] * count_unlabel + ['training'] * (count - count_unlabel))
    np.random.shuffle(subset)
    return subset


def thumos():
    parser = argument_parser()
    args = parser.parse_args()

    video_info = pd.read_csv(args.file)
    annotations = pd.read_csv(args.annotations) 

    # --- let's see if every class is represented everywhere. ---

    #count_of_each_class_sorted = annotations[['video', 'type']].sort_values(['type']).groupby(['type']).count()
    annotations_sorted = annotations.sort_values(['type'])

    #table = annotations_sorted[['type']].drop_duplicates(subset=['type'])
    #table['count'] = list(count_of_each_class_sorted['video'])
    #table.apply(lambda row: build_subset(row['count']), axis=1)
    #annotations_sorted['subset'] = subset
    #annotations = annotations_sorted.sort_index()

    # Build the table that, for each video, gives the class count
    table = video_info[['video']].set_index('video')
    for c in list(annotations_sorted['type'].drop_duplicates()):
        ctable = annotations_sorted[['video', 'type']][annotations_sorted['type'] == c].groupby(['video']).count().reset_index().rename(columns={'type':c})
        #breakpoint()
        #table = table.set_index('video').join(c_counts.set_index('video'), how='outer')
        table = table.join(ctable.set_index('video'), on='video')
    table = table.reset_index()

    # - check
    #video_info_training = video_info[video_info['subset'] == 'training']['video']
    #table_training = table[table['video'].isin(video_info_training)]

    # To keep the dataset balanced, we maximize the minimum number of videos with an example of a class for each class.
    planned_forced_videos_per_class = int(round(len(video_info) * (1 - args.percent_unlabel)) // len(annotations['type'].drop_duplicates()))
    forced_videos_per_class = max(planned_forced_videos_per_class, 1) if args.percent_unlabel < 1.0 else 0

    video_selections = []
    for c in annotations['type'].drop_duplicates():
        current_video_selection = annotations[annotations['type'] == c]['video'].drop_duplicates().sample(n=forced_videos_per_class)
        video_selections.append(current_video_selection)
    video_selections = pd.concat(video_selections)

    forced_selection = video_info[video_info['video'].isin(video_selections)]
    table_selection = table[table['video'].isin(video_selections)]

    table_binary = table_selection[table_selection.columns[1:]].fillna(0)
    table_binary[table_binary > 1] = 1

    planned_forced_videos = int(round(len(video_info) * (1 - args.percent_unlabel)))
    excess = len(forced_selection) - planned_forced_videos

    if excess > 0: # This case may happen if the number of classes exceeds the number of videos that we want to be unlabeled.

        # if we have more videos than we should, we will see if we can pop a redundant video (ie. find a video so that if it is removed, there is still at least one of each video)
        while excess > 0:
            change_made = False
            for i in table_binary.index:
                tb = table_binary.drop(index=i)
                if tb.sum().all():
                    table_binary = tb
                    change_made = True
                    excess -= 1
                    if excess == 0:
                        break
            if not change_made: # The dataset cannot be reduced without not having at least one example of each class => accept the larger dataset
                break
        
        forced_selection = forced_selection.loc[table_binary.index]
        table_selection = table_selection.loc[table_binary.index]
        selection = table_selection

    else: # after the forced_selection has ensured that we have at least one example of each action, we are free to add the remaining videos using the videos we have not yet used.
        unforced_selection = video_info[~video_info['video'].isin(forced_selection['video'])].sample(n=-excess)
        selection = pd.concat([forced_selection, unforced_selection])


    video_info['subset'] = video_info.index.isin(selection.index)
    video_info['subset'] = video_info.apply(lambda r : ['training_unlabel', 'training'][r['subset']], axis=1)

    video_info.to_csv(os.path.join(args.save_dir, "val_video_info_" + str(args.percent_unlabel) + ".csv"), index=False)

    # --- propose a partition into training_unlabel and training ---
    #video_info['subset'] = build_subset(len(video_info))


def i5O():
    print('copy in the stuff from __main__ here after debugging') 


if __name__ == "__main__":
    i5O()
    df_info = pd.read_csv('/data/i5O/i5OData/annotations/i5Oannotations.csv')
    df_info = df_info.drop('Unnamed: 0', axis=1)
    df_info['action_orientation'] = df_info.apply(lambda r: 'left' if 'left' in r['video_path'] else 'right', axis=1)

    # TODO: check this case undercover-left_20220413_144609.npy
    # Confirmed: all videos exist in the directory, and none don't.

    df_info = df_info[['video_path', 'action_orientation', 'video_basename', 'frame_rate', 'duration_secs', 'frame_count', 'split']].drop_duplicates(['video_path']) # This forms the basis of our video info file.


    #df_info[['video_path']].drop_duplicates().apply(lambda r : re.search(".*(left|right).*", str(r['video_path'])).group(1), axis=1)
    video_dirnames = df_info[['video_path']].drop_duplicates().apply(lambda r : re.search(".*videos/(\d+).*", str(r['video_path'])).group(1), axis=1) 
    df_info.insert(2, 'video_dirname', video_dirnames)

    df_info['video_basename'] = df_info[['video_path']].drop_duplicates().apply(lambda r : re.search(".*/(\d+)\.mp4.*", str(r['video_path'])).group(1), axis=1)
    
    df_info = df_info.reset_index().drop(['index'], axis=1) # axis is now range(num_videos)
    
    for k in range(11):
        df_info_k = df_info.copy()
        unlabel_percentage = (10 - k) / 10
        df_info_k['split'] = df_info.apply(lambda r: r['split'] + str(['_unlabel', ''][(r.name % (12 - round(k / 12)*2) in range(k)) or (r['split'] == 'Test') or (r.name % 60 in range(60-k,60))]), axis=1)
        df_info_k.to_csv("~/models/SPOT/data/i5O_annotations/video_info_new_" + str(unlabel_percentage) + ".csv", index=False)
        #breakpoint()
     

