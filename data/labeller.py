# Serves to divide into testing, train_unsupervised, test_unsupervised, validation or any other label you so desire.
# for i in {0..9}; do p=$(bc <<< "scale=1; $i / 10"); python labeller.py -u $p; done

import argparse
import os
import numpy as np
import re
import pandas as pd


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


if __name__ == "__main__":
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
     

