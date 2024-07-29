# test_Annotation_ours.csv -> anet_anno_action.json (-type structure)

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
        default="/root/models/SPOTworktree/data/thumos_annotations/test_Annotation_ours.csv",
        help=".csv file used as the base (lists each video).",
    )
    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="/root/models/SPOTworktree/data/thumos_annotations/test_video_info_new.csv",
        help=".csv with video info.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default="/root/models/SPOTworktree/data/thumos_annotations/thumos_anno_action.json",
        help=".json to save the resulting file.",
    )
    parser.add_argument("--save_dir", type=str, default="/root/models/SPOTworktree/data/thumos_annotations/", help="output directory")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()

    annos = pd.read_csv(args.file)
    infos = pd.read_csv(args.info)

    dic = {}
    for v in infos['video']:
        infos_row = infos[infos['video'] == v]
        dic[v] = {}
        dic[v]['duration_second'] = (infos_row['count'] / infos_row['fps']).item()
        dic[v]['duration_frame'] = int(infos_row['count'].item())
        dic[v]['feature_frame'] = int(infos_row['count'].item())
        dic[v]['annotations'] = []
