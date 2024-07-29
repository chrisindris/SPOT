import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou
from configs.dataset_class import activity_dict, thumos_dict
import pdb

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

pred_data = pd.read_csv("spot_output_semi.csv")
pred_videos = list(pred_data.video_name.values[:])

"""
class Detection:
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def evaluate(self):
        raise NotImplementedError("Subclasses of Detection should implement evaluate")
"""

# --------------------------------------------------------------------------------------


class THUMOSdetection(object):

    # This code is originally from the official ActivityNet repo
    # https://github.com/activitynet/ActivityNet
    # Small modification from ActivityNet Code

    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.thumos_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """

        data = pd.read_csv(ground_truth_filename)

        ground_truth = data[['video', 'start', 'end', 'type']]
        ground_truth.columns = ['video-id', 't-start', 't-end', 'label']
        ground_truth.loc[:, 'label'] = ground_truth.apply(lambda r: thumos_dict[r['label']], axis=1).values

        return ground_truth, thumos_dict

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        thumos_index_prep = thumos_dict
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                if result['label'] not in self.thumos_index:
                  label = thumos_index_prep[result['label']]
                else:
                  label = self.thumos_index[result['label']]
                # label = self.activity_index[result['label']]
                video_lst.append(videoid)
                # print("pred_vid",videoid)
                # print("pred_label",result['label'])
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.thumos_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label') # video-id, t-start, t-end, label - the ground truth actions  
        prediction_by_label = self.prediction.groupby('label') # video-id, t-start, t-end, label, score - the prediction actions

        results = Parallel(n_jobs=1)(#len(self.thumos_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.thumos_index.items())

        for i, cidx in enumerate(self.thumos_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP

# --------------------------------------------------------------------------------------

class ANETdetection(object):

    # This code is originally from the official ActivityNet repo
    # https://github.com/activitynet/ActivityNet
    # Small modification from ActivityNet Code

    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):

        if not ground_truth_filename: # './evaluation/activity_net_1_3_new.json'
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset # validation
        self.tiou_thresholds = tiou_thresholds # [0.5, 0.55, ..., 0.9, 0.95]
        self.verbose = verbose # False
        self.gt_fields = ground_truth_fields # ['database']
        self.pred_fields = prediction_fields # ['results', 'version', 'external_data']
        self.ap = None
        self.check_status = check_status # False
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        # self.ground_truth: pd table: video-id (video name), t-start, t-end (times of the annotation), label (numerical index), for the videos used for testing; video-id, t-start, t-end directly correspond to both activity_net_1_3_new and anet_anno_action.
        # self.activity_index: len 200 dict of string name : numerical index, although the index doesn't match either the dataset_class.py or action_name.csv (although it of course matches the indexes used in self.ground_truth) 
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename) # the predictions (detection_result_nms0.6.json), expressed as a pd dataframe (each row is video-id, t-start, t-end, label, score)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """

        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0 # I assume why we start with an empty activity_index is so that we can handle the case in which not every class is represented?
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            # print(videoid, pred_videos[2:])
            if "v_"+videoid in pred_videos:
                
                # print(v)
                if self.subset != v['subset']: # only the testing subset videos are included
                    continue
                if videoid in self.blocked_videos:
                    continue
                for ann in v['annotations']:
                    if ann['label'] not in activity_index:
                        activity_index[ann['label']] = cidx
                        cidx += 1
                    video_lst.append(videoid)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """

        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj) # data['results'] has keys being video names and the value is a list of predictions, where each prediction is a label,score,segment dictionary.
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        activity_index_prep = activity_dict # the original dataset
        for videoid, v in data['results'].items(): # videoid is the key (name), v is the value (the list)
            if videoid in self.blocked_videos:
                continue
            for result in v:
                if result['label'] not in self.activity_index: # this seems suspicious; by using two different indexings simultaneously, this could lead to overwrites.
                  label = activity_index_prep[result['label']]
                else:
                  label = self.activity_index[result['label']]
                # label = self.activity_index[result['label']]
                video_lst.append(videoid)
                # print("pred_vid",videoid)
                # print("pred_label",result['label'])
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction # the "data", but now represented as a pd dataframe.

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label') # video-id, t-start, t-end, label - the ground truth actions
        prediction_by_label = self.prediction.groupby('label') # video-id, t-start, t-end, label, score - the prediction actions

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision() # numpy array, size (len(self.tiou_thresholds)=10, num_classes=200)

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """

    #breakpoint()

    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth)) # the number of ground truth actions of a particular label
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1 # an array; will store the ap at various tiou thresholds for each ground truth action
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True) # re-sort so that the highest-scoring predictions come first

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])


    return ap
