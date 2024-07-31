# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import sys
from utils.arguments import handle_args, modify_config

from configs.dataset_class import activity_dict # NOTE: would need to be modified to include THUMOS, although I don't know if this class is ever used.

import pdb

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
        temporal_scale = config['model']['temporal_scale']
        num_classes = config['dataset']['num_classes']


ce = nn.CrossEntropyLoss()
bce = nn.BCELoss()
mse = nn.MSELoss()
nll = nn.NLLLoss()

lambda_1 = config['loss']['lambda_1']
lambda_2 = config['loss']['lambda_2']

easy_class = ['Windsurfing','Using the pommel horse','Using the monkey bar','Tango','Table soccer','Swinging at the playground','Surfing','Springboard diving','Snowboarding','Snow tubing','Slacklining','Skiing','Shoveling snow','Sailing','Rock climbing','River tubing','Riding bumper cars','Raking leaves','Rafting','Putting in contact lenses','Preparing pasta','Pole vault','Volleyball','Playing pool','Playing field hockey','Playing blackjack','Playing beach volleyball','Playing accordion','Plataform diving','Plastering','Mixing drinks','Making an omelette','Longboarding','Hurling','Horseback riding','Hitting a pinata','Hanging wallpaper','Hammer throw','Grooming dog','Getting a piercing','Elliptical trainer','Drum corps','Doing motocross','Decorating the Christmas tree','Curling','Croquet','Cleaning sink','Clean and jerk','Carving jack-o-lanterns','Camel ride']
hard_class = ['Drinking coffee','Doing a powerbomb','Polishing forniture','Putting on shoes','Removing curlers','Rock-paper-scissors','Gargling mouthwash','Having an ice cream','Polishing shoes','Smoking a cigarette','Applying sunscreen','Drinking beer','Washing face','Doing nails','Brushing hair','Playing harmonica','Painting furniture','Peeling potatoes','Cumbia','Cleaning shoes','Doing karate','Chopping wood','Hand washing clothes','Painting','Shaving legs','Using parallel bars','Baking cookies','Playing drums','Bathing dog','Kneeling','Hopscotch','Playing kickball','Doing crunches','Playing saxophone','Roof shingle removal','Shot put','Playing flauta','Swimming','Preparing salad','Washing dishes','Getting a tattoo','Getting a haircut','Fixing bicycle','Playing guitarra','Tai chi','Washing hands','Vacuuming floor','Waxing skis','Doing step aerobics','Putting on makeup']
easy_and_hard = easy_class + hard_class
common_class = set(easy_and_hard) ^ set(activity_dict.keys())
common_class_list = list(common_class)

rare_id = [activity_dict[hard_class[i]] for i in range(len(hard_class))]
common_id = [activity_dict[easy_class[i]] for i in range(len(easy_class))]
freq_id = [activity_dict[common_class_list[i]] for i in range(len(common_class_list))]

freq_dict = {'rare': rare_id, 'common': freq_id, 'freq': common_id}

class ACSL(nn.Module):

    def __init__(self, score_thr=0.3, loss_weight=1.0):

        super(ACSL, self).__init__()

        self.score_thr = score_thr
        assert self.score_thr > 0 and self.score_thr < 1
        self.loss_weight = loss_weight

        self.freq_group = freq_dict

    def forward(self, cls_logits_, labels_, weight=None, avg_factor=None, reduction_override=None, **kwargs):

        device = cls_logits_.device
        
        self.n_i, self.n_c, _ = cls_logits_.size()
        cls_loss = 0
        for snip_id in range(temporal_scale):
            cls_logits = cls_logits_[:,:,snip_id]  # batch x class
            labels = labels_[:,:,snip_id]
            
            # expand the labels to all their parent nodes
            target = cls_logits.new_zeros(self.n_i, self.n_c)
         
            labels = torch.argmax(labels,dim=1)
            unique_label = torch.unique(labels)
            
            # print(unique_label)
            with torch.no_grad():
                sigmoid_cls_logits = torch.sigmoid(cls_logits)
            # for each sample, if its score on unrealated class hight than score_thr, their gradient should not be ignored
            # this is also applied to negative samples
            high_score_inds = torch.nonzero(sigmoid_cls_logits>=self.score_thr)
            weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), cls_logits.new_ones(high_score_inds.shape[0]), size=(self.n_i, self.n_c), device=device).to_dense()
            # print(weight_mask.size())
            for cls in unique_label:
                cls = cls.item()
                # print(cls)
                cls_inds = torch.nonzero(labels == cls).squeeze(1)
                # print(cls_inds.size())
                if cls == num_classes:
                    # construct target vector for background samples
                    target[cls_inds, num_classes] = 1
                    # for bg, set the weight of all classes to 1
                    weight_mask[cls_inds] = 0

                    cls_inds_cpu = cls_inds.cpu()

                    # Solve the rare categories, random choost 1/3 bg samples to suppress rare categories
                    rare_cats = self.freq_group['rare']
                    rare_cats = torch.tensor(rare_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 0.01)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, rare_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask

                    # Solve the common categories, random choost 2/3 bg samples to suppress rare categories
                    common_cats = self.freq_group['common']
                    common_cats = torch.tensor(common_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 0.1)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, common_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask
                    
                    # Solve the frequent categories, random choost all bg samples to suppress rare categories
                    freq_cats = self.freq_group['freq']
                    freq_cats = torch.tensor(freq_cats, device=cls_logits.device)
                    choose_bg_num = int(len(cls_inds) * 1.0)
                    choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

                    tmp_weight_mask = weight_mask[choose_bg_inds]
                    tmp_weight_mask[:, freq_cats] = 1

                    weight_mask[choose_bg_inds] = tmp_weight_mask

                    # Set the weight for bg to 1
                    weight_mask[cls_inds, num_classes] = 1
                    
                else:
                    # construct target vector for foreground samples
                    cur_labels = [cls]
                    cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
                    tmp_label_vec = cls_logits.new_zeros(self.n_c)
                    tmp_label_vec[cur_labels] = 1
                    tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
                    target[cls_inds] = tmp_label_vec
                    # construct weight mask for fg samples
                    tmp_weight_mask_vec = weight_mask[cls_inds]
                    # set the weight for ground truth category
                    tmp_weight_mask_vec[:, cur_labels] = 1

                    weight_mask[cls_inds] = tmp_weight_mask_vec

            cls_loss+= F.binary_cross_entropy_with_logits(cls_logits, target.float(), reduction='none')

        return torch.sum(weight_mask * cls_loss) / (self.n_i*temporal_scale)


def top_lr_loss(target,pred):

    gt_action = target
    pred_action = pred
    topratio = 0.6
    alpha = 10

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 

    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    eps = 0.000001
    pred_p = torch.log(pred_action + eps)
    pred_n = torch.log(1.0 - pred_action + eps)


    topk = int(num_classes * topratio)
    # targets = targets.cuda()
    count_pos = num_positive
    hard_neg_loss = -1.0 * (1.0-gt_action) * pred_n
    topk_neg_loss = -1.0 * hard_neg_loss.topk(topk, dim=1)[0]#topk_neg_loss with shape batchsize*topk

    loss = (gt_action * pred_p).sum() / count_pos + alpha*(topk_neg_loss.cuda()).mean()

    return -1*loss


class BinaryDiceLoss(nn.Module):
   
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

dice = BinaryDiceLoss()
acsl = ACSL()


def top_ce_loss(gt_cls, pred_cls, nm=False):

    ce_loss = F.cross_entropy(pred_cls,gt_cls)
    pt = torch.exp(-ce_loss)
    if nm:
        focal_loss = ((1 - pt) **2 * ce_loss)
    else:
        focal_loss = ((1 - pt) **2 * ce_loss).mean()
    loss = focal_loss.mean() 

    return loss


def bottom_branch_loss(gt_action, pred_action, f_loss=False, cross_entropy_loss=False, parabola=False): # gt action and pred action are both shape [256, temporal_scale, temporal_scale] 

    pmask = (gt_action == 1).float()
    nmask = (gt_action == 0).float()
    nmask = nmask 
    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_action + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_action + epsilon) * nmask
    w_bce_loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    BCE_loss = F.binary_cross_entropy(pred_action,gt_action,reduce=False)
    pt = torch.exp(-BCE_loss)
    # F_loss = 0.4*loss2 + 0.6*dice(pred_action,gt_action)
    F_loss = lambda_2*w_bce_loss + (1 - lambda_2)*dice(pred_action,gt_action)

    #breakpoint()

    #print("losses", w_bce_loss, dice(pred_action, gt_action))

    if f_loss:
        #print("F_loss")
        return F_loss
    elif cross_entropy_loss:
        return bce(pred_action, gt_action) 
    elif parabola:
        return torch.mean(torch.pow(gt_action-pred_action, 2))
    else:
        #print("MSELoss")
        return mse(gt_action, pred_action) #mse(gt_action, pred_action)
    
    #return F_loss #nn.MSELoss()(gt_action, pred_action) #dice(pred_action, gt_action) # F_loss, could try MSE or L1 # HACK, as just using MSELoss seems to improve the results a bit and let the model improve. However, it would be best to get the author's goal function working.


def top_branch_loss(gt_cls, pred_cls, mask_gt, nll_loss=False):
    #breakpoint()
    if nll_loss:
        return nll(nn.LogSoftmax(dim=1)(pred_cls), gt_cls.cuda()) #nn.MSELoss()(gt_cls, pred_cls) # lambda_1*top_ce_loss(gt_cls.cuda(), pred_cls)
    else:
        return ce(pred_cls, gt_cls.cuda().to(int))


def spot_loss(gt_cls, pred_cls ,gt_action , pred_action, mask_gt, label_gt, pretrain=False):
    

    if pretrain:
        bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action)
        return bottom_loss
    else:
        #breakpoint()
        top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
        bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) # NOTE: for ANet, I keep f_loss=true. 
        tot_loss = top_loss + bottom_loss
        return tot_loss, top_loss, bottom_loss


# def dynamic_thres()

def spot_loss_bot(gt_cls, pred_cls ,gt_action , pred_action, mask_gt , label_gt):

    
    top_loss = top_branch_loss(gt_cls, pred_cls, mask_gt)
    bottom_loss = bottom_branch_loss(gt_action.cuda(), pred_action) 

    tot_loss = bottom_loss 
    top_loss = 0

    return tot_loss, top_loss, bottom_loss


def ce_loss_thresh(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
