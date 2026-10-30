import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_old_proto(train_loader, model):
    
    model.eval()
    
    feat_vis = []
    feat_inf = []
    id_vis = []
    id_inf = []

    with torch.no_grad():
        for inputs, labels, mods in train_loader:
            inputs = Variable(inputs.cuda())
            feats = model(inputs, mods)

            for feat, label, mod in zip(feats, labels, mods):
                if mod == 1:
                    feat_vis.append(feat)
                    id_vis.append(label.item())
                else:
                    feat_inf.append(feat)
                    id_inf.append(label.item())
    
    feat_vis_collect = {}
    for feature, label in zip(feat_vis, id_vis):
        if label in feat_vis_collect:
            feat_vis_collect[label].append(feature)
        else:
            feat_vis_collect[label] = [feature]

    feat_inf_collect = {}
    for feature, label in zip(feat_inf, id_inf):
        if label in feat_inf_collect:
            feat_inf_collect[label].append(feature)
        else:
            feat_inf_collect[label] = [feature]
    
    vis_labels_named = list(set(id_vis))  # obtain valid features
    vis_labels_named.sort()
    vis_features_mean=[]
    for x in vis_labels_named:
        if x in feat_vis_collect.keys():
            feats=torch.stack(feat_vis_collect[x])
            feat_mean=feats.mean(dim=0)
            vis_features_mean.append(feat_mean)
        else:
            print ("Error, unexpected ID")
        
    inf_labels_named = list(set(id_inf))  # obtain valid features
    inf_labels_named.sort()
    inf_features_mean=[]
    for x in inf_labels_named:
        if x in feat_inf_collect.keys():
            feats=torch.stack(feat_inf_collect[x])
            feat_mean=feats.mean(dim=0)
            inf_features_mean.append(feat_mean)
        else:
            print ("Error, unexpected ID")

    return torch.stack(vis_features_mean), vis_labels_named, torch.stack(inf_features_mean), inf_labels_named

def get_normal_affinity(x,Norm=100):
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix

def cosine_similarity(input1, input2):
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat


##################################################################################################################################
# Loggers

import os
import sys
import errno

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
