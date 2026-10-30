#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataloader import TestData_Vis, TestData_Inf
from datamanager import (process_gallery_sysu, process_query_sysu,
                         process_test_regdb, process_query_llcm,
                         process_gallery_llcm)
from datamanager import VCM, VideoDataset_test_Inf, VideoDataset_test_Vis
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
import gc
from model.make_model import build_vision_transformer
from utils import set_seed, Logger
from transforms import transform_test
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb
import argparse

parser = argparse.ArgumentParser(description="Evaluation-only for a saved checkpoint")
parser.add_argument('--ckpt', required=True, type=str,
                    help='path to the .pth checkpoint to evaluate')
parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                    help='1: regdb->sysu->llcm->vcm  2: vcm->llcm->sysu->regdb')
parser.add_argument('--up_to', type=str, default='',
                    help='evaluate only up to this dataset (default: all four)')
parser.add_argument('--trial', default=1, type=int)
parser.add_argument('--test_batch', default=128, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--logs-dir', default='', type=str,
                    help='if set, tee stdout to <logs-dir>/eval_log.txt')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.logs_dir:
    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.logs_dir, 'eval_log.txt'))

cfg.merge_from_list(args.opts)
cfg.freeze()
set_seed(cfg.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.setting == 1:
    training_set = ['regdb', 'sysu', 'llcm', 'vcm']
else:
    training_set = ['vcm', 'llcm', 'sysu', 'regdb']

if args.up_to:
    idx = training_set.index(args.up_to)
else:
    idx = len(training_set) - 1

print('==========\nArgs:{}\n=========='.format(args))
print('[INFO] Order: ' + ' -> '.join(training_set))
print('[INFO] Evaluating on: ' + ', '.join(training_set[:idx + 1]))

# ---- Build model; classifier size is read from the checkpoint so that
# ---- load_param does not have to skip it (features are what we need).
param_dict = torch.load(args.ckpt, map_location='cpu')
num_classes = None
for key in param_dict:
    if 'classifier.weight' in key:
        num_classes = param_dict[key].shape[0]
        break
if num_classes is None:
    raise RuntimeError('classifier.weight not found in {}'.format(args.ckpt))
print('[INFO] Checkpoint classifier num_classes = {}'.format(num_classes))
del param_dict

model = build_vision_transformer(num_classes=num_classes, cfg=cfg)
model.load_param(args.ckpt)
model.to(device)
model.eval()

# ---- Test loaders ----
# Build and release one dataset's loaders at a time to reduce peak memory use.
def build_loaders(name):
    """Return (query_loader, gall_loader, query_label, gall_label, query_cam, gall_cam)."""
    if name == 'sysu':
        data_path = cfg.DATA_PATH_SYSU
        query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
        queryset = TestData_Inf(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
        gallset = TestData_Vis(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    elif name == 'regdb':
        data_path = cfg.DATA_PATH_RegDB
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
        queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
        gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        query_cam, gall_cam = None, None
    elif name == 'llcm':
        data_path = cfg.DATA_PATH_LLCM
        query_img, query_label, query_cam = process_query_llcm(data_path, modal=2)
        queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, modal=1)
        gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    elif name == 'vcm':
        processed_data = VCM()
        queryset = VideoDataset_test_Inf(processed_data.query, 1, "video_test", transform_test, processed_data.query_cam)
        gallset = VideoDataset_test_Vis(processed_data.gallery, 1, "video_test", transform_test, processed_data.gallary_cam)
        queryset.test_label = processed_data.query_labels
        gallset.test_label = processed_data.gallery_labels
        query_cam, gall_cam = None, None
    else:
        raise ValueError('unknown dataset: {}'.format(name))

    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
    return (query_loader, gall_loader, queryset.test_label, gallset.test_label,
            query_cam if name == 'sysu' else None,
            gall_cam if name == 'sysu' else None)


def test(model, query_loader, gall_loader, dataset='sysu',
         query_label=None, gall_label=None, query_cam=None, gall_cam=None):
    model.eval()
    print('[INFO] Testing...')

    gall_feat = []
    with torch.no_grad():
        for input, label, cam in gall_loader:
            input = Variable(input.cuda())
            feat = model(input, cam)
            gall_feat.append(feat.detach().cpu().numpy())
    gall_feat = np.concatenate(gall_feat, axis=0)

    query_feat = []
    with torch.no_grad():
        for input, label, cam in query_loader:
            input = Variable(input.cuda())
            feat = model(input, cam)
            query_feat.append(feat.detach().cpu().numpy())
    query_feat = np.concatenate(query_feat, axis=0)

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    if dataset == 'sysu':
        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    else:
        cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)
    return cmc, mAP, mInp


head_str, results_str, copy_str = '|', '|', ''
mean_R1, mean_mAP = 0, 0
for ii, testset_name in enumerate(training_set):
    if ii > idx:
        continue
    print('[INFO] Building loaders for {} ...'.format(testset_name))
    q_loader, g_loader, q_label, g_label, q_cam, g_cam = build_loaders(testset_name)
    cmc, mAP, mINP = test(model, q_loader, g_loader,
                          testset_name, q_label, g_label, q_cam, g_cam)
    # release this dataset's in-RAM image arrays before building the next one
    del q_loader, g_loader, q_label, g_label, q_cam, g_cam
    gc.collect()
    head_str += testset_name + '|\t'
    results_str += '{:.2f}/{:.2f}|\t'.format(mAP * 100, cmc[0] * 100)
    copy_str += '{:.2f}\t{:.2f}\t'.format(mAP * 100, cmc[0] * 100)
    mean_R1 += cmc[0]
    mean_mAP += mAP

mean_R1 /= (idx + 1)
mean_mAP /= (idx + 1)
head_str += 'AVG|\t'
results_str += '{:.2f}/{:.2f}|\t'.format(mean_mAP * 100, mean_R1 * 100)
copy_str += '{:.2f}\t{:.2f}'.format(mean_mAP * 100, mean_R1 * 100)

print("\nResults:")
print(head_str)
print(results_str)
print(copy_str)
