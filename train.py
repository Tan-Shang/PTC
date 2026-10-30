#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PTC training with torch.nn.DataParallel. See README.md for configuration.
"""

# ============== Imports ==============
from dataloader import SYSUData, RegDBData, LLCMData, VCMData, TrainData, TestData_Vis, TestData_Inf, GenIdx, IdentitySampler
from datamanager import process_gallery_sysu, process_query_sysu, process_test_regdb, process_query_llcm, process_gallery_llcm, process_query_vcm, process_gallery_vcm, process_train_regdb, process_train_sysu, process_train_llcm, process_train_vcm
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
from torch.cuda import amp
import torch.nn as nn
import os.path as osp
import os
from model.make_model import build_vision_transformer
import time
from scheduler import create_scheduler
from loss.Triplet import TripletLoss
from loss.TripletCross import TripletCrossLoss
from loss.s_ptc import compute_s_ptc_loss
from loss.cm_ptc import compute_cm_ptc_loss
from utils import AverageMeter, set_seed, get_old_proto, get_normal_affinity, cosine_similarity
from transforms import transform_rgb, transform_rgb2gray, transform_thermal, transform_test
from optimizer import make_optimizer
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb
import argparse
from datamanager import VCM, VideoDataset_test_Inf, VideoDataset_test_Vis
import sys
from utils import Logger


def load_state_dict_flexible(model, state_dict):
    """
    Flexibly load state_dict, handling module. prefix differences
    and shape mismatches (e.g., classifier layers in continual learning)
    
    This is needed when loading weights saved from DataParallel model,
    and when the number of classes changes between stages.
    """
    model_state = model.state_dict()
    
    # Check if we need to add or remove 'module.' prefix
    model_has_module = any(k.startswith('module.') for k in model_state.keys())
    ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
    
    new_state_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        # Adjust key based on module. prefix
        if model_has_module and not ckpt_has_module:
            new_key = 'module.' + k
        elif not model_has_module and ckpt_has_module:
            new_key = k.replace('module.', '')
        else:
            new_key = k
        
        if new_key in model_state:
            # Check if shapes match
            if model_state[new_key].shape == v.shape:
                new_state_dict[new_key] = v
            else:
                # Shape mismatch - skip this layer (typically classifier)
                skipped_keys.append(new_key)
                print(f'[WARNING] Shape mismatch, skipping: {new_key} '
                      f'(checkpoint: {v.shape} vs model: {model_state[new_key].shape})')
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f'[INFO] Loaded {len(new_state_dict)}/{len(model_state)} parameters')
    if skipped_keys:
        print(f'[INFO] Skipped {len(skipped_keys)} layers due to shape mismatch (will be randomly initialized)')
    return model


# ============== Argument Parser ==============
parser = argparse.ArgumentParser(description="PTC training")
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--num_workers', default=8,
                    help='number of data loading workers', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
parser.add_argument("--logs-dir", help="Log directory",
                    default='runs/ptc', type=str)
parser.add_argument('--setting', type=int, default=1, choices=[1], 
                    help="training order setting")
parser.add_argument('--new_weight', type=float, default=1.0)
parser.add_argument('--cross_weight', type=float, default=0.5)
parser.add_argument('--lambda_topo', type=float, default=0.1,
                    help='weight for topological relational distillation loss')
parser.add_argument('--lambda_cm_ptc', type=float, default=0.05,
                    help='weight for Cross-Modal Prototype Topology Consistency (CM-PTC) loss')
parser.add_argument('--topo_temperature', type=float, default=0.05,
                    help='temperature for topo loss softmax')
parser.add_argument('--tau_cm', type=float, default=0.2,
                help='temperature for CM-PTC cross-modal topology')
parser.add_argument('--start_stage', type=str, default='',
                    help='Start training from this dataset (skip previous stages). '
                         'Options: regdb, sysu, llcm, vcm')

args = parser.parse_args()
if args.topo_temperature <= 0:
    parser.error('S-PTC temperature must be positive.')

# ============== Setup ==============
if os.path.exists(args.logs_dir) is False:
    os.makedirs(args.logs_dir)
log_name = 'log.txt'
sys.stdout = Logger(osp.join(args.logs_dir, log_name))

cfg.merge_from_list(args.opts)
cfg.freeze()
print("==========\nArgs:{}\n==========".format(args))

set_seed(cfg.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f'[INFO] Number of available GPUs: {num_gpus}')
print(f'[INFO] Using DataParallel for multi-GPU training')
print(
    f'[INFO] Batch size: {cfg.BATCH_SIZE} per modality, '
    f'{2 * cfg.BATCH_SIZE} in total'
)

torch.cuda.empty_cache()

if args.setting == 1:
    training_set = ['regdb', 'sysu', 'llcm', 'vcm']

# Handle --start_stage: skip stages before the specified one
start_idx = 0
if args.start_stage:
    if args.start_stage in training_set:
        start_idx = training_set.index(args.start_stage)
        print(f'[INFO] Starting from stage: {args.start_stage} (index {start_idx})')
        print(f'[INFO] Skipping stages: {training_set[:start_idx]}')
    else:
        print(f'[WARNING] Invalid start_stage: {args.start_stage}, starting from beginning')

# ============== Main Training Loop ==============
for idx, dataset_name in enumerate(training_set):
    # Skip stages before start_idx
    if idx < start_idx:
        print(f'[INFO] Skipping stage {idx}: {dataset_name} (already completed)')
        continue
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(training_set)}] Training on: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    if dataset_name == 'sysu':
        data_path = cfg.DATA_PATH_SYSU
        trainset_rgb = SYSUData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'regdb':
        data_path = cfg.DATA_PATH_RegDB
        trainset_rgb = RegDBData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'llcm':
        data_path = cfg.DATA_PATH_LLCM
        trainset_rgb = LLCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'vcm':
        data_path = cfg.DATA_PATH_VCM
        trainset_rgb = VCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    
    num_classes = len(np.unique(trainset_rgb.train_color_label))
    print(f'[INFO] Dataset: {dataset_name}, classes: {num_classes}')
    
    # Build model
    model = build_vision_transformer(num_classes=num_classes, cfg=cfg)
    model.to(device)
    
    # Wrap with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f'[INFO] Model wrapped with DataParallel on {num_gpus} GPUs')
    
    # Load previous model for continual learning
    if idx > 0:
        args.resume = training_set[idx-1] + '.pth'
        model_path = os.path.join(args.logs_dir, args.resume)
        
        # Load parameters flexibly (handle module. prefix)
        state_dict = torch.load(model_path, map_location=device)
        model = load_state_dict_flexible(model, state_dict)
        print(f'[INFO] Loaded old model: {args.resume}')
        
        # Get old model's number of classes
        for key in state_dict:
            if 'classifier.weight' in key:
                old_num_classes = state_dict[key].shape[0]
                break
        
        # Build old model for knowledge distillation
        old_model = build_vision_transformer(num_classes=old_num_classes, cfg=cfg)
        old_model.to(device)
        
        # Wrap old_model with DataParallel
        if num_gpus > 1:
            old_model = nn.DataParallel(old_model)
        
        # Load old model weights
        old_model = load_state_dict_flexible(old_model, state_dict)
        old_model.eval()
        print(f'[INFO] Created frozen old model for KD')
        
        # Load prototype features
        old_proto = torch.load(model_path.replace('.pth', '_proto.pth'), map_location=device)
        vis_features_mean = old_proto['vis_features_mean']
        vis_labels_named = old_proto['vis_labels_named']
        inf_features_mean = old_proto['inf_features_mean']
        inf_labels_named = old_proto['inf_labels_named']
        
        # Move to GPU if needed
        if isinstance(vis_features_mean, torch.Tensor):
            vis_features_mean = vis_features_mean.to(device)
            inf_features_mean = inf_features_mean.to(device)
    else:
        old_model = None
    
    # Loss functions
    criterion_ID = nn.CrossEntropyLoss()
    criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
    criterion_TriCros = TripletCrossLoss(margin=cfg.MARGINCROSS, feat_norm='no')
    criterion_TriCent = TripletCrossLoss(margin=cfg.MARGINCENTER, feat_norm='no')
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    
    # Optimizer - need to handle DataParallel
    if isinstance(model, nn.DataParallel):
        optimizer = make_optimizer(cfg, model.module)
    else:
        optimizer = make_optimizer(cfg, model)
    
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler()
    
    # Prepare test loaders
    query_loaders, gall_loaders, query_labels, gall_labels, query_cams, gall_cams = [], [], [], [], [], []
    for ii in range(idx + 1):
        if training_set[ii] == 'sysu':
            data_path = cfg.DATA_PATH_SYSU
            query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
            queryset = TestData_Inf(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
            gallset = TestData_Vis(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'regdb':
            data_path = cfg.DATA_PATH_RegDB
            query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            query_cam, gall_cam = None, None
        elif training_set[ii] == 'llcm':
            data_path = cfg.DATA_PATH_LLCM
            query_img, query_label, query_cam = process_query_llcm(data_path, modal=2)
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, modal=1)
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'vcm':
            data_path = cfg.DATA_PATH_VCM
            processed_data = VCM()
            queryset = VideoDataset_test_Inf(processed_data.query, 1, "video_test", transform_test, processed_data.query_cam)
            gallset = VideoDataset_test_Vis(processed_data.gallery, 1, "video_test", transform_test, processed_data.gallary_cam)
            queryset.test_label = processed_data.query_labels
            gallset.test_label = processed_data.gallery_labels
            query_cam, gall_cam = None, None
        
        query_loaders.append(data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers))
        gall_loaders.append(data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers))
        query_labels.append(queryset.test_label)
        gall_labels.append(gallset.test_label)
        query_cams.append(query_cam if training_set[ii] == 'sysu' else None)
        gall_cams.append(gall_cam if training_set[ii] == 'sysu' else None)
    
    # ============== Training Function ==============
    def train(epoch, model, scheduler, optimizer, scaler):
        loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_tri_meter = AverageMeter()
        loss_topo_meter = AverageMeter()
        loss_cm_ptc_meter = AverageMeter()
        acc_rgb_meter = AverageMeter()
        acc_ir_meter = AverageMeter()
        
        loss_meter.reset()
        loss_ce_meter.reset()
        loss_tri_meter.reset()
        loss_topo_meter.reset()
        loss_cm_ptc_meter.reset()
        acc_rgb_meter.reset()
        acc_ir_meter.reset()
        
        scheduler.step(epoch)
        model.train()
        
        for idx_, (input1, input2, label1, label2) in enumerate(trainloader):
            optimizer.zero_grad()
            input1 = input1.to(device)
            input2 = input2.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            labels = torch.cat((label1, label2), 0)
            
            with amp.autocast(enabled=True):
                # Forward pass - DataParallel handles distribution automatically
                scores, feats = model(
                    torch.cat([input1, input2]), 
                    torch.cat([torch.ones_like(label1), torch.zeros_like(label2)])
                )
                
                score1, score2 = scores.chunk(2, 0)
                feat1, feat2 = feats.chunk(2, 0)
                loss_id = criterion_ID(score1, label1.long()) + criterion_ID(score2, label2.long())
                
                loss_tri = criterion_Tri(feats, feats, labels)
                loss = loss_id + loss_tri
                
                feat1_norm = feat1
                feat2_norm = feat2
                
                def merge_feat(features_all, labels_all):
                    features_collect = {}
                    for feature, label in zip(features_all, labels_all):
                        label = label.cpu().item()
                        if label in features_collect:
                            features_collect[label].append(feature)
                        else:
                            features_collect[label] = [feature]
                    
                    labels_named = list(set(labels_all))
                    labels_named.sort()
                    
                    features_mean = []
                    for x in labels_named:
                        x = x.cpu().item()
                        if x in features_collect.keys():
                            feats = torch.stack(features_collect[x])
                            feat_mean = feats.mean(dim=0)
                            features_mean.append(feat_mean)
                        else:
                            features_mean.append(torch.zeros_like(features_all[0]))
                    
                    return torch.stack(features_mean), labels_named
                
                feat1_mean, feat1_mean_label = merge_feat(feat1_norm, label1)
                feat2_mean, feat2_mean_label = merge_feat(feat2_norm, label2)
                
                loss_tri_cent = 0.5 * criterion_TriCent(feat1_mean, feat2_mean, feat1_mean_label, feat2_mean_label) + \
                                0.5 * criterion_TriCent(feat2_mean, feat1_mean, feat2_mean_label, feat1_mean_label)
                loss_tri_cross = 0.5 * criterion_TriCros(feat1_norm, feat2_mean, label1, feat2_mean_label) + \
                                 0.5 * criterion_TriCros(feat1_mean, feat2_norm, feat1_mean_label, label2)
                
                loss += args.new_weight * ((1 - args.cross_weight) * loss_tri_cent + args.cross_weight * loss_tri_cross)
                
                # --- Stage-wise prototype topology consistency (stage > 0) ---
                loss_topo = torch.tensor(0.0, device=device)
                if old_model is not None:
                    old_model.eval()
                    with torch.no_grad():
                        feats_old_norm = old_model(
                            torch.cat([input1, input2]), 
                            torch.cat([torch.ones_like(label1), torch.zeros_like(label2)]), 
                            fkd=True
                        )
                        feats_old_norm = feats_old_norm.detach()
                        feat1_old, feat2_old = feats_old_norm.chunk(2, 0)
                    
                    # Compute in fp32 to avoid fp16 underflow in KL divergence
                    # Same-modality topo matching only:
                    #   feat1 (RGB/vis) x vis_features_mean (vis prototypes)
                    #   feat2 (IR/inf)  x inf_features_mean (inf prototypes)
                    with torch.cuda.amp.autocast(enabled=False):
                        topo_vis = compute_s_ptc_loss(
                            feat1.float(), feat1_old.float(), vis_features_mean.float(),
                            args.topo_temperature, KLDivLoss)
                        topo_inf = compute_s_ptc_loss(
                            feat2.float(), feat2_old.float(), inf_features_mean.float(),
                            args.topo_temperature, KLDivLoss)
                    loss_topo = 0.5 * topo_vis + 0.5 * topo_inf
                    loss += args.lambda_topo * loss_topo

                # --- Cross-Modal Prototype Topology Consistency (CM-PTC) ---
                # Goal: align the *topology* (pairwise prototype similarities) across RGB and IR.
                # We form *dynamic batch prototypes* per identity for each modality, build
                # modality-specific topology matrices, then apply symmetric KL divergence.
                loss_cm_ptc = torch.tensor(0.0, device=device)
                if args.lambda_cm_ptc != 0:
                    batch_size = label1.size(0)
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_cm_ptc = compute_cm_ptc_loss(
                            feats, labels, batch_size, args.tau_cm)

                    # 5) Weighted sum into total loss.
                    loss = loss + args.lambda_cm_ptc * loss_cm_ptc
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc_rgb = (score1.max(1)[1] == label1).float().mean()
            acc_ir = (score2.max(1)[1] == label2).float().mean()
            
            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())
            loss_topo_meter.update(loss_topo.item() if isinstance(loss_topo, torch.Tensor) else loss_topo)
            loss_cm_ptc_meter.update(loss_cm_ptc.item() if isinstance(loss_cm_ptc, torch.Tensor) else loss_cm_ptc)
            
            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)
            
            torch.cuda.synchronize()
            
            if (idx_ + 1) % 100 == 0 or (idx_ + 1) == len(trainloader):
                print(f'Epoch[{epoch}] Iteration[{idx_+1}/{len(trainloader)}] '
                      f'Loss: {loss_meter.avg:.3f}, Tri:{loss_tri_meter.avg:.3f} CE:{loss_ce_meter.avg:.3f}, '
                      f'Topo:{loss_topo_meter.avg:.3f}, CM-PTC:{loss_cm_ptc_meter.avg:.3f}, '
                      f'Acc_RGB: {acc_rgb_meter.avg:.3f}, Acc_IR: {acc_ir_meter.avg:.3f}, '
                      f'Base Lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.2e}')
        
        return model
    
    # ============== Test Function ==============
    def test(model, query_loader, gall_loader, dataset='sysu', 
             query_label=None, gall_label=None, query_cam=None, gall_cam=None):
        model.eval()
        print('[INFO] Testing...')
        
        gall_feat = []
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                # Handle DataParallel - use module for inference
                if isinstance(model, nn.DataParallel):
                    feat = model.module(input, cam)
                else:
                    feat = model(input, cam)
                gall_feat.append(feat.detach().cpu().numpy())
        gall_feat = np.concatenate(gall_feat, axis=0)
        
        query_feat = []
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                if isinstance(model, nn.DataParallel):
                    feat = model.module(input, cam)
                else:
                    feat = model(input, cam)
                query_feat.append(feat.detach().cpu().numpy())
        query_feat = np.concatenate(query_feat, axis=0)
        
        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        if dataset == 'sysu':
            cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        else:
            cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)
        
        return cmc, mAP, mInp
    
    # ============== Training Loop ==============
    best_mAP = 0
    print('==> Start Training...')
    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):
        sampler_rgb = IdentitySampler(
            trainset_rgb.train_color_label, 
            trainset_rgb.train_thermal_label,
            color_pos_rgb, thermal_pos_rgb, 
            cfg.BATCH_SIZE, per_img=cfg.NUM_POS
        )
        
        trainset_rgb.cIndex = sampler_rgb.index1
        trainset_rgb.tIndex = sampler_rgb.index2
        trainloader = data.DataLoader(
            trainset_rgb, 
            batch_size=cfg.BATCH_SIZE, 
            sampler=sampler_rgb, 
            num_workers=args.num_workers, 
            drop_last=True, 
            pin_memory=True
        )
        
        model = train(epoch, model, scheduler, optimizer, scaler)
        
        # Final epoch: save and evaluate
        if epoch == cfg.MAX_EPOCH:
            # Merge model parameters (Dynamic EMA) if not first dataset
            if idx > 0:
                model_path = os.path.join(args.logs_dir, args.resume)
                dynamic_ema_alpha = idx / (idx + 1)  # Stage2: 0.5, Stage3: 0.667, Stage4: 0.75
                if isinstance(model, nn.DataParallel):
                    model.module.merge_param(model_path, dynamic_ema_alpha)
                else:
                    model.merge_param(model_path, dynamic_ema_alpha)
                print(f'[INFO] Dynamic EMA: alpha={dynamic_ema_alpha:.4f} '
                      f'(old_weight={dynamic_ema_alpha:.4f}, new_weight={1-dynamic_ema_alpha:.4f})')
            
            # Save model - unwrap DataParallel before saving
            save_path = osp.join(args.logs_dir, training_set[idx] + '.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'[INFO] Saved model to {save_path}')
            
            # Extract and save prototypes
            if dataset_name == 'regdb':
                train_img, train_label, train_mod = process_train_regdb(data_path)
            elif dataset_name == 'sysu':
                train_img, train_label, train_mod = process_train_sysu(data_path)
            elif dataset_name == 'llcm':
                train_img, train_label, train_mod = process_train_llcm(data_path)
            elif dataset_name == 'vcm':
                train_img, train_label, train_mod = process_train_vcm(data_path)
            
            trainset = TrainData(train_img, train_label, train_mod, transform=transform_test, img_size=(cfg.W, cfg.H))
            train_loader = data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=8)
            
            # Get prototypes - need to handle DataParallel
            if isinstance(model, nn.DataParallel):
                vis_features_mean, vis_labels_named, inf_features_mean, inf_labels_named = get_old_proto(train_loader, model.module)
            else:
                vis_features_mean, vis_labels_named, inf_features_mean, inf_labels_named = get_old_proto(train_loader, model)
            
            proto_type = {
                'vis_features_mean': vis_features_mean,
                'vis_labels_named': vis_labels_named,
                'inf_features_mean': inf_features_mean,
                'inf_labels_named': inf_labels_named
            }
            proto_path = osp.join(args.logs_dir, dataset_name + '_proto.pth')
            torch.save(proto_type, proto_path)
            print(f'[INFO] Saved prototypes to {proto_path}')
            
            # Evaluate on all learned datasets
            R1_list, mAP_list = [], []
            head_str, results_str, copy_str = '|', '|', ''
            mean_R1, mean_mAP = 0, 0
            
            for ii, testset_name in enumerate(training_set):
                if ii > idx:
                    continue
                cmc, mAP, mINP = test(model, query_loaders[ii], gall_loaders[ii], 
                                      testset_name, query_labels[ii], gall_labels[ii], 
                                      query_cams[ii], gall_cams[ii])
                R1_list.append(cmc[0])
                mAP_list.append(mAP)
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

print('\n[INFO] Training completed!')
