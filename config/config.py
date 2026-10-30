from yacs.config import CfgNode as CN
import os
cfg = CN()

cfg.SEED = 0

# dataset
data_root = os.path.abspath(os.environ.get('PTC_DATA_ROOT', 'datasets'))
cfg.DATA_PATH_RegDB = os.path.join(data_root, 'RegDB', '')
cfg.DATA_PATH_SYSU = os.path.join(data_root, 'SYSU-MM01', '')
cfg.DATA_PATH_LLCM = os.path.join(data_root, 'LLCM', '')
cfg.DATA_PATH_VCM = os.path.join(data_root, 'HITSZ-VCM', '')
cfg.PRETRAIN_PATH = os.environ.get('PTC_PRETRAIN_PATH', 'jx_vit_base_p16_224-80ecf9dd.pth')

cfg.START_EPOCH = 1
cfg.MAX_EPOCH = 50

cfg.H = 256
cfg.W = 128
cfg.BATCH_SIZE = 32  # num of images for each modality in a mini batch
cfg.NUM_POS = 4

cfg.MARGIN = 0.1    # margin for triplet
cfg.MARGINCROSS = 0.1
cfg.MARGINCENTER = 0.1


# model
cfg.STRIDE_SIZE =  [12, 12]
cfg.DROP_OUT = 0.03
cfg.ATT_DROP_RATE = 0.0
cfg.DROP_PATH = 0.1

# optimizer
cfg.OPTIMIZER_NAME = 'AdamW'  # AdamW or SGD
cfg.MOMENTUM = 0.9    # for SGD

cfg.BASE_LR = 3e-4
cfg.WEIGHT_DECAY = 1e-4
cfg.WEIGHT_DECAY_BIAS = 1e-4
cfg.BIAS_LR_FACTOR = 1

cfg.LR_PRETRAIN = 0.5
cfg.LR_MIN = 0.01
cfg.LR_INIT = 0.01
cfg.WARMUP_EPOCHS = 3








