# Prototype Topology Consistency for Visible-Infrared Lifelong Person Re-Identification

## Introduction

Implementation of PTC (NeurIPS 2026) for visible-infrared lifelong person re-identification. S-PTC preserves historical prototype assignments across stages, while CM-PTC aligns inter-class prototype topology between visible and infrared modalities.

## Installation

Install a CUDA-enabled PyTorch and matching torchvision, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Download [RegDB](http://dm.dongguk.edu/link.html), [SYSU-MM01](https://github.com/wuancong/SYSU-MM01), [LLCM](https://github.com/ZYK100/LLCM), and [HITSZ-VCM](https://github.com/VCM-project233/HITSZ-VCM-data), and arrange them as follows:

```text
datasets/
├── RegDB/
├── SYSU-MM01/
├── LLCM/
└── HITSZ-VCM/
```

Set `PTC_DATA_ROOT` to the dataset root and `PTC_PRETRAIN_PATH` to the pretrained ViT checkpoint (`jx_vit_base_p16_224-80ecf9dd.pth`). Use `pre_process_sysu.py`, `pre_process_llcm.py`, and `pre_process_vcm.py` to prepare the datasets before training.

## Usage

Train and evaluate in the order RegDB → SYSU-MM01 → LLCM → VCM:

```bash
bash run.sh
```

Evaluate a trained checkpoint:

```bash
python eval_ckpt.py --ckpt runs/ptc/vcm.pth --setting 1
```
