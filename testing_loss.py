# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import importlib

from PIL import Image
from torch import nn, optim
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import datasets
import transforms
import constants

# new 
import torch.nn as nn
import torch.nn.functional as F
import pickle as pickle
from tqdm import tqdm 
import torchvision.models as models
import torchvision.transforms as ttransforms
#import torchvision.datasets as datasets
import matplotlib.pyplot as plt


import scipy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('name', type=str, metavar='NAME',
                    help='name of experiment')
parser.add_argument('--version', default=1, type=int, metavar='N',
                    help='version of model')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset on which to train the model')
parser.add_argument('--port-id', default=58472, type=int, metavar='N',
                    help='distributed training port number (default: 58472)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=0.024, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--tensorboard', action='store_true',
                    help='log training statistics to tensorboard')
parser.add_argument('--no-flip', action='store_true',
                    help='no random horizontal flips')

args, other_argv = parser.parse_known_args()
other_args = importlib.import_module('.' + args.name, 'models').get_parser().parse_args(other_argv)
args = argparse.Namespace(**vars(args), **vars(other_args))
# ----------------------------------------------------

Model = importlib.import_module('.' + args.name, 'models').Model
model = Model(args)
model = torch.nn.DataParallel(model).cuda()

#ckpt = torch.load('/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_v5/checkpoint.pth')
ckpt = torch.load('/home/ec3731/checkpoints/barlowtwins/notexture/factorize_avgpool_equivariant_all_bn_v5/checkpoint.pth')

model.load_state_dict(ckpt["model"])
print('epoch: ', ckpt["epoch"])
#model.load_state_dict(ckpt['state_dict'], strict=False)
model.fc = nn.Linear(2048, 2)
model = model.module

print(f"loaded barlowtwins faces resnet50")
model.eval()

# ----------------------------------------------------

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
#model.backbone.avgpool.register_forward_hook(get_activation('feats'))
model.avgpool.register_forward_hook(get_activation('feats'))

# ----------------------------------------------------
data_root = '/mnt/smb/locker/issa-locker/users/Josh/data/vbspl_texture_1ki/val/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
if crop_center:
    _trans = ttransforms.Compose([
            ttransforms.Resize(224),
            ttransforms.CenterCrop(112),
            ttransforms.Resize(224),
            ttransforms.ToTensor(),
            normalize,
        ])
else:
    _trans = ttransforms.Compose([
            ttransforms.Resize(224),
            ttransforms.CenterCrop(224),
            ttransforms.ToTensor(),
            normalize,
        ])

## read & sort img filenames
all_filenames = []
# read img filenames
for filename in os.listdir(valdir):
    if filename[-4:]=='.png':
        all_filenames.append(filename)

print('len(all_filenames): ', len(all_filenames))


# extract feats
FEATS = []
LABELS = []
# loop through batches
for idx, filename in enumerate(all_filenames):
    # read & transform img
    img = Image.open(os.path.join(valdir, filename)).convert("RGB")
    img_trans = _trans(img)

    # move to device
    inputs = img_trans.unsqueeze(0).cuda()
    # forward pass [with feature extraction]
    preds = model.backbone(inputs)

    FEATS.append(activation['feats'].cpu().squeeze().unsqueeze(0).numpy())
    if save_label:
        LABELS.append(outputs.cpu().squeeze().unsqueeze(0).numpy())
FEATS = np.concatenate(FEATS, axis=0)
if save_label:
    LABELS = np.concatenate(LABELS, axis=0)
print(FEATS.shape)

# save feats
filename = '/mnt/smb/locker/issa-locker/users/Eugénie/'+str(ind)+'_'+model_name+filename_postfix+'.pth'
torch.save(FEATS, filename)
print(f'saving to {filename}\n')
