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

"""import datasets
import transforms
import constants"""

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
# -- 

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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def main():
  
    # ------------------------------------------------------------------------------------------------------------------
    # not in notebook 
    # ------------------------------------------------------------------------------------------------------------------
    
    args, other_argv = parser.parse_known_args()
    other_args = importlib.import_module('.' + args.name, 'models').get_parser().parse_args(other_argv)
    args = argparse.Namespace(**vars(args), **vars(other_args))

    # define different paths
    args.data = Path(constants.IMAGENET_PATH)
    if args.dataset == 'faces': 
      args.data = Path(constants.FACES_PATH)
    args.checkpoint_dir = Path(constants.CHECKPOINTS_PATH) / (args.name + f'_v{args.version}')
    #args.checkpoint_dir =  '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_v5'
    #args.checkpoint_dir =  '/home/ec3731/checkpoints/barlowtwins/notexture/original_v5'
    args.checkpoint_dir = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/notexture/original_v5'
    args.tensorboard_dir = Path(constants.TENSORBOARD_PATH) / (args.name + f'_v{args.version}')
    
    print('PATHS: ')
    print('data: ', args.data)
    print('checkpoint: ', args.checkpoint_dir)
    print('tensorboard: ', args.tensorboard_dir)
    
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:{args.port_id}'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f'tcp://localhost:{args.port_id}'
        args.world_size = args.ngpus_per_node
    print(args.dist_url)
    args.ngpus_per_node =1
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

    #gpu = args.ngpus_per_node
    
    # ------------------------------------------------------------------------------------------------------------------
    
# ------------------------------------------------------------------------------------------------------------------      
def load_model(model_name, args): 
  print(f"loading {model_name}")
  Model = importlib.import_module('.' + model_name, 'models').Model
  model = Model(args)
  model = torch.nn.DataParallel(model).cuda()
  
  #ckpt = torch.load('/home/ec3731/checkpoints/barlowtwins/notexture/original_v5/checkpoint.pth')
  ckpt = torch.load('/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/backbone/factorize_avgpool_equivariant_all_bn_injection_v1/checkpoint.pth')
  #ckpt = torch.load('/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/equivariant_all_bn_v2_v2/checkpoint.pth')
         
  model.load_state_dict(ckpt["model"])
  #print('epoch: ', ckpt["epoch"])
  #model.load_state_dict(ckpt['state_dict'], strict=False)
  model.fc = nn.Linear(2048, 2)
  model = model.module
  print("loaded model")
  model.eval()
  
  print('model: \n', model)
  return model 


# ------------------------------------------------------------------------------------------------------------------
    
def main_worker(gpu, args):
 
  #model_names_list = ['resnet50-barlowtwins-faces']
  exp_names_list = ['stim_matrix.npy']
  model_name = 'factorize_avgpool_equivariant_all_bn_injection'
  # load model
  model = load_model(model_name, args)

  # add hook 
  activation = {}
  def get_activation(name):
      def hook(model, input, output):
          activation[name] = output.detach()
      return hook
  model.backbone.avgpool.register_forward_hook(get_activation('feats'))

  # load data

  # extract feats
  """FEATS = []
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
      #preds = model(inputs)

      FEATS.append(activation['feats'].cpu().squeeze().unsqueeze(0).numpy())
      if save_label:
          LABELS.append(outputs.cpu().squeeze().unsqueeze(0).numpy())
  FEATS = np.concatenate(FEATS, axis=0)
  if save_label:
      LABELS = np.concatenate(LABELS, axis=0)
  print(FEATS.shape)

  # save feats
  #filename = exp_name+'_'+model_name+filename_postfix+'.pth'
  #filename = '/mnt/smb/locker/issa-locker/users/Eugénie/'+'_'+model_name+filename_postfix+'.pth'
  filename = '/mnt/smb/locker/issa-locker/users/Eugénie/'+str(ind)+ 'original_correlation.pth'
  torch.save(FEATS, filename)
  print(f'saving to {filename}\n')

  if save_label:
      filename = exp_name+'_'+model_name+filename_postfix+'_label.pth'
      filename = '/mnt/smb/locker/issa-locker/users/Eugénie/'+'_'+model_name+filename_postfix+'_label.pth'
      filename = '/mnt/smb/locker/issa-locker/users/Eugénie/'+str(ind)+ 'original_correlation.pth'

      torch.save(LABELS, filename)"""
# ------------------------------------------------------------------------------------------------------------------
 
           
def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


def save_args(filename='versions.txt'):
    with open(filename, "a") as f:
        data = np.array([['python', f'{" ".join(sys.argv)}']])
        np.savetxt(f, data, fmt='%s', delimiter=' ')


def get_transform(args):
    if args.no_flip:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(p=1.0),
            transforms.Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(p=0.1),
            transforms.Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(p=1.0),
            transforms.Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(p=0.1),
            transforms.Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transforms.Multiplex([transform, transform_prime])


if __name__ == '__main__':
    main()
