from pathlib import Path
import argparse
import json
import math
import os
import torchvision.transforms as T
import random
import signal
import subprocess
import sys
import time
import importlib
import scipy.io.matlab as mio
from scipy import array
from PIL import Image
from torch import nn, optim
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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


layer_list = ['maxpool', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3','layer3.0', 'layer3.1', 'layer3.2', 
                 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'avgpool', 'fc']
    
def compute_curvature(X, n_frames=-1): 
  #print(np.isnan(X))
  if n_frames == -1:
      assert X.shape[0] >= 3
  else:
      assert n_frames >= 3 and n_frames <= X.shape[0] and isinstance(n_frames, int)
      delta_frames = (X.shape[0]-1) // (n_frames-1)
      X = X[::delta_frames][:n_frames]
  
  #X = X[~np.any(np.isnan(X))]
  if not np.any(np.isnan(X)): 
    X = X.reshape(X.shape[0], -1)

    print('after: ', X.shape)

    vecs = X[1:] - X[:-1] # displacement vectors

    vecs = vecs/np.linalg.norm(vecs, axis=-1, keepdims=True) # normalize the displacement vectors

    dots = np.einsum('ni,ni->n',vecs[1:],vecs[:-1]) # dot product between successive normalized displacement vectors
    #print(dots)
    angles = np.arccos(dots) # angles in radians
    angles = angles/np.pi # normalize to 1, since np.arccos outputs values between 0 and pi.
    #print(np.isnan(angles))
    return angles 
  
pixels = m(images.cuda().float())
  
value_pixels = compute_curvature(pixels.cpu().detach().numpy())
print('value: ', value_pixels)
  
values_pixels = [np.mean(value_pixels)] * len(layer_list)
