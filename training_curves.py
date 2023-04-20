import json
import pickle
import numpy as np 
import matplotlib.pyplot as plt 

# different paths of checkpoints folders for analysis 
v1 = 'equivariant_all_bn_v1_v2'
v2 = 'equivariant_all_bn_v2_v2'
v3 = 'equivariant_all_bn_v3_v2'
faces_texture = 'factorize_avgpool_equivariant_all_bn_v5'


path = '/home/ec3731/checkpoints/barlowtwins/equivariant_all_bn_v1_v2/stats.txt'

list_lines = []
with open(path, 'r') as f: 
  lines = f.readlines()
  list_lines.append(lines)
  
  
print('len: ', len(list_lines))

list_loss = []
for stats_epoch in list_lines: 
  list_loss.append(stats_epoch["loss"])

  
plt.plot(list_loss)  
#print('content 1 line:', list_lines[0])
