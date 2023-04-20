import json
import pickle
import numpy as np 
import matplotlib as plt#.pyplot as plt 

# different paths of checkpoints folders for analysis 
v1 = 'equivariant_all_bn_v1_v2'
v2 = 'equivariant_all_bn_v2_v2'
v3 = 'equivariant_all_bn_v3_v2'
faces_texture = 'factorize_avgpool_equivariant_all_bn_v5'


path = '/home/ec3731/checkpoints/barlowtwins/equivariant_all_bn_v1_v2/stats.txt'

list_lines = []

with open(path, 'r') as f: 
  data = json.load(f)

print('data type:', data.type)
"""with open(path, 'r') as f: 
  lines = f.readlines()
  list_lines.append(lines)
  
  
print('len: ', len(list_lines))

list_loss = []


for stats_epoch in list_lines[:1]: 
  # each step for the epoch 
  print(stats_epoch)
  for stats_step in stats_epoch[:5]: 
    print('step: ', stats_step)
    if stats_step[2:6] == str("epoch"): 
      print(stats_step)
    #list_loss.append(stats_step["loss"])
  
  
  #list_loss.append(stats_epoch["loss"])"""

  
plt.plot(list_loss)  
plt.save('/mnt/smb/locker/issa-locker/users/Eugénie/')
#print('content 1 line:', list_lines[0])
