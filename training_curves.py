import json
import pickle
import numpy as np 
import matplotlib.pyplot as plt 

# different paths of checkpoints folders for analysis 
v1 = 'equivariant_all_bn_v1_v2'
v2 = 'equivariant_all_bn_v2_v2'
v3 = 'equivariant_all_bn_v3_v2'
faces_texture = 'factorize_avgpool_equivariant_all_bn_v5'


path = '/home/ec3731/checkpoints/barlowtwins/equivariant_all_bn_v3_v2/stats.txt'

list_lines = []

#with open(path) as f: 
#  data = f.read()
  

#print('data type:', data.type)
with open(path, 'r') as f:
  lines = f.readlines()
  
print('len: ', len(lines))

list_lines = []
for line in lines: 
  if line[:4] != "main": 
    list_lines.append(line)

print('len: ', len(list_lines))
list_losses = []
for line in list_lines:
  ind = line.find("loss")
  loss = line[ind+7: ind+17]
  #print(loss)
  if loss: 
    if loss != "or-weig" and loss != "lor-wei" and loss != "olor-we" and loss !="" and loss.find('}')==-1 and loss != "100 90." and loss !='': 
      try: 
        loss_ = float(loss)
        if loss_ <300: 
          list_losses.append(loss_)
      except Exception as e: 
        print(e)
          
          
print('final len: ', len(list_losses))

plt.plot(list_losses)  
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/traininglossv1.png')

"""list_loss = []


for stats_epoch in list_lines[:1]: 
  # each step for the epoch 
  print(stats_epoch)
  for stats_step in stats_epoch[:5]: 
    print('step: ', stats_step)
    if stats_step[2:6] == str("epoch"): 
      print(stats_step)
    #list_loss.append(stats_step["loss"])
  
  
  #list_loss.append(stats_epoch["loss"])

  
plt.plot(list_loss)  
plt.save('/mnt/smb/locker/issa-locker/users/Eugénie/trainingloss.png')
#print('content 1 line:', list_lines[0])"""
