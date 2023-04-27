import json
import pickle
import numpy as np 
import matplotlib.pyplot as plt 

# different paths of checkpoints folders for analysis 
v1 = 'equivariant_all_bn_v1_v2'
v2 = 'equivariant_all_bn_v2_v2'
v3 = 'equivariant_all_bn_v3_v2'
faces_texture = 'factorize_avgpool_equivariant_all_bn_v5'

# path to stats.txt file
path = '/home/ec3731/checkpoints/barlowtwins/equivariant_all_bn_v3_v2/stats.txt'
path = '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_v5/stats.txt'
path = '/home/ec3731/checkpoints/barlowtwins/notexture/factorize_avgpool_equivariant_all_bn_v5/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_injection_v1/stats.txt'
# -- 

list_lines = []

#with open(path) as f: 
#  data = f.read()

# read file, remove lines that are not epochs, redump it and then read as dict 

#print('data type:', data.type)
with open(path, 'r') as f:
  lines = f.readlines()
  
print('len: ', len(lines))

list_lines = []

for line in lines: 
  if line[0] == "{": 
    list_lines.append(json.loads(line))
  
 
print('len: ', len(list_lines))
epochs = []
steps = []
losses = []
inds = []
current_e = 0 
ind = 0 
for line in list_lines: 
  print(line)
  
  epochs.append(line["epoch"])
  steps.append(line["step"])
  losses.append(line["loss"])
  
  if line["epoch"] == current_e: 
    ind +=1 
  else: 
    inds.append(ind)
    ind = 0 
    current_e =line["epoch"]

print(inds)
plt.plot(steps, losses)  
print(len(losses)/30)
#ticks = [0, 
plt.xticks(np.arange(0, 29, step=1000))
plt.title('training curve imagenet factorize avgpool equivariant')
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/training_curves/losstest1.png')


          
          
"""print('final len: ', len(list_losses))
print('debut: ', list_losses[0])
#print('fin: ', list_losses[9024])
print('fin: ', list_losses[7500])
      
plt.plot(list_losses)  
plt.title('training curve imagenet factorize avgpool equivariant')
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/training_curves/trainingloss_v1_f_a_e.png')"""

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
