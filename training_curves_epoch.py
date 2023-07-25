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
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/equivariant_all_bn_v1_v2/equivariant_all_bn_v1_v2/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_v5/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_injection_v1/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/factorize_avgpool_equivariant_all_bn_injection_v1/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/notexture/original_v5/stats.txt'
#path = '/home/ec3731/checkpoints/barlowtwins/equivariant_all_bn_v3_v2/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/backbone/factorize_avgpool_equivariant_all_bn_injection_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/faces_notexture/original_v5/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/pretrained/factorize_avgpool_equivariant_all_bn_v5/factorize_avgpool_equivariant_all_bn_v5/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/v1_no_injection_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/v2_no_injection_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/v4_no_injection_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/IT_no_injection_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/random_injection/new_injection_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/random_injection/new_injection_v2_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/random_injection/new_injection_v4_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/random_injection/new_injection_IT_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/convolution_injection/new_injection_conv_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/convolution_injection/new_injection_conv_v2_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/convolution_injection/new_injection_conv_v4_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/convolution_injection/new_injection_conv_IT_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_conv_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_conv_v2_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_conv_v4_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_conv_IT_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_v2_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_v4_v1/stats.txt'
path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_IT_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_control_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_control_v2_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_control_v4_v1/stats.txt'
#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/no_projector_linear_control_IT_v1/stats.txt'

#path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/original_v2/stats.txt'



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
inds = [0]
current_e = 0 
ind = 0 
for line in list_lines: 
  #print(line)
  if line["epoch"] <= 29:
    epochs.append(line["epoch"])
    steps.append(line["step"])
    losses.append(line["loss"])

  #while line["epoch"] <= 29: 
    if line["epoch"] == current_e: 
      ind +=1 
    else: 
      if inds: 
        inds.append(ind+inds[-1])
      else : 
        inds.append(ind)
      ind = 0 
      current_e =line["epoch"]
print(current_e) 
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#labels = np.arange(0, 29, step=1)
new_labels = []
new_inds= []  
i = 0
for element in labels: 
  
  if element == 0 or element %5 ==0 or element ==29: 
    new_labels.append(element)
    new_inds.append(inds[i])
  i +=1
    
print('last loss: ', losses[-1])

plt.figsize=(60, 30)
plt.plot(losses, color = 'k')  
plt.xticks(ticks= new_inds, labels = new_labels)
#plt.tight_layout()
plt.title('training curve ')
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/prediction_loss/v1_noinj.png')
    

