import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as im

save_images = False 

# --------------------------------------------------------------------------------------------------
# HK2 DATASET 
# --------------------------------------------------------------------------------------------------
path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

if save_images: 
  for i in range(img_array.shape[0]):
    plt.imshow(img_array[i], cmap='gray')
    im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hk2/image{}.png'.format(i), img_array[i]) 
    plt.show()

path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_classes.npy'
classes_array = np.load(path)
print('size: ', classes_array.shape)
print('classes:')
for i in range(10): 
  print(classes_array[i])
path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_class_to_idx.npy'
classes_to_idx_array = np.load(path, allow_pickle=True)
print('size: ', classes_to_idx_array.shape)

# --------------------------------------------------------------------------------------------------
# HVM DATASET 
# --------------------------------------------------------------------------------------------------
"""path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

if save_images: 
  for i in range(img_array.shape[0]):
    plt.imshow(img_array[i], cmap='gray')
    im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hvm/image{}.png'.format(i), img_array[i]) 
    plt.show()"""
  
# --------------------------------------------------------------------------------------------------
# MKTURK DATASET 
# --------------------------------------------------------------------------------------------------

"""path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/mkturk/test/images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

#img_array = img_array.reshape((432, 256, 256, 3))
if save_images: 
  for j in range(img_array.shape[0]): 
    for i in range(img_array.shape[1]):
      plt.imshow(img_array[j][i].reshape((256, 256, 3)))
      im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/mkturk_test/folder_{}/image{}.png'.format(j,i), img_array[j][i].reshape((256, 256, 3))) 
      plt.show()"""

