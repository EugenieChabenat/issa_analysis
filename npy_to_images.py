import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as im


# --------------------------------------------------------------------------------------------------
# HK2 DATASET 
# --------------------------------------------------------------------------------------------------
"""path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

for i in range(img_array.shape[0]):
  plt.imshow(img_array[i], cmap='gray')
  im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hk2/image{}.png'.format(i), img_array[i]) 
  plt.show()"""

  
# --------------------------------------------------------------------------------------------------
# HVM DATASET 
# --------------------------------------------------------------------------------------------------
"""path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

for i in range(img_array.shape[0]):
  plt.imshow(img_array[i], cmap='gray')
  im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hvm/image{}.png'.format(i), img_array[i]) 
  plt.show()"""
  
# --------------------------------------------------------------------------------------------------
# MKTURK DATASET 
# --------------------------------------------------------------------------------------------------

path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/mkturk/test/images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

img_array = img_array.reshape((432, 256, 256, 3))

for i in range(img_array.shape[0]):
  plt.imshow(img_array[i], cmap='gray')
  im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/mkturk_test/image{}.png'.format(i), img_array[i]) 
  plt.show()

