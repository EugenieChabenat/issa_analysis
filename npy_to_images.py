import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as im

path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

for i in range(img_array.shape[0]):
  plt.imshow(img_array[i], cmap='gray')
  im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hk2/image{}.png'.format(i), img_array[i]) 
  plt.show()
