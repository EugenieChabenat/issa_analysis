import numpy as np

from matplotlib import pyplot as plt

path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HK2/HK2_images.npy'
img_array = np.load(path)
print('size: ', img_array.shape)

for i in range(3):
  plt.imshow(img_array[i], cmap='gray')
  plt.show()
