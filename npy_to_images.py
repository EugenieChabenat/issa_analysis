import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as im

save_images = False 
hk2 = False 
hvm = False
mkturk = True 
# --------------------------------------------------------------------------------------------------
# HK2 DATASET 
# --------------------------------------------------------------------------------------------------
if hk2: 
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
if hvm: 
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_images.npy'
  img_array = np.load(path)
  print('size: ', img_array.shape)
  
  if save_images: 
    for i in range(img_array.shape[0]):
      plt.imshow(img_array[i], cmap='gray')
      im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/hvm/image{}.png'.format(i), img_array[i]) 
      plt.show()
  
  print('\nobject categories')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_categories.npy'
  obj_array = np.load(path)
  print('size: ', obj_array.shape)
  print('categories examples: ')
  for i in range(10): 
    print(obj_array[i])
  
  print('\nobject category to idx')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_category_to_idx.npy'
  obj_array = np.load(path, allow_pickle=True)
  print('size: ', obj_array.shape)
  
  print('\nobject class to idx')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_class_to_idx.npy'
  obj_array = np.load(path, allow_pickle=True)
  print('size: ', obj_array.shape)
  
  print('\nobject classes')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_classes.npy'
  obj_array = np.load(path)
  print('size: ', obj_array.shape)
  print('classes examples: ')
  for i in range(10): 
    print(obj_array[i])
    
    
    
  print('\nobject poses')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_poses.npy'
  obj_array = np.load(path)
  print('size: ', obj_array.shape)
  print('poses examples: ')
  for i in range(10): 
    print(obj_array[i])
    
    
  print('\nobject positions')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_positions.npy'
  obj_array = np.load(path)
  print('size: ', obj_array.shape)
  
  print('\nobject sizes')
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/HVM/HVM_obj_sizes.npy'
  obj_array = np.load(path)
  print('size: ', obj_array.shape)
  
  
# --------------------------------------------------------------------------------------------------
# MKTURK DATASET 
# --------------------------------------------------------------------------------------------------
if mkturk: 
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/mkturk/test/images.npy'
  img_array = np.load(path)
  print('size: ', img_array.shape)

  #img_array = img_array.reshape((432, 256, 256, 3))
  if save_images: 
    for j in range(img_array.shape[0]): 
      for i in range(img_array.shape[1]):
        plt.imshow(img_array[j][i].reshape((256, 256, 3)))
        im.imsave('/mnt/smb/locker/issa-locker/users/Eugénie/images/mkturk_test/folder_{}/image{}.png'.format(j,i), img_array[j][i].reshape((256, 256, 3))) 
        plt.show()
  
  path = '/mnt/smb/locker/issa-locker/users/hc3190/datasets/imagesets/mkturk/test/target_labels.json'
  labels = np.load(path)
  print('size: ', labels.shape)
