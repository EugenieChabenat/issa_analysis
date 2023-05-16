#!/usr/bin/env python
# this scripts converts a matlab mat-file to an importable python-file

import scipy.io.matlab as mio
from scipy import array
import re, pprint
import pandas as pd
import numpy as np

def mat_to_npy(matfile=None, pyfile=None):
    if not pyfile:
        pyfile = "stim_matrix.npy"
    matfile = '/mnt/smb/locker/issa-locker/users/Eugénie/datasets/stim_matrix.mat'
    mat = mio.loadmat(matfile)
    #print(mat.keys())
    
    mat_info = mio.loadmat("/mnt/smb/locker/issa-locker/users/Eugénie/datasets/stim_info.mat")
    print(mat_info)
    print(mat_info.keys())
    
    print('articifial labels: ', mat_info['artificial_movie_labels'][0][0].type)
    #print('natural labels: ', mat_info['natural_movie_labels'])
    #print('contrast labels: ', mat_info['contrast_movie_labels'])
    
    
    # read mkturk test 
    ##data = np.load('../Desktop/images.npy')
    #print('data shape:', data.shape)
    #rust = np.load('../Desktop/rust_images.npy')
    #print('rust shape:', rust.shape)
    print('stim shape: ', mat['stim_matrix'].shape)
    #np.save('stim_matrix', mat['stim_matrix'])
    print('final shape: ', mat['stim_matrix'][:, :, 7, :, :, :].shape)
    """for i in range(10): 
        mat_ = mat['stim_matrix'][:, :, i, :, :, :].reshape(2,11, 3,  512, 512)
        print('after reshape: ', mat_.shape)
    #np.save('/mnt/smb/locker/issa-locker/users/Eugénie/datasets/stim_matrix', mat['stim_matrix'][:, :, 0, :, :, :])
        np.save('/mnt/smb/locker/issa-locker/users/Eugénie/datasets/stim_matrix'+str(i), mat_)"""
    
    
    """print(mat['artificial_movie_labels'])
    
    dataframe = pd.DataFrame(columns = ['artificial_movie_labels', 'contrast_movie_labels', 'image_paths', 'natural_movie_labels', 
                                        'stim_matrix', 'stim_matrix_blurred'])
    dataframe['artificial_movie_labels'] = mat['artificial_movie_labels']
    dataframe['constrast_movie_labels'] = mat['contrast_movie_labels']
    dataframe['image_paths'] = mat['image_paths']
    dataframe['natural_movie_labels'] = mat['natural_movie_labels']
    dataframe['stim_matrix'] = mat['stim_matrix']
    dataframe['stim_matrix_blurred'] = mat['stim_matrix_blurred']
    print(dataframe.head())"""
    
    """fd = file(pyfile,"w")
    fd.write("# file autogenerated by mat_to_py.py from %s\n\n"%matfile)"""
    
    """"for k,v in mat.items():
        if k.startswith("__"):
            continue
        try:
            if 1 in v.shape:
                v=v.flatten()
            l = v.tolist()
            fd.write(k+" = ")
            pprint.pprint(l, fd, indent=4)
            #print("processed %s"%k)
            print('process')
        except:
            print("error processing %s")


    fd.close()"""


if __name__=="__main__":
    import sys
    try:
        matfile = sys.argv[1]
    except:
        print("convert mat files to python files")
        print("usage: matfile inputfile.mat [outputfile.py]")
    try:
        pyfile = sys.argv[2]
    except:
        pyfile = None

    mat_to_npy()
