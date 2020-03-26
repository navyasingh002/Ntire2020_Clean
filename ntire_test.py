#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.layers import (Activation, Conv3D, Dropout, Convolution2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_absolute_error
from keras.models import Sequential,load_model
from keras.optimizers import Adam
import os, math
import tensorflow as tf
from skimage import io, img_as_uint, img_as_ubyte
import hdf5storage
from IPython.display import clear_output
from keras.models import load_model,Model
import cv2 
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import argparse
# io.use_plugin('pil')
from keras.models import load_model, Model
import tensorflow as tf
import matplotlib.pyplot as plt
from EvalMetrics import computeMRAE
from keras.layers import Input, MaxPool2D
from keras_octave_conv import OctaveConv2D
# import utils_latest


# In[9]:


track='real3'
models=os.listdir('models_rgb2hs/'+track)
models.sort()
models


# In[10]:


# f='dataset/NTIRE2020_Test_Clean/'
# # f2='dataset/NTIRE2020_Validation_Spectral_Clean/'
# l1=10
f='dataset/NTIRE2020_Test_RealWorld/'
# f2='dataset/NTIRE2020_Validation_Spectral_RealWorld/'
l1=14
paths=os.listdir(f)
paths.sort()
paths


# In[144]:


def oct_srcnn():
    inputs = Input(shape=(256,256, 3))
    high, low = OctaveConv2D(filters=64, kernel_size=3)(inputs)

    high, low = Activation('relu')(high), Activation('relu')(low)
    high, low = MaxPool2D()(high), MaxPool2D()(low)
    high, low = OctaveConv2D(filters=32, kernel_size=3)([high, low])

    high, low = Activation('relu')(high), Activation('relu')(low)
    high, low = MaxPool2D()(high), MaxPool2D()(low)
    conv = OctaveConv2D(filters=31, kernel_size=3, ratio_out=0.0)([high, low])
    Output=conv
    model = Model(inputs=inputs, outputs=Output)
    return model
m=oct_srcnn()


# In[145]:


import scipy.io
dic = scipy.io.loadmat(f2+'ARAD_HS_0451.mat')
dic['cube'].shape


# In[11]:


for j in range(len(models)):
#     if models[j]==best_model:
#         print("skipped_best_model")
#         continue;
    print(models[j])
    model = load_model('models_rgb2hs/'+track+'/'+models[j])
#     model=oct_srcnn()
#     model.load_weights('models_rgb2hs/'+track+'/'+models[j])
#     print(model.summary())
    try:
        os.makedirs('Validation_results/'+track+'/'+models[j][:-3])
    except OSError as error: 
        print(error)
    mrae=0;
    for i in range(len(paths)):
        im = cv2.imread(f + paths[i])
        plt.imshow(im)
#         im = cv2.resize(im, (256,256), interpolation=cv2.INTER_CUBIC)
        im=im*1.0/im.max()
        x=np.expand_dims(im, axis=0)
#         print(x.shape)
        y=model.predict(x)
        y=y[0]
#         y = cv2.resize(y, (512,482), interpolation=cv2.INTER_CUBIC)
        print(y.shape)
        mat={}
        mat[u'cube']=y
        hdf5storage.write(mat, '.', 'Validation_results/'+track+'/'+models[j][:-3]+'/'+paths[i][:-l1]+'.mat', matlab_compatible=True)
#         dic = scipy.io.loadmat(f2+paths[i][:-l1]+'.mat')
#         mrae=mrae+computeMRAE(dic['cube'],mat['cube']);
#     mrae=mrae*1.0/len(paths)
#     print(mrae)


# In[12]:


im=y
# im = cv2.resize(im, (512,482), interpolation=cv2.INTER_AREA)
print(im.shape)
plt.imshow(im[:,:,27:30])


# In[67]:


plt.imshow(y[0,:,:,30])


# In[43]:


2*1.0/3


# In[ ]:




