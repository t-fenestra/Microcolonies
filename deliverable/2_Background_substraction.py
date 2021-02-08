#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import skimage
from skimage import util
import skimage.io as io
from skimage import exposure
import cv2 as cv

import sys

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


file_name="42835_chanel_1_SideLight.tif"
Data=io.imread("../data/"+file_name)
Data.shape


# ## Function for background substraction

# In[3]:


def subtract_med_background(img, d=51): 
    from skimage.morphology import rectangle
    from skimage.filters import gaussian,median
    
    img_conv=util.img_as_float(img)
    
    # median filter with smoothed with gaussian
    selem=rectangle(d,d)
    bg = median(img,selem)
    bg=gaussian(bg,sigma=3)
    
    # substract background
    img_corrected=img_conv-bg
    img_corrected=np.where(img_corrected<0,0,img_corrected)
    
    return util.img_as_uint(img_corrected),util.img_as_uint(bg)


# ## Calculate in parallel by  multiprocess

# In[4]:


import multiprocess
number_of_cpus = multiprocess.cpu_count(); number_of_cpus

# make a list with images for multiprocessor
img_list=[Data[i] for i in range(Data.shape[0])]

pool = multiprocess.Pool(processes=number_of_cpus)
result = pool.map(subtract_med_background,img_list)


# ## Save results
# 

# In[5]:


bg_file_name=file_name.split(".")[0]+"_"+"bg.tif"
corrected_file_name=file_name.split(".")[0]+"_"+"corrected.tif"


bg=np.zeros_like(Data)
corrected_bg=np.zeros_like(Data)

for frame in range(Data.shape[0]):
    corrected_bg[frame],bg[frame]=result[frame]

io.imsave("../data/"+corrected_file_name,corrected_bg,check_contrast=False,plugin='tifffile')
io.imsave("../data/"+bg_file_name,bg,check_contrast=False,plugin='tifffile')

