#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


SideLight_42835=io.imread("../data/42835_chanel_1_SideLight_corrected.tif")
BottomLight_42835=io.imread("../data/42835_chanel_2_BottomLight_corrected.tif")
SideLight_42834=io.imread("../data/42834_chanel_1_SideLight_corrected.tif")
BottomLight_42834=io.imread("../data/42834_chanel_2_BottomLight_corrected.tif")

images_dict={"42835_chanel_1_SideLight":SideLight_42835,
             "42835_chanel_2_BottomLight":BottomLight_42835,
             "42834_chanel_1_SideLight":SideLight_42834,
             "42834_chanel_2_BottomLight":BottomLight_42834}


# In[10]:


file_name="42835_chanel_1_SideLight.tif"
Data=io.imread("../data/"+file_name)
Data.shape


# In[11]:


def Gaussian_smooth(img):
    from skimage.filters import gaussian
    img=gaussian(img,sigma=2)
    return util.img_as_uint(img)


def calculate_img_operation(img_stack,file_prefix):
    
    img_list=[img_stack[i] for i in range(img_stack.shape[0])]
    
    # count on several cores
    import multiprocess
    number_of_cpus = multiprocess.cpu_count(); 
    pool = multiprocess.Pool(processes=number_of_cpus)
    result = pool.map(Gaussian_smooth,img_list)
    
    # save to the folder
    file_name_prefix=file_prefix+"corrected_GaussSmooth.tif"

    smoothed=np.zeros_like(img_stack)
    for frame in range(img_stack.shape[0]):
        smoothed[frame]=result[frame]

    io.imsave("../data/"+file_name_prefix,smoothed,check_contrast=False,plugin='tifffile')


# In[12]:


for (name,images) in images_dict.items():
       calculate_img_operation(images,name)

