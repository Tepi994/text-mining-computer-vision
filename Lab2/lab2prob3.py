# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:35:53 2019

@author: jctep
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def histogram_3D(image,a_zim=-60,elevation=30):
    img = cv2.imread(image)
    ##turning image into grayscale
    gray_image = (0.11*img[:,:,0] + 0.59*img[:,:,1] + 0.3*img[:,:,2])
    
    ##defining a grid to plot the image, based on the pixel dimensions of the image
    xx, yy = np.mgrid[0:gray_image.shape[0], 0:gray_image.shape[1]]
    
    
    fig = plt.figure()
    ax = Axes3D(fig)

    # Get current rotation angle
     #print(ax.azim)
    #ax.view_init(azim=0, elev=90)

    # Set rotation angle to 30 degrees
    ax.view_init(azim=a_zim,elev=elevation)

    ax.plot_surface(xx,yy,gray_image,rstride=1,cstride=1,color="k",alpha=0.75,cmap=plt.cm.gray,
            linewidth=0)

    plt.title(image)    
    plt.show()
    
    
histogram_3D("beatles.jpg")
histogram_3D("beatles.jpg",a_zim=0,elevation=90)
