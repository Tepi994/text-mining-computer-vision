# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:47:25 2019

@author: jctep
"""
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def histogram_eq(img):
    image = cv2.imread(img)
    or_image = cv2.imread(img)
    cv2.imshow("Original Image",or_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("",image)
    height = image.shape[0]
    width = image.shape[1]
    #image_shape = image.shape()

    
    ####gray_scale histogram####
    gray_scale_counts, gray_scale_bins = np.histogram(image, range(256))
    plt.bar(gray_scale_bins[:-1] - 0.5, gray_scale_counts, width=1, edgecolor='none',color ='gray')
    plt.axvline(image.mean(), color='k', linestyle='dashed', linewidth=2)
    plt.xlim([-0.5, 255.5])
    plt.title("Weighted Grayscale Channel Histogram")
    plt.show()
    
    
    counts, bins = np.histogram(image, bins = np.arange(0,256))
    cdf_un = np.cumsum(counts, dtype=float)
    cdf=cdf_un/np.amax(cdf_un)
    out = np.zeros((height,width,1))
    
    for i in range(0,height):
        for j in range(0,width):
            pixel = image[i,j]
            if pixel == 255:
                out[i,j] = 255
            else:
                
            #print(pixel)
                out[i,j] = round(255*cdf[pixel],0)
            #out[i,j] = 255*cdf[pixel]
            #out[i,j]= np.where(pixel==255,np.multiply(255,1),np.multiply(255,cdf[pixel]))
     
    
     ####equalized histogram####
    out_scale_counts, out_scale_bins = np.histogram(out, range(256))
    plt.bar(out_scale_bins[:-1] - 0.5, out_scale_counts, width=1, edgecolor='none',color ='gray')
    plt.axvline(out.mean(), color='k', linestyle='dashed', linewidth=2)
    plt.xlim([-0.5, 255.5])
    plt.title("Equalized Channel Histogram")
    plt.show()
    
    
    
    cv2.imwrite("g.jpg",out)
    gray_image = cv2.imread("g.jpg")     
    cv2.imshow('Equalized image',gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
    
            
histogram_eq("car.jpg")
