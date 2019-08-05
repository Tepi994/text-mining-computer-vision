# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:28:11 2019

@author: Juan Carlos Giron 1900607
"""

import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def binarization(img,threshold):
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    binary_image = np.where(gray_image > threshold, 255,0)
    cv2.imshow("Original Image",image)
    #print(np.mean(image))
    #print(binary_image)
    cv2.imwrite("binary_image.jpg",binary_image)
    transformed_image = cv2.imread("binary_image.jpg")     
    cv2.imshow('Binary Image',transformed_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
 

binarization("beatles.jpg",128)