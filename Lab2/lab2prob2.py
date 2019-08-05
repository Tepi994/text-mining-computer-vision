# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:33:10 2019

@author: jctep
"""
import numpy as np
import cv2
import sys


def operations(img1,img2,threshold):
    image_1 = cv2.imread(img1)
    image_2 = cv2.imread(img2)
    
    ###transforming input images to grayscale images and then binarization
    ##image 1
    
    blue_component1 = image_1[:,:,0]
    green_component1 = image_1[:,:,1]
    red_component1 = image_1[:,:,2]
    gray_image_1 = (0.11*blue_component1 + 0.59*green_component1 + 0.3*red_component1)
    
    binary_image_1 =  np.where(gray_image_1 > threshold, 255,0)
    print(binary_image_1)

 ##image 2
    
    blue_component2 = image_2[:,:,0]
    green_component2 = image_2[:,:,1]
    red_component2 = image_2[:,:,2]
    gray_image_2 = (0.11*blue_component2 + 0.59*green_component2 + 0.3*red_component2)
    
    binary_image_2 = np.where(gray_image_2 > threshold,255,0)
    
    ##addition
    #image_3 = binary_image_1 + binary_image_2
    image_3_add = cv2.add(binary_image_1,binary_image_2)
    
    ##substraction
    #image_3 = binary_image_1 - binary_image_2
    image_3_sub = cv2.subtract(binary_image_1,binary_image_2)
    
    ### logical xor
    xor = np.logical_xor(binary_image_1, binary_image_2)
    image_3_xor = np.where(xor > 0,255,0)
    
    ### logical andd
    #andd = np.logical_and(binary_image_1,binary_image_2)
    image_3_andd = np.where((binary_image_1*binary_image_2>0), 255,0)
    
     ### logical or
    orr = np.logical_or(binary_image_1,binary_image_2)
    image_3_orr = np.where(orr > 0,255,0)

    
   
    cv2.imwrite("image_add.jpg",image_3_add)
    image_add = cv2.imread("image_add.jpg")     
    cv2.imshow('Addition of Images',image_add)
    
    cv2.imwrite("image_sub.jpg",image_3_sub)
    image_sub = cv2.imread("image_sub.jpg")     
    cv2.imshow('Subtraction of Images',image_sub)
    
    cv2.imwrite("image_xor.jpg",image_3_xor)
    image_xor = cv2.imread("image_xor.jpg")     
    cv2.imshow('Logical XOR of Images',image_xor)
    
    cv2.imwrite("image_and.jpg",image_3_andd)
    image_andd = cv2.imread("image_and.jpg")     
    cv2.imshow('Logical AND of Images',image_andd)
    
    cv2.imwrite("image_or.jpg",image_3_orr)
    image_orr = cv2.imread("image_or.jpg")     
    cv2.imshow('Logical OR of Images',image_orr)
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
    
operations("red_triangle.png","red_circle.png",200)